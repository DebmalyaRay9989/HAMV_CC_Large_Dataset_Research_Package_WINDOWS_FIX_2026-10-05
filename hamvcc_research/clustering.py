from __future__ import annotations
import time
import numpy as np, pandas as pd
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score, adjusted_rand_score, normalized_mutual_info_score

try:
    from threadpoolctl import threadpool_limits
except Exception:  # pragma: no cover
    threadpool_limits = None


def _threads(cfg):
    try:
        return max(1, int(cfg.get("cluster_threads", 1)))
    except Exception:
        return 1


def fit_kmeans(X, k, seed, cfg, purpose="final"):
    """Fit the configured clustering algorithm with bounded BLAS/OpenMP threads.

    purpose is one of: k_selection, stability, bootstrap, coassociation, final.
    Separate settings avoid spending expensive initialization/iterations on exploratory scans.
    """
    purpose = str(purpose)
    prefix = {"k_selection":"[CLUSTER/K]", "stability":"[CLUSTER/STABILITY]",
              "bootstrap":"[CLUSTER/BOOT]", "coassociation":"[CLUSTER/COASSOC]",
              "final":"[CLUSTER/FINAL]"}.get(purpose, "[CLUSTER]")
    method = cfg.get("cluster_method", "minibatch_kmeans")
    if purpose == "k_selection":
        n_init = int(cfg.get("cluster_n_init_k_selection", 2))
        max_iter = int(cfg.get("cluster_max_iter_k_selection", 100))
        batch_size = int(cfg.get("cluster_batch_size_k_selection", 512))
    elif purpose in ("stability", "bootstrap"):
        n_init = int(cfg.get("cluster_n_init_stability", cfg.get("cluster_n_init", 10)))
        max_iter = int(cfg.get("cluster_max_iter_stability", cfg.get("cluster_max_iter", 300)))
        batch_size = int(cfg.get("cluster_batch_size_stability", cfg.get("cluster_batch_size", 2048)))
    elif purpose == "coassociation":
        n_init = int(cfg.get("cluster_n_init_coassociation", 2))
        max_iter = int(cfg.get("cluster_max_iter_coassociation", 100))
        batch_size = int(cfg.get("cluster_batch_size_coassociation", 512))
    else:
        n_init = int(cfg.get("cluster_n_init_final", cfg.get("cluster_n_init", 10)))
        max_iter = int(cfg.get("cluster_max_iter_final", cfg.get("cluster_max_iter", 300)))
        batch_size = int(cfg.get("cluster_batch_size_final", cfg.get("cluster_batch_size", 2048)))

    if method == "minibatch_kmeans":
        model = MiniBatchKMeans(
            n_clusters=int(k), random_state=int(seed), n_init=n_init,
            max_iter=max_iter, batch_size=max(batch_size, int(k)),
            reassignment_ratio=float(cfg.get("cluster_reassignment_ratio", 0.01)),
            max_no_improvement=int(cfg.get("cluster_max_no_improvement", 20)),
            tol=float(cfg.get("cluster_tol", 1e-3)),
        )
    elif method == "kmeans":
        model = KMeans(n_clusters=int(k), random_state=int(seed), n_init=n_init, max_iter=max_iter)
    else:
        raise ValueError(f"Unsupported cluster_method={method!r}")

    X = np.asarray(X, dtype=np.float32)
    if not np.isfinite(X).all():
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    if threadpool_limits is None:
        return model.fit_predict(X)
    with threadpool_limits(limits=_threads(cfg)):
        return model.fit_predict(X)


def safe_internal_metrics(X, labels, max_samples=10000, seed=42):
    labels = np.asarray(labels); keep = labels != -1
    ids = np.unique(labels[keep])
    if len(ids) < 2 or keep.sum() < 3:
        return {"silhouette": np.nan, "CH": np.nan, "DB": np.nan}
    idx = np.where(keep)[0]
    if len(idx) > max_samples:
        idx = np.random.default_rng(seed).choice(idx, max_samples, replace=False)
    z = np.asarray(X[idx], dtype=np.float32); y = labels[idx]
    return {"silhouette": float(silhouette_score(z, y)),
            "CH": float(calinski_harabasz_score(z, y)),
            "DB": float(davies_bouldin_score(z, y))}


def select_k(X, kmin, kmax, seed=42, max_samples=10000, cfg=None):
    cfg = dict(cfg or {})
    n = len(X); lo = max(2, int(kmin)); hi = min(int(kmax), max(2, n - 1))
    rows = []
    rng = np.random.default_rng(seed)
    idx = np.arange(n)
    if n > max_samples:
        idx = np.sort(rng.choice(n, max_samples, replace=False))
    Z = np.asarray(X[idx], dtype=np.float32)
    print(f"[CLUSTER/K] testing k={lo}..{hi} on n={len(Z)} samples", flush=True)
    t0 = time.perf_counter()
    for pos, k in enumerate(range(lo, hi + 1), 1):
        t = time.perf_counter()
        lab = fit_kmeans(Z, k, seed, cfg, purpose="k_selection")
        m = safe_internal_metrics(Z, lab, max_samples=len(Z), seed=seed)
        rows.append({"k": k, **m, "seconds": round(time.perf_counter() - t, 3)})
        print(f"[CLUSTER/K] {pos}/{hi-lo+1}: k={k} silhouette={m['silhouette']:.5f} ({rows[-1]['seconds']:.2f}s)", flush=True)
    df = pd.DataFrame(rows)
    if df.empty:
        return 2, df
    best = int(df.loc[df["silhouette"].fillna(-np.inf).idxmax(), "k"])
    print(f"[CLUSTER/K] selected k={best}; elapsed={time.perf_counter()-t0:.1f}s", flush=True)
    return best, df


def multi_seed_stability(X, k, seeds, cfg, max_samples=20000):
    X = np.asarray(X, dtype=np.float32)
    if len(X) > max_samples:
        rng = np.random.default_rng(int(seeds[0]))
        idx = np.sort(rng.choice(len(X), max_samples, replace=False)); Z = X[idx]
    else:
        Z = X
    labs = []
    t0 = time.perf_counter(); seeds = list(seeds)
    print(f"[CLUSTER/STABILITY] k={k}, seeds={seeds}, n={len(Z)}", flush=True)
    for pos, s in enumerate(seeds, 1):
        t = time.perf_counter(); labs.append(fit_kmeans(Z, k, s, cfg, purpose="stability"))
        print(f"[CLUSTER/STABILITY] {pos}/{len(seeds)} seed={s} ({time.perf_counter()-t:.2f}s)", flush=True)
    pairs = [adjusted_rand_score(labs[i], labs[j]) for i in range(len(labs)) for j in range(i+1, len(labs))]
    return (float(np.mean(pairs)) if pairs else 1.0), labs


def bootstrap_stability(X, ref, k, cfg, B=10, frac=.8, seed=42, max_samples=20000):
    rng = np.random.default_rng(seed); aris=[]; n=len(X); m=max(k,int(frac*n))
    print(f"[CLUSTER/BOOT] B={B}, fraction={frac:.2f}, sample_size={min(m,max_samples)}", flush=True)
    for b in range(B):
        idx=rng.choice(n,m,replace=False)
        if len(idx)>max_samples: idx=rng.choice(idx,max_samples,replace=False)
        t=time.perf_counter(); lb=fit_kmeans(X[idx],k,seed+b,cfg,purpose="bootstrap")
        aris.append(adjusted_rand_score(ref[idx],lb))
        print(f"[CLUSTER/BOOT] {b+1}/{B} ARI={aris[-1]:.5f} ({time.perf_counter()-t:.2f}s)", flush=True)
    return float(np.mean(aris)),float(np.std(aris)),aris


def coassociation(X,k,R=10,max_n=5000,seed=42,cfg=None):
    cfg=dict(cfg or {}); n=len(X)
    if n>max_n:
        rng=np.random.default_rng(seed); idx=np.sort(rng.choice(n,max_n,replace=False)); Xs=X[idx]
    else: idx=np.arange(n); Xs=X
    A=np.zeros((len(idx),len(idx)),dtype=np.float32)
    print(f"[CLUSTER/COASSOC] R={R}, n={len(idx)}, k={k}", flush=True)
    for r in range(R):
        lab=fit_kmeans(Xs,k,seed+r,cfg,purpose="coassociation")
        A += (lab[:,None]==lab[None,:])
        print(f"[CLUSTER/COASSOC] {r+1}/{R}", flush=True)
    return idx,A/float(max(R,1))


def external_metrics(pred,true):
    return {"ARI":float(adjusted_rand_score(true,pred)),"NMI":float(normalized_mutual_info_score(true,pred))}
