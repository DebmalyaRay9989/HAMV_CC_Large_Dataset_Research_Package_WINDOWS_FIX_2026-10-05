from __future__ import annotations
import numpy as np, pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


def standardize_pca(X, target=None):
    X=np.asarray(X,dtype=np.float32)
    X=np.nan_to_num(X,nan=0.0,posinf=0.0,neginf=0.0)
    scaler=StandardScaler()
    Z=scaler.fit_transform(X).astype(np.float32)
    pca=None
    if target and target < Z.shape[1]:
        target=min(int(target),Z.shape[0]-1,Z.shape[1])
        pca=PCA(n_components=target,random_state=42)
        Z=pca.fit_transform(Z).astype(np.float32)
    return Z,scaler,pca


def remove_constant_columns(X, names=None, tol=1e-12):
    X=np.asarray(X)
    var=np.nanvar(X,axis=0)
    keep=var>tol
    names2=[n for n,k in zip(names or range(X.shape[1]),keep) if k]
    return X[:,keep], names2, var


# IMPORTANT: this Dataset must remain at module scope.  Windows (and Python 3.14
# multiprocessing contexts) pickle DataLoader workers; a nested/local class
# such as resnet50_features.<locals>.DS is not picklable.
class ResNetImageDataset:
    def __init__(self, rows, work_size, transform):
        self.rows = rows.reset_index(drop=True)
        self.work_size = tuple(work_size)
        self.transform = transform

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        import cv2
        from PIL import Image
        p = self.rows.iloc[i].path
        im = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if im is None:
            raise IOError(f"Could not read image: {p}")
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        im = cv2.resize(im, self.work_size, interpolation=cv2.INTER_AREA)
        return self.transform(Image.fromarray(im)), i


def resnet50_features(manifest, cfg, out_npy, batch_size=64, num_workers=0):
    import torch
    from torchvision.models import resnet50, ResNet50_Weights
    from torch.utils.data import DataLoader

    # Windows-safe default: single-process loading avoids spawn/pickle failures.
    # A positive worker count is still supported when the environment permits it.
    num_workers = max(0, int(num_workers))
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weights=ResNet50_Weights.DEFAULT
    model=resnet50(weights=weights); model.fc=torch.nn.Identity(); model.eval().to(device)
    transform=weights.transforms()
    ds=ResNetImageDataset(manifest, cfg["work_size"], transform)
    loader=DataLoader(
        ds,
        batch_size=int(batch_size),
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type=="cuda"),
        persistent_workers=(num_workers > 0),
    )
    mm=np.lib.format.open_memmap(out_npy,mode="w+",dtype="float32",shape=(len(manifest),2048))
    with torch.inference_mode():
        for xb,idx in loader:
            z=model(xb.to(device,non_blocking=True)).detach().cpu().numpy().astype("float32")
            mm[np.asarray(idx)]=z
    mm.flush()
    d=int(cfg.get("resnet_dims",100))
    return np.asarray(mm)[:,:d]


def build_base_representation(hand, visual=None, pca_target=50):
    parts=[np.asarray(hand,dtype=np.float32)]
    if visual is not None: parts.append(np.asarray(visual,dtype=np.float32))
    X=np.concatenate(parts,axis=1)
    X,names,_=remove_constant_columns(X)
    Z,scaler,pca=standardize_pca(X,pca_target)
    return Z, scaler, pca
