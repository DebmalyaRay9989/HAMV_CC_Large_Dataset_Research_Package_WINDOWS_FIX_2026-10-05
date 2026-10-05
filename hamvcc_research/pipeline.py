from __future__ import annotations
import json, os, time, platform
from pathlib import Path
import numpy as np, pandas as pd
from .provenance import build_manifest, write_environment
from .features import extract_handcrafted, FEATURE_NAMES
from .representations import resnet50_features, build_base_representation
from .clustering import select_k, multi_seed_stability, bootstrap_stability, coassociation, safe_internal_metrics, fit_kmeans
from .ssl import train_feature_ssl
from .hamvcc import train_hamvcc

def _resolve_dataset(cfg):
    override=os.environ.get('HAMVCC_DATASET')
    root=Path(override or cfg['dataset_root']).expanduser()
    return root

def _save_plots(out, curves, comparison, weights=None):
    if not comparison.empty:
        import matplotlib.pyplot as plt
        fig,ax=plt.subplots(figsize=(8,4));
        for name,df in curves.items():
            if not df.empty: ax.plot(df['k'],df['silhouette'],marker='o',label=name)
        ax.set_xlabel('k'); ax.set_ylabel('Silhouette'); ax.set_title('HAMV-CC cluster-count diagnostics'); ax.legend(); fig.tight_layout(); fig.savefig(out/'k_diagnostics.png',dpi=220,bbox_inches='tight'); plt.close(fig)
        fig,ax=plt.subplots(figsize=(8,4));
        ax.bar(comparison['representation'],comparison['silhouette']); ax.set_ylabel('Silhouette'); ax.set_title('Representation comparison'); fig.tight_layout(); fig.savefig(out/'representation_comparison.png',dpi=220,bbox_inches='tight'); plt.close(fig)
    if weights is not None and len(weights):
        fig,ax=plt.subplots(figsize=(7,4)); ax.hist(weights[:,0],bins=20,alpha=.8,label='structural weight'); ax.hist(weights[:,1],bins=20,alpha=.8,label='visual weight'); ax.set_xlabel('Learned reliability weight'); ax.set_ylabel('Documents'); ax.legend(); fig.tight_layout(); fig.savefig(out/'hamvcc_reliability_weights.png',dpi=220,bbox_inches='tight'); plt.close(fig)

def run(config):
    cfg=dict(config.__dict__) if hasattr(config,'__dict__') else dict(config)
    t_run=time.perf_counter()
    out=Path(cfg['output_root']); out.mkdir(parents=True,exist_ok=True)
    print(f'[PIPELINE] output={out.resolve()}', flush=True)
    root=_resolve_dataset(cfg)
    cfg['resolved_dataset_root']=str(root.resolve())
    cfg['dataset_source']='HAMVCC_DATASET environment override' if os.environ.get('HAMVCC_DATASET') else 'config'
    Path(out/'config.json').write_text(json.dumps(cfg,indent=2,default=list),encoding='utf-8')
    write_environment(out/'environment.json')
    if not root.exists(): raise FileNotFoundError(f"Dataset root not found: {root.resolve()}\nSet HAMVCC_DATASET or edit configs/large_dataset.json.")
    print(f'[PIPELINE] dataset={root.resolve()}', flush=True)
    manifest_path=out/'manifest.csv'; manifest=build_manifest(root,cfg['image_extensions'],manifest_path)
    print(f'[PIPELINE] discovered {len(manifest)} input files', flush=True)
    if len(manifest)==0: raise RuntimeError(f'No supported raster files found under {root.resolve()}')
    cache=out/'cache'; cache.mkdir(exist_ok=True)
    hand_path=cache/'handcrafted.npy'
    valid=False
    if cfg.get('cache_features',True) and hand_path.exists():
        try: valid=np.load(hand_path,mmap_mode='r').shape==(len(manifest),len(FEATURE_NAMES))
        except Exception: valid=False
    if not valid:
        print('[PIPELINE] extracting handcrafted features...', flush=True)
        extract_handcrafted(manifest,cfg,hand_path,cache/'feature_errors.csv')
    else:
        print('[PIPELINE] using cached handcrafted features', flush=True)
    hand=np.load(hand_path,mmap_mode='r')
    visual=None
    if cfg.get('use_resnet50',True):
        vp=cache/'resnet50.npy'; valid=False
        if vp.exists():
            try: valid=np.load(vp,mmap_mode='r').shape==(len(manifest),2048)
            except Exception: valid=False
        if not valid:
            print('[PIPELINE] extracting ResNet50 visual features...', flush=True)
            visual=resnet50_features(manifest,cfg,vp,cfg['batch_size'],cfg['num_workers'])
        else:
            print('[PIPELINE] using cached ResNet50 features', flush=True)
            visual=np.load(vp,mmap_mode='r')[:,:cfg['resnet_dims']]
    base,_,_=build_base_representation(hand,visual,cfg.get('pca_target',50) if cfg.get('pca',True) else None)
    np.save(cache/'base_representation.npy',base)
    pd.DataFrame(hand,columns=FEATURE_NAMES).to_csv(out/'handcrafted_features.csv',index=False)
    reps={'base':base}; curves={}
    if cfg.get('run_ssl',True):
        _,H,hist=train_feature_ssl(base,epochs=cfg['ssl_epochs'],batch_size=cfg['ssl_batch_size'],lr=cfg['ssl_lr'],temp=cfg['ssl_temperature'],proj_dim=cfg['ssl_projection_dim'],noise=cfg['ssl_noise'],seed=cfg['seeds'][0])
        np.save(cache/'ssl_representation.npy',H); hist.to_csv(out/'ssl_history.csv',index=False); reps['ssl']=H
    results=[]
    for name,X in reps.items():
        print(f'[PIPELINE] representation={name}', flush=True)
        k,kdf=select_k(X,cfg['cluster_k_min'],cfg['cluster_k_max'],cfg['seeds'][0],max_samples=cfg.get('max_k_selection_samples',10000),cfg=cfg); curves[name]=kdf; kdf.to_csv(out/f'k_diagnostics_{name}.csv',index=False)
        stab,labs=multi_seed_stability(X,k,cfg['seeds'],cfg,max_samples=cfg.get('max_stability_samples',20000)); lab=labs[0]
        metrics=safe_internal_metrics(X,lab); boot_mean,boot_std,_=bootstrap_stability(X,lab,k,cfg,B=cfg['bootstrap_runs'],frac=cfg['bootstrap_fraction'],seed=cfg['seeds'][0],max_samples=cfg.get('max_stability_samples',20000))
        results.append({'representation':name,'k':k,'seed_stability_ARI':stab,'bootstrap_ARI':boot_mean,'bootstrap_ARI_std':boot_std,**metrics})
        pd.DataFrame({'index':np.arange(len(lab)),'cluster':lab}).to_csv(out/f'assignments_{name}.csv',index=False)

    # Full two-view HAMV-CC research extension: structural handcrafted + visual ResNet view.
    hamvcc_weights=None
    if cfg.get('run_hamvcc',True) and visual is not None:
        provisional_k=int([r for r in results if r['representation']=='base'][0]['k'])
        _,fused,hamvcc_weights,hist=train_hamvcc(
            hand, np.asarray(visual,dtype=np.float32), provisional_k,
            epochs=cfg['hamvcc_epochs'],latent=cfg['hamvcc_latent'],batch_size=cfg['hamvcc_batch_size'],lr=cfg['hamvcc_lr'],
            lambda_inst=cfg['lambda_inst'],lambda_view=cfg['lambda_view'],lambda_cc=cfg['lambda_cc'],use_reliability=cfg['use_reliability'],
            seed=cfg['seeds'][0])
        np.save(cache/'hamvcc_fused.npy',fused); np.save(cache/'hamvcc_reliability_weights.npy',hamvcc_weights); hist.to_csv(out/'hamvcc_history.csv',index=False)
        hk,hkdf=select_k(fused,cfg['cluster_k_min'],cfg['cluster_k_max'],cfg['seeds'][0],max_samples=cfg.get('max_k_selection_samples',10000),cfg=cfg); curves['hamvcc']=hkdf; hkdf.to_csv(out/'k_diagnostics_hamvcc.csv',index=False)
        hstab,hlabs=multi_seed_stability(fused,hk,cfg['seeds'],cfg,max_samples=cfg.get('max_stability_samples',20000)); hlab=hlabs[0]; hm=safe_internal_metrics(fused,hlab); hbmean,hbstd,_=bootstrap_stability(fused,hlab,hk,cfg,B=cfg['bootstrap_runs'],frac=cfg['bootstrap_fraction'],seed=cfg['seeds'][0],max_samples=cfg.get('max_stability_samples',20000))
        results.append({'representation':'hamvcc','k':hk,'seed_stability_ARI':hstab,'bootstrap_ARI':hbmean,'bootstrap_ARI_std':hbstd,**hm})
        pd.DataFrame({'index':np.arange(len(hlab)),'cluster':hlab,'structural_weight':hamvcc_weights[:,0],'visual_weight':hamvcc_weights[:,1]}).to_csv(out/'assignments_hamvcc.csv',index=False)
        chosen_X=fused; chosen_name='hamvcc'; chosen_k=int(hk)
    else:
        chosen_name=None; chosen_X=None; chosen_k=None

    rdf=pd.DataFrame(results)
    rdf['rank']=(rdf['silhouette'].rank(ascending=False)+rdf['DB'].rank(ascending=True)+rdf['seed_stability_ARI'].rank(ascending=False))/3
    if chosen_name is None:
        rdf=rdf.sort_values('rank').reset_index(drop=True); chosen_name=str(rdf.iloc[0]['representation']); chosen_X=reps[chosen_name]; chosen_k=int(rdf.iloc[0]['k'])
    rdf.to_csv(out/'representation_comparison.csv',index=False)
    final=fit_kmeans(chosen_X,chosen_k,cfg['seeds'][0],cfg)
    pd.DataFrame({'index':np.arange(len(final)),'path':manifest['path'],'relative_path':manifest['relative_path'],'sha256':manifest['sha256'],'cluster':final}).to_csv(out/'final_assignments.csv',index=False)
    if len(chosen_X)<=cfg['max_coassociation_n']:
        idx,A=coassociation(chosen_X,chosen_k,R=10,max_n=cfg['max_coassociation_n'],seed=cfg['seeds'][0],cfg=cfg); np.save(out/'coassociation.npy',A); pd.DataFrame({'index':idx}).to_csv(out/'coassociation_indices.csv',index=False)
    if cfg.get('save_plots',True): _save_plots(out,curves,rdf,hamvcc_weights)
    err_path=out/'cache'/'feature_errors.csv'
    feature_errors=0
    if err_path.exists() and err_path.stat().st_size:
        try: feature_errors=len(pd.read_csv(err_path))
        except Exception: feature_errors=0
    report={'documents':len(manifest),'dataset_root':str(root.resolve()),'chosen_representation':chosen_name,'k':chosen_k,'hamvcc_executed':bool(cfg.get('run_hamvcc',True) and visual is not None),'representation_comparison':rdf.to_dict(orient='records'),'duplicate_sha256_groups':int((manifest.groupby('sha256').size()>1).sum()),'feature_errors':int(feature_errors),'scientific_boundary':'Internal metrics are geometric diagnostics. New results must be generated by actual execution; this package does not promote recorded notebook values to new evidence.'}
    report['elapsed_seconds']=round(time.perf_counter()-t_run,2)
    Path(out/'research_summary.json').write_text(json.dumps(report,indent=2,default=float),encoding='utf-8')
    print(f"[PIPELINE] COMPLETE: {report['elapsed_seconds']:.1f}s; chosen={chosen_name}, k={chosen_k}", flush=True)
    return report
