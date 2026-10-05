
from __future__ import annotations
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

@dataclass
class Config:
    dataset_root: str="C:/Users/Debmalya Ray/OneDrive/Desktop/NEW_MODEL/NEW/HAMV_CC_DATA/dataset/dataset"
    output_root: str="./results"
    image_extensions: tuple=(".png",".jpg",".jpeg",".bmp",".tif",".tiff",".webp")
    work_size: tuple=(224,224)
    use_denoise: bool=True
    use_deskew: bool=True
    use_clahe: bool=True
    use_binarize: bool=False
    use_resnet50: bool=True
    resnet_dims: int=100
    batch_size: int=64
    num_workers: int=0
    feature_dtype: str="float32"
    cache_features: bool=True
    pca: bool=True
    pca_target: int=50
    cluster_k_min: int=2
    cluster_k_max: int=20
    cluster_method: str="minibatch_kmeans"
    cluster_batch_size: int=2048
    cluster_n_init: int=10
    cluster_max_iter: int=300
    cluster_threads: int=1
    cluster_n_init_k_selection: int=2
    cluster_max_iter_k_selection: int=100
    cluster_batch_size_k_selection: int=512
    cluster_n_init_stability: int=3
    cluster_max_iter_stability: int=200
    cluster_batch_size_stability: int=1024
    cluster_n_init_final: int=5
    cluster_max_iter_final: int=300
    cluster_batch_size_final: int=1024
    cluster_n_init_coassociation: int=2
    cluster_max_iter_coassociation: int=100
    cluster_batch_size_coassociation: int=512
    cluster_max_no_improvement: int=20
    cluster_tol: float=1e-3
    cluster_reassignment_ratio: float=0.01
    max_k_selection_samples: int=10000
    max_stability_samples: int=20000
    seeds: tuple=(42,43,44,45,46)
    bootstrap_runs: int=10
    bootstrap_fraction: float=0.8
    max_coassociation_n: int=5000
    run_ssl: bool=True
    run_hamvcc: bool=True
    hamvcc_epochs: int=150
    hamvcc_latent: int=64
    hamvcc_batch_size: int=64
    hamvcc_lr: float=1e-3
    lambda_inst: float=1.0
    lambda_view: float=0.5
    lambda_cc: float=0.5
    use_reliability: bool=True
    ssl_epochs: int=30
    ssl_batch_size: int=256
    ssl_lr: float=1e-3
    ssl_temperature: float=0.1
    ssl_projection_dim: int=128
    ssl_noise: float=0.05
    save_embeddings: bool=True
    save_plots: bool=True
    label_csv: Optional[str]=None
    label_path_column: str="path"
    label_column: str="label"

    @classmethod
    def from_json(cls, path):
        d=json.loads(Path(path).read_text(encoding="utf-8"))
        for k in ("image_extensions","work_size","seeds"):
            if k in d: d[k]=tuple(d[k])
        return cls(**d)

    def to_json(self,path):
        Path(path).write_text(json.dumps(asdict(self),indent=2,default=list),encoding="utf-8")
