# HAMV-CC Large-Dataset Research Package — v1.1.0

This package is an execution/reproducibility layer derived from the supplied **HAMV_CC_InDepth_Reproducible_Research_Notebook_Enhanced**. It is designed for the user's specified HAMV-CC archive and keeps the notebook's evidence boundary intact.

## Dataset configured for this run

Default Windows dataset root:

`C:\Users\Debmalya Ray\OneDrive\Desktop\NEW_MODEL\NEW\HAMV_CC_DATA\dataset\dataset`

The package first checks the environment variable `HAMVCC_DATASET`. If it is set, that path takes precedence. This is the recommended way to move the experiment to another machine without editing the configuration.

The supplied notebook previously reported a reference archive of 683 raster files (651 JPEG + 32 TIFF, with 8 duplicate SHA-256 groups). Those are **reference observations, not assumptions**; this package recomputes the manifest from the configured dataset before each fresh run. fileciteturn0file0L47-L48

## What was improved

1. Exact requested dataset path is configured, with `HAMVCC_DATASET` override.
2. Manifest/provenance is generated before feature extraction.
3. Empty/failed feature caches are handled safely.
4. The original scalable baseline remains available (handcrafted + ResNet50 + PCA + feature-space SSL).
5. A separate executable **reliability-gated two-view HAMV-CC research extension** is now included. It uses explicit structural/visual views, availability masks, learned view reliability, cross-view consistency, cluster consistency, and instance contrast, matching the extension described in the supplied notebook. fileciteturn0file0L19-L29
6. HAMV-CC fused embeddings, learned reliability weights, training history, assignments, k diagnostics and publication-ready diagnostic plots are saved.
7. Ten matched seeds are configured for downstream clustering stability.
8. The smoke test now runs directly from the source tree.
9. Scientific claim boundaries are written into `research_summary.json`: new results are never silently substituted with previously recorded notebook values.

## Installation

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

For an existing CUDA/PyTorch environment, retain a compatible PyTorch + torchvision installation.

## Run on the requested dataset

From the extracted package directory:

```powershell
python -m hamvcc_research --config configs/large_dataset.json
```

Or:

```powershell
python scripts/run_research.py --config configs/large_dataset.json
```

If the dataset is stored elsewhere:

```powershell
$env:HAMVCC_DATASET = "D:\your\dataset"
python -m hamvcc_research --config configs/large_dataset.json
```

## Main outputs

- `results/HAMV_CC/manifest.csv`
- `results/HAMV_CC/environment.json`
- `results/HAMV_CC/config.json`
- `results/HAMV_CC/cache/handcrafted.npy`
- `results/HAMV_CC/cache/resnet50.npy`
- `results/HAMV_CC/cache/hamvcc_fused.npy`
- `results/HAMV_CC/cache/hamvcc_reliability_weights.npy`
- `results/HAMV_CC/hamvcc_history.csv`
- `results/HAMV_CC/k_diagnostics_hamvcc.csv`
- `results/HAMV_CC/assignments_hamvcc.csv`
- `results/HAMV_CC/final_assignments.csv`
- `results/HAMV_CC/representation_comparison.csv`
- `results/HAMV_CC/research_summary.json`
- `results/HAMV_CC/k_diagnostics.png`
- `results/HAMV_CC/representation_comparison.png`
- `results/HAMV_CC/hamvcc_reliability_weights.png`

## Scientific boundary

The supplied notebook explicitly distinguishes the source-aligned baseline from the full HAMV-CC extension and states that executing the extension does not make previously reported numbers true for a new run. This package follows that rule. fileciteturn0file0L25-L31

Internal clustering metrics are geometric diagnostics rather than semantic ground truth. Independent ARI/NMI should only be interpreted as semantic validation when independent labels exist. The supplied notebook likewise requires frozen preprocessing, feature ordering, hyperparameters, seeds and evaluation policy before confirmatory runs. fileciteturn0file0L37-L46

## Notebook relationship

`notebooks/HAMV_CC_Large_Dataset_Research.ipynb` is the interactive research entry point. The Python package is the canonical command-line execution path; the notebook is retained for interactive inspection and audit. The supplied enhanced notebook is also retained under `source/` for audit/reference.


## 1.2.0 runtime improvements

This release keeps the experimental protocol (k=2..20 and matched seeds 42..51) while making the clustering stage more suitable for a 683-image Windows workstation. Exploratory k-selection, seed-stability, bootstrap, co-association, and final clustering have separate iteration/initialisation settings. CPU thread use is bounded through `cluster_threads`, and the runner prints per-stage progress and elapsed time.

The package does not claim new experimental results until the pipeline is executed against the configured dataset.

## 1.2.1 runtime compatibility fix

The ResNet50 image dataset class is module-scoped so PyTorch DataLoader workers can be pickled on Windows. The large-dataset configuration now uses `num_workers=0` by default, avoiding multiprocessing spawn/forkserver incompatibilities under modern Python/Miniconda installations. Positive worker counts remain supported when the runtime is known to permit them.

This is a runtime compatibility fix only; it does not change the HAMV-CC scientific protocol or claim new experimental results.
