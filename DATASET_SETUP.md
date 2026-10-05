# Dataset setup for HAMV-CC

## Requested dataset

`C:\Users\Debmalya Ray\OneDrive\Desktop\NEW_MODEL\NEW\HAMV_CC_DATA\dataset\dataset`

The current configuration points to this location. The package cannot read a path on another computer during packaging; the actual experiment must be executed on the Windows machine where this directory exists.

## Recommended override

```powershell
$env:HAMVCC_DATASET = "C:\Users\Debmalya Ray\OneDrive\Desktop\NEW_MODEL\NEW\HAMV_CC_DATA\dataset\dataset"
```

Then run:

```powershell
python -m hamvcc_research --config configs/large_dataset.json
```

## Dataset audit

The run creates `manifest.csv` containing absolute path, relative path, filename, extension, size, modification timestamp and SHA-256. Duplicate hashes are reported rather than silently deleted, matching the supplied notebook's reproducibility contract. fileciteturn0file0L211-L221
