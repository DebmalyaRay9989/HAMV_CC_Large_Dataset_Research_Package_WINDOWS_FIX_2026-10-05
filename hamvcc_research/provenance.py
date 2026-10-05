
from __future__ import annotations
import hashlib, json, os, platform, sys
from pathlib import Path
from datetime import datetime, timezone
import pandas as pd

def sha256_file(path, chunk_size=4*1024*1024):
    h=hashlib.sha256()
    with open(path,"rb") as f:
        while True:
            b=f.read(chunk_size)
            if not b: break
            h.update(b)
    return h.hexdigest()

def build_manifest(root, extensions, out_csv, workers=4):
    root=Path(root); files=sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in extensions)
    rows=[]
    from concurrent.futures import ThreadPoolExecutor
    def one(p):
        try:
            st=p.stat()
            return {"path":str(p.resolve()),"relative_path":str(p.relative_to(root)),
                    "name":p.name,"ext":p.suffix.lower(),"bytes":st.st_size,
                    "mtime_ns":st.st_mtime_ns,"sha256":sha256_file(p),"error":""}
        except Exception as e:
            return {"path":str(p.resolve()),"relative_path":str(p.relative_to(root)),
                    "name":p.name,"ext":p.suffix.lower(),"bytes":-1,"mtime_ns":-1,
                    "sha256":"","error":repr(e)}
    with ThreadPoolExecutor(max_workers=max(1,workers)) as ex:
        for r in ex.map(one, files): rows.append(r)
    df=pd.DataFrame(rows)
    Path(out_csv).parent.mkdir(parents=True,exist_ok=True); df.to_csv(out_csv,index=False)
    return df

def environment_record():
    rec={"timestamp_utc":datetime.now(timezone.utc).isoformat(),
         "python":sys.version,"platform":platform.platform(),
         "executable":sys.executable}
    try:
        import numpy,pandas,sklearn,cv2
        rec.update({"numpy":numpy.__version__,"pandas":pandas.__version__,
                    "sklearn":sklearn.__version__,"opencv":cv2.__version__})
    except Exception as e: rec["scientific_import_error"]=repr(e)
    try:
        import torch
        rec.update({"torch":torch.__version__,"cuda_available":bool(torch.cuda.is_available()),
                    "cuda_version":torch.version.cuda})
    except Exception as e: rec["torch_import_error"]=repr(e)
    return rec

def write_environment(path):
    Path(path).write_text(json.dumps(environment_record(),indent=2),encoding="utf-8")
