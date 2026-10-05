
from pathlib import Path
import tempfile, cv2, numpy as np, json, subprocess, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

def make_data(root,n=24):
    root.mkdir(parents=True,exist_ok=True)
    for i in range(n):
        im=np.full((224,224,3),255,np.uint8)
        cv2.putText(im, f"DOC {i%4}", (20,110), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,0), 2)
        cv2.rectangle(im,(10+i%20,10),(210,210-i%20),(0,0,0),1)
        cv2.imwrite(str(root/f"{i:04d}.png"),im)

if __name__=="__main__":
    with tempfile.TemporaryDirectory() as td:
        root=Path(td); data=root/"dataset"; out=root/"results"; make_data(data)
        cfg={"dataset_root":str(data),"output_root":str(out),"use_resnet50":False,
             "run_ssl":False,"cluster_k_min":2,"cluster_k_max":4,"seeds":[1,2],
             "bootstrap_runs":2,"max_coassociation_n":100}
        (root/"cfg.json").write_text(json.dumps(cfg))
        from hamvcc_research.config import Config
        from hamvcc_research.pipeline import run
        r=run(Config.from_json(root/"cfg.json"))
        assert r["documents"]==24
        assert (out/"manifest.csv").exists()
        assert (out/"final_assignments.csv").exists()
        print("SMOKE TEST PASSED")
