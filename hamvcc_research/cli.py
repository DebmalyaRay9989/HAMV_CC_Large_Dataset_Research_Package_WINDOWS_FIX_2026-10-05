
from __future__ import annotations
import argparse
from .config import Config
from .pipeline import run

def main():
    ap=argparse.ArgumentParser(description="Scalable HAMV-CC archive research pipeline")
    ap.add_argument("--config",required=True)
    args=ap.parse_args()
    cfg=Config.from_json(args.config)
    report=run(cfg)
    print("\nCompleted.")
    print("Documents:",report["documents"])
    print("Chosen representation:",report["chosen_representation"])
    print("k:",report["k"])
    print("Output:",cfg.output_root)
if __name__=="__main__": main()
