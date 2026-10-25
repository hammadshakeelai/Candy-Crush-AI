"""
Create an updated, production-ready candy_crush_pkg.zip for Kaggle.
Includes:
  - core/ (complete reverse engine & level loader)
  - models/ (all grandmaster models, agents, ensembles, heuristics)
  - checkpoints/ (all master neural checkpoints including 14.4M AlphaCandy Large Master)
  - campaign/ (campaign scripts & progress)
  - analysis/ (benchmarks & metrics)
"""

import os
import zipfile

def create_package():
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    out_zip = os.path.join(repo_root, "kaggle_deploy/src/candy_crush_pkg.zip")

    include_dirs = ["core", "models", "checkpoints", "campaign", "analysis"]
    include_files = ["README.md", "requirements.txt"]

    print(f"Creating updated package at: {out_zip}...")
    with zipfile.ZipFile(out_zip, "w", zipfile.ZIP_DEFLATED) as zf:
        for d in include_dirs:
            full_d = os.path.join(repo_root, d)
            if not os.path.exists(full_d):
                continue
            for root, _, files in os.walk(full_d):
                for f in files:
                    if f.endswith((".pyc", ".tmp", ".kaggle-partial")) or "__pycache__" in root:
                        continue
                    full_p = os.path.join(root, f)
                    arcname = os.path.relpath(full_p, repo_root)
                    zf.write(full_p, arcname)
                    print(f"  + Added {arcname}")

        for f in include_files:
            full_f = os.path.join(repo_root, f)
            if os.path.exists(full_f):
                zf.write(full_f, f)
                print(f"  + Added {f}")

    sz_mb = os.path.getsize(out_zip) / (1024 * 1024)
    print(f"\nSuccessfully built {out_zip} ({sz_mb:.2f} MB)!")

if __name__ == "__main__":
    create_package()
