"""
Kaggle Production GPU Runner: AlphaCandy Spatial Attention Transformer.
Executes on NVIDIA Tesla T4 / P100 with PyTorch CUDA Acceleration.
"""

import os
import sys
import time
import zipfile
import torch

print("=================================================================")
print("  ALPHACANDY SPATIAL TRANSFORMER: KAGGLE GPU PIPELINE            ")
print("=================================================================")

work_dir = "/kaggle/working"
os.chdir(work_dir)
sys.path.insert(0, work_dir)

# 1. Locate source files and datasets in /kaggle/input
src_root = None
data_json_found = None

for root, dirs, files in os.walk("/kaggle/input"):
    if "candy_crush_pkg.zip" in files and src_root is None:
        zip_path = os.path.join(root, "candy_crush_pkg.zip")
        print(f"[INFO] Extracting {zip_path} into {work_dir}...")
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(work_dir)
        src_root = work_dir
    elif "train_spatial_transformer_gpu.py" in files and src_root is None:
        src_root = root

    if ("candycrush_master_levels.json" in files or "configs_v1.208.json" in files) and not data_json_found:
        data_json_found = os.path.join(root, "candycrush_master_levels.json" if "candycrush_master_levels.json" in files else "configs_v1.208.json")

if src_root:
    print(f"[INFO] Source root resolved: {src_root}")
    if src_root not in sys.path:
        sys.path.insert(0, src_root)
    # Also copy all .py files to work_dir if running from read-only input
    if src_root != work_dir:
        import shutil
        for item in os.listdir(src_root):
            s_item = os.path.join(src_root, item)
            d_item = os.path.join(work_dir, item)
            if not os.path.exists(d_item):
                if os.path.isdir(s_item):
                    shutil.copytree(s_item, d_item)
                else:
                    shutil.copy2(s_item, d_item)
else:
    print(f"[WARN] No source root found; using current directory {work_dir}")

if data_json_found:
    print(f"[INFO] Setting KING_CONFIG_PATH={data_json_found}")
    os.environ["KING_CONFIG_PATH"] = data_json_found

# 2. Verify GPU status
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Compute Device: {device}")
if torch.cuda.is_available():
    print(f"[INFO] GPU Name: {torch.cuda.get_device_name(0)}")
    print(f"[INFO] Total VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

# 3. Import and execute AlphaCandy Transformer Pipeline
try:
    from train_spatial_transformer_gpu import train_alphacandy_transformer_pipeline
    from models.spatial_transformer_net import AlphaCandySpatialNet
    from core.king_level_loader import KingLevelLoader

    print("\n>>> LAUNCHING ALPHACANDY TRANSFORMER GPU PIPELINE...")
    t0 = time.time()
    ckpt_path = os.path.join(work_dir, "alphacandy_spatial_transformer.pt")

    model = train_alphacandy_transformer_pipeline(
        episodes=40,
        pretrain_levels=30,
        epochs=4,
        group_size=6,
        lr=1e-4,
        checkpoint_out=ckpt_path
    )

    total_time = time.time() - t0
    print("\n=================================================================")
    print(f"  ALPHACANDY TRANSFORMER TRAINING COMPLETE IN {total_time/60.0:.2f} MIN! ")
    print(f"  Final Model Artifact: {ckpt_path} ({os.path.getsize(ckpt_path)/1e6:.2f} MB)")
    print("=================================================================")

except Exception as e:
    print(f"[FATAL ERROR] Pipeline failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
