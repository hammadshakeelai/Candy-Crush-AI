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

# KAGGLE_INPUT_ROOT / KAGGLE_WORK_DIR overrides allow a faithful local dry-run of this exact script
input_root = os.environ.get("KAGGLE_INPUT_ROOT", "/kaggle/input")
work_dir = os.environ.get("KAGGLE_WORK_DIR", "/kaggle/working")
os.makedirs(work_dir, exist_ok=True)
os.chdir(work_dir)
sys.path.insert(0, work_dir)

# Training scale (overridable for quick local dry-runs)
EPISODES = int(os.environ.get("AC_EPISODES", "200"))
PRETRAIN_LEVELS = int(os.environ.get("AC_PRETRAIN_LEVELS", "200"))
EPOCHS = int(os.environ.get("AC_EPOCHS", "10"))
GROUP_SIZE = int(os.environ.get("AC_GROUP_SIZE", "6"))

# 1. Locate source files and datasets in the input root
src_root = None
data_json_found = None
sqlite_found = None

for root, dirs, files in os.walk(input_root):
    if "candy_crush_pkg.zip" in files and src_root is None:
        zip_path = os.path.join(root, "candy_crush_pkg.zip")
        print(f"[INFO] Extracting {zip_path} into {work_dir}...")
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(work_dir)
        src_root = work_dir
    elif "train_spatial_transformer_gpu.py" in files and src_root is None:
        src_root = root

    if "configs_v1.208.json" in files and not data_json_found:
        data_json_found = os.path.join(root, "configs_v1.208.json")
    if "candycrush_master_levels.sqlite" in files and not sqlite_found:
        sqlite_found = os.path.join(root, "candycrush_master_levels.sqlite")

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
if sqlite_found:
    print(f"[INFO] Setting KING_SQLITE_PATH={sqlite_found}")
    os.environ["KING_SQLITE_PATH"] = sqlite_found
if not data_json_found and not sqlite_found:
    print("[FATAL] No level data (configs_v1.208.json / candycrush_master_levels.sqlite) found in /kaggle/input")
    sys.exit(1)

# Hot-patch device bug in train_spatial_transformer_gpu if running from cached input
for check_path in [
    os.path.join(work_dir, "train_spatial_transformer_gpu.py"),
    os.path.join(work_dir, "candy_crush_pkg", "train_spatial_transformer_gpu.py")
]:
    if os.path.exists(check_path):
        with open(check_path, "r", encoding="utf-8") as f:
            t_code = f.read()
        if "action_dist.log_prob(torch.tensor(a_idx)).item()" in t_code:
            print("[INFO] Hot-patching action_dist.log_prob device target...")
            t_code = t_code.replace(
                "action_dist.log_prob(torch.tensor(a_idx)).item()",
                "action_dist.log_prob(torch.tensor(a_idx, device=device)).item()"
            )
            with open(check_path, "w", encoding="utf-8") as f:
                f.write(t_code)

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
        episodes=EPISODES,
        pretrain_levels=PRETRAIN_LEVELS,
        epochs=EPOCHS,
        group_size=GROUP_SIZE,
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
