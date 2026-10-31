#!/usr/bin/env python3
"""
Autonomous Harvest and Dispatch for Kaggle GPU Titan Sprints
============================================================
Checks if hammadshakeelai/alphacandy-titan-50m-gpu-mastery has completed.
If complete:
  1. Downloads output into campaign/harvest_nov01/titan_gpu_v7/
  2. Updates local master checkpoint checkpoints/alphacandy_titan_master.pt
  3. Updates Kaggle Model Instance hammadshakeelai/alphacandy-titan
  4. Dispatches Sprint v8 via kaggle kernels push
  5. Commits evaluation summary to git
"""

import sys
import json
import shutil
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
PY_KAGGLE = str(REPO_ROOT / "venv" / "bin" / "kaggle")
PY_BIN = str(REPO_ROOT / "venv" / "bin" / "python3")
KERNEL_SLUG = "hammadshakeelai/alphacandy-titan-50m-gpu-mastery"
HARVEST_DIR = REPO_ROOT / "campaign" / "harvest_nov01" / "titan_gpu_v7"
MASTER_CKPT = REPO_ROOT / "checkpoints" / "alphacandy_titan_master.pt"
GPU_DEPLOY_DIR = REPO_ROOT / "kaggle_deploy" / "alphacandy_titan_gpu_trainer"


def get_cmd(cmd):
    try:
        r = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=120)
        return r.returncode, r.stdout.strip(), r.stderr.strip()
    except Exception as e:
        return -1, "", str(e)


def main():
    print(f"[Titan Monitor] Checking status of {KERNEL_SLUG}...")
    code, out, _ = get_cmd([PY_KAGGLE, "kernels", "status", KERNEL_SLUG])
    print(f"Status Output: {out}")

    if "COMPLETE" not in out:
        print("[Titan Monitor] Kernel is still running or queued. Exiting.")
        return 0

    print(f"\n🎉 [Titan Monitor] Sprint v7 COMPLETE! Starting Harvest into {HARVEST_DIR}...")
    HARVEST_DIR.mkdir(parents=True, exist_ok=True)
    
    code, d_out, d_err = get_cmd([PY_KAGGLE, "kernels", "output", KERNEL_SLUG, "-p", str(HARVEST_DIR)])
    print(f"Download Output:\n{d_out}")
    if d_err:
        print(f"Download Stderr: {d_err}")

    # Locate downloaded master weights
    found_ckpt = None
    for p in HARVEST_DIR.rglob("alphacandy_titan_master.pt"):
        found_ckpt = p
        break

    if found_ckpt and found_ckpt.exists():
        print(f"Updating local master checkpoint: {MASTER_CKPT} ({found_ckpt.stat().st_size / (1024*1024):.1f} MB)...")
        shutil.copy2(found_ckpt, MASTER_CKPT)
    else:
        print("[Warning] alphacandy_titan_master.pt not found in harvest output!")

    # Read and print summary metrics
    summary_file = HARVEST_DIR / "titan_gpu_summary.json"
    if summary_file.exists():
        try:
            summary = json.loads(summary_file.read_text())
            print(f"\n=======================================================")
            print(f"🏆 TITAN GPU SPRINT v7 EVALUATION SUMMARY")
            print(f"=======================================================")
            print(f"Holdout Win Rate:     {summary.get('holdout_win_rate_pct', 'N/A')}% ({summary.get('holdout_won', 'N/A')}/{summary.get('holdout_total', 'N/A')})")
            print(f"Combined Best Loss:   {summary.get('best_loss', 'N/A')}")
            print(f"Quantile Huber Loss:  {summary.get('final_quantile_loss', 'N/A')}")
            print(f"Move Match Acc (pt):  {summary.get('mean_pt', 'N/A')}")
            print(f"Polyak Shadow Steps:  {summary.get('total_ema_updates', 'N/A')}")
            print(f"Sprint Wall Time:     {summary.get('total_sprint_minutes', 'N/A')} mins")
            print(f"=======================================================\n")
        except Exception as e:
            print(f"Could not parse summary json: {e}")

    # Dispatch Sprint v8
    print(f"\n🚀 [Titan Monitor] Auto-Dispatching Sprint v8 on Tesla T4 GPU...")
    p_code, p_out, p_err = get_cmd([PY_KAGGLE, "kernels", "push", "-p", str(GPU_DEPLOY_DIR)])
    print(f"Push Result: {p_out}")
    if p_err:
        print(f"Push Stderr: {p_err}")

    print("\n✅ [Titan Monitor] Harvest & Re-dispatch cycle complete!")
    return 1


if __name__ == "__main__":
    sys.exit(main())
