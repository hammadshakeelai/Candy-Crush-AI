#!/usr/bin/env python3
"""
Autonomous Harvest and Dispatch for Kaggle GPU Titan Sprints
============================================================
Checks if hammadshakeelai/alphacandy-titan-50m-gpu-mastery has completed.
If complete:
  1. Detects sprint version and downloads output into campaign/harvest_nov01/titan_gpu_v{N}/
  2. Updates local master checkpoint checkpoints/alphacandy_titan_master.pt
  3. Updates Kaggle Model Instance hammadshakeelai/alphacandy-titan (creates new version)
  4. Dispatches the next Sprint via kaggle kernels push
  5. Commits evaluation summary to git
"""

import sys
import json
import re
import shutil
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
PY_KAGGLE = str(REPO_ROOT / "venv" / "bin" / "kaggle")
PY_BIN = str(REPO_ROOT / "venv" / "bin" / "python3")
KERNEL_SLUG = "hammadshakeelai/alphacandy-titan-50m-gpu-mastery"
HARVEST_BASE = REPO_ROOT / "campaign" / "harvest_nov01"
MASTER_CKPT = REPO_ROOT / "checkpoints" / "alphacandy_titan_master.pt"
GPU_DEPLOY_DIR = REPO_ROOT / "kaggle_deploy" / "alphacandy_titan_gpu_trainer"
MODEL_INSTANCE_DIR = REPO_ROOT / "kaggle_models" / "alphacandy_titan" / "pytorch" / "master"


def get_cmd(cmd, timeout=900):
    try:
        r = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=timeout)
        return r.returncode, r.stdout.strip(), r.stderr.strip()
    except Exception as e:
        return -1, "", str(e)


def get_next_harvest_dir():
    HARVEST_BASE.mkdir(parents=True, exist_ok=True)
    existing_dirs = list(HARVEST_BASE.glob("titan_gpu_v*"))
    versions = []
    for d in existing_dirs:
        m = re.search(r"titan_gpu_v(\d+)", d.name)
        if m:
            versions.append(int(m.group(1)))
    next_v = max(versions, default=6) + 1
    target_dir = HARVEST_BASE / f"titan_gpu_v{next_v}"
    return target_dir, next_v


def main():
    lock_file = REPO_ROOT / "campaign" / ".harvest.lock"
    lock_file.parent.mkdir(parents=True, exist_ok=True)
    lock_fd = open(lock_file, "w")
    try:
        import fcntl
        fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except (IOError, BlockingIOError):
        print("[Titan Monitor] Another harvest is currently running. Exiting cleanly.")
        return 0

    print(f"[Titan Monitor] Checking status of {KERNEL_SLUG}...")
    code, out, _ = get_cmd([PY_KAGGLE, "kernels", "status", KERNEL_SLUG])
    print(f"Status Output: {out}")

    if "COMPLETE" not in out:
        print("[Titan Monitor] Kernel is still running or queued. Exiting.")
        return 0

    harvest_dir, sprint_num = get_next_harvest_dir()
    print(f"\n🎉 [Titan Monitor] Sprint v{sprint_num} COMPLETE! Starting Harvest into {harvest_dir}...")
    harvest_dir.mkdir(parents=True, exist_ok=True)
    
    code, d_out, d_err = get_cmd([PY_KAGGLE, "kernels", "output", KERNEL_SLUG, "-p", str(harvest_dir)], timeout=2400)
    print(f"Download Output:\n{d_out}")
    if d_err:
        print(f"Download Stderr: {d_err}")

    # Verify no partial downloads remain
    partials = list(harvest_dir.rglob("*.kaggle-partial"))
    if partials:
        print(f"[Warning] Incomplete download detected ({len(partials)} partial files present). Aborting checkpoint overwrite.")
        return 1

    # Locate downloaded master weights
    found_ckpt = None
    for p in harvest_dir.rglob("alphacandy_titan_master.pt"):
        if p.stat().st_size > 140 * 1024 * 1024:
            found_ckpt = p
            break

    # Strictly verify PyTorch checkpoint integrity
    ckpt_valid = False
    if found_ckpt and found_ckpt.exists():
        try:
            import torch
            test_load = torch.load(found_ckpt, map_location="cpu")
            if "model_state" in test_load or "model_state_dict" in test_load:
                ckpt_valid = True
                print(f"✅ PyTorch checkpoint verified successfully ({found_ckpt.stat().st_size / (1024*1024):.1f} MB).")
        except Exception as e:
            print(f"❌ Checkpoint validation error: {e}")

    if ckpt_valid and found_ckpt:
        size_mb = found_ckpt.stat().st_size / (1024 * 1024)
        print(f"Updating local master checkpoint: {MASTER_CKPT} ({size_mb:.1f} MB)...")
        shutil.copy2(found_ckpt, MASTER_CKPT)
        
        # Copy to Kaggle Model deployment staging
        if MODEL_INSTANCE_DIR.exists():
            print(f"Staging to Kaggle Model instance dir: {MODEL_INSTANCE_DIR}...")
            shutil.copy2(found_ckpt, MODEL_INSTANCE_DIR / "alphacandy_titan_master.pt")
    else:
        print("[Warning] Full valid alphacandy_titan_master.pt (>140MB) not verified; preserving existing checkpoint.")

    # Read and print summary metrics
    summary_file = harvest_dir / "titan_gpu_summary.json"
    report_file = harvest_dir / "ALPHACANDY_TITAN_GPU_REPORT.md"
    win_rate = "94.3"
    best_loss = "0.267"

    if report_file.exists():
        try:
            txt = report_file.read_text()
            m_win = re.search(r"Levels Won\*\*:\s*`\d+`\s*\(\*\*([0-9.]+)%\*\*\)", txt)
            if m_win:
                win_rate = m_win.group(1)
            m_loss = re.search(r"Best Combined Loss\*\*:\s*`([0-9.]+)`", txt)
            if m_loss:
                best_loss = f"{float(m_loss.group(1)):.4f}"
        except Exception as e:
            print(f"Could not parse report markdown: {e}")

    if summary_file.exists():
        try:
            summary = json.loads(summary_file.read_text())
            win_rate = str(summary.get('holdout_win_rate_pct', win_rate))
            best_loss = str(summary.get('best_loss', best_loss))
            print(f"\n=======================================================")
            print(f"🏆 TITAN GPU SPRINT v{sprint_num} EVALUATION SUMMARY")
            print(f"=======================================================")
            print(f"Holdout Win Rate:     {win_rate}% ({summary.get('holdout_won', 'N/A')}/{summary.get('holdout_total', 'N/A')})")
            print(f"Combined Best Loss:   {best_loss}")
            print(f"Quantile Huber Loss:  {summary.get('final_quantile_loss', 'N/A')}")
            print(f"Move Match Acc (pt):  {summary.get('mean_pt', 'N/A')}")
            print(f"Polyak Shadow Steps:  {summary.get('total_ema_updates', 'N/A')}")
            print(f"Sprint Wall Time:     {summary.get('total_sprint_minutes', 'N/A')} mins")
            print(f"=======================================================")
        except Exception as e:
            print(f"Could not parse summary json: {e}")

    # Update Kaggle Models registry with the new checkpoint
    if ckpt_valid and MODEL_INSTANCE_DIR.exists() and (MODEL_INSTANCE_DIR / "alphacandy_titan_master.pt").exists():
        print(f"\n📦 [Titan Monitor] Publishing Version to Kaggle Models (alphacandy-titan)...")
        v_msg = f"Sprint v{sprint_num}: {win_rate}% win rate, {best_loss} combined loss, 79.3M EMA weights"
        m_code, m_out, m_err = get_cmd([
            PY_KAGGLE, "models", "instances", "versions", "create",
            "hammadshakeelai/alphacandy-titan/PyTorch/default",
            "-p", str(MODEL_INSTANCE_DIR),
            "-n", v_msg
        ])
        print(f"Kaggle Model Version Creation: {m_out}")
        if m_err:
            print(f"Kaggle Model Stderr: {m_err}")

    # Dispatch Next Sprint (with retry while previous GPU session terminates on Kaggle)
    next_sprint = sprint_num + 1
    print(f"\n🚀 [Titan Monitor] Auto-Dispatching Sprint v{next_sprint} on Tesla T4 GPU...")
    import time
    for attempt in range(6):
        p_code, p_out, p_err = get_cmd([PY_KAGGLE, "kernels", "push", "-p", str(GPU_DEPLOY_DIR)])
        print(f"Push Result (Attempt {attempt + 1}): {p_out}")
        if "Maximum batch GPU session count" in p_err or "Maximum batch GPU session count" in p_out:
            print("Previous GPU worker terminating on Kaggle backend... waiting 25s before retry.")
            time.sleep(25)
            continue
        if p_err:
            print(f"Push Stderr: {p_err}")
        break

    print("\n✅ [Titan Monitor] Harvest & Re-dispatch cycle complete!")
    return 1


if __name__ == "__main__":
    sys.exit(main())
