#!/usr/bin/env python3
"""
Autonomous Multi-Tier Fleet Orchestrator
=======================================
Performs a lightweight (<2s), high-efficiency health check and dispatch cycle
across all four compute tiers:
  1. Kaggle GPU: Bounded 60–75 min sprints for AlphaCandyTitanNet (79.3M)
  2. Kaggle CPU: Continuous mass mastery hunt (4-vCPU)
  3. Local CPU: Multi-core fortress assault targeting isolated near-miss levels
  4. GitHub Actions: Headless cloud notebook verification (2 vCPU, 7 GB RAM)
"""

import os
import sys
import time
import json
import shutil
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

LOG_FILE = REPO_ROOT / "runs" / "orchestrator_health.jsonl"


def log_event(event_type: str, data: dict):
    """Append structured telemetry record."""
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        "event": event_type,
        **data
    }
    with open(LOG_FILE, "a") as f:
        f.write(json.dumps(record) + "\n")
    print(f"[{record['timestamp']}] [{event_type}] {data}", flush=True)


def get_cmd_output(cmd, timeout=15):
    try:
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=timeout)
        return res.returncode, res.stdout.strip(), res.stderr.strip()
    except Exception as e:
        return -1, "", str(e)


def check_local_strike():
    """Ensure local multi-core fortress assault is active."""
    code, out, _ = get_cmd_output(["pgrep", "-f", "local_mastery_strike.py"])
    if code == 0 and out:
        pids = out.split()
        return "RUNNING", f"{len(pids)} worker processes active"
    
    # Launch new local strike
    strike_script = REPO_ROOT / "alphacandy_zero" / "local_mastery_strike.py"
    log_path = REPO_ROOT / "runs" / "local_strike.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    
    py = str(REPO_ROOT / "venv" / "bin" / "python3")
    if not Path(py).exists():
        py = sys.executable

    with open(log_path, "a") as f_out:
        p = subprocess.Popen(
            [py, str(strike_script)],
            stdout=f_out,
            stderr=subprocess.STDOUT,
            cwd=str(REPO_ROOT),
            start_new_session=True,
        )
    
    return "LAUNCHED", f"PID {p.pid} logging to {log_path.name}"


def check_kaggle_gpu():
    """Check Kaggle GPU flagship trainer status."""
    py = str(REPO_ROOT / "venv" / "bin" / "kaggle")
    code, out, _ = get_cmd_output([py, "kernels", "status", "hammadshakeelai/alphacandy-titan-50m-gpu-mastery"])
    if "RUNNING" in out:
        return "RUNNING", "GPU sprint active"
    elif "COMPLETE" in out:
        return "COMPLETE", "GPU sprint finished; ready for harvest or next sprint"
    elif "QUEUED" in out:
        return "QUEUED", "GPU kernel queued"
    else:
        return "IDLE/UNKNOWN", out or "No status"


KAGGLE_CPU_FLEET = [
    ("CPU-1 (Mastery Hunter)", "hammadshakeelai/alphacandy-mastery-hunter-100-percent", "kaggle_deploy/mastery_100_kernel"),
    ("CPU-2 (Arena & PLR)", "hammadshakeelai/alphacandy-zero-cpu-arena-and-plr-trainer", "kaggle_deploy/zero_cpu_arena_and_plr"),
    ("CPU-3 (Oracle 1001-1500)", "hammadshakeelai/alphacandy-zero-oracle-shard-1001-to-1500", "kaggle_deploy/zero_oracle_1001_1500"),
    ("CPU-4 (Oracle 1501-2000)", "hammadshakeelai/alphacandy-zero-oracle-shard-1501-to-2000", "kaggle_deploy/zero_oracle_1501_2000"),
    ("CPU-5 (Oracle 651-1000)", "hammadshakeelai/alphacandy-zero-oracle-shard-651-to-1000", "kaggle_deploy/zero_oracle_651_1000"),
]


def check_and_maintain_kaggle_cpu_fleet():
    """Check all 5 Kaggle 4-vCPU instances and auto-relaunch any completed or idle workers."""
    py = str(REPO_ROOT / "venv" / "bin" / "kaggle")
    fleet_statuses = {}
    active_count = 0
    
    for name, slug, deploy_dir in KAGGLE_CPU_FLEET:
        code, out, _ = get_cmd_output([py, "kernels", "status", slug], timeout=15)
        st = "UNKNOWN"
        if "RUNNING" in out:
            st = "RUNNING"
            active_count += 1
        elif "QUEUED" in out:
            st = "QUEUED"
            active_count += 1
        elif "COMPLETE" in out:
            st = "COMPLETE"
            # Auto-relaunch completed worker so all 5 instances stay active continuously (30-50 min runs)
            deploy_path = REPO_ROOT / deploy_dir
            if deploy_path.exists():
                r_code, r_out, _ = get_cmd_output([py, "kernels", "push", "-p", str(deploy_path)], timeout=30)
                if r_code == 0:
                    st = "RELAUNCHED"
                    active_count += 1
        else:
            st = out or "IDLE"
            
        fleet_statuses[name] = {"slug": slug, "status": st, "raw": out}

    summary_msg = f"{active_count}/5 instances active"
    overall_status = "RUNNING" if active_count >= 3 else "PARTIAL"
    return overall_status, summary_msg, fleet_statuses


def check_github_actions():
    """Check GitHub Actions cloud notebook runner status and auto-dispatch if idle."""
    if not shutil.which("gh"):
        return "UNAVAILABLE", "gh CLI not installed"
    
    code, out, _ = get_cmd_output([
        "gh", "run", "list",
        "--workflow=cloud_notebook_runner.yml",
        "--limit=1",
        "--json", "databaseId,status,conclusion,createdAt"
    ])
    try:
        data = json.loads(out)
        if data and len(data) > 0:
            latest = data[0]
            st = (latest.get("status") or "").upper()
            conc = latest.get("conclusion")
            rid = latest.get("databaseId")
            if st in ("IN_PROGRESS", "QUEUED", "WAITING", "REQUESTED", "PENDING"):
                return st, f"Run #{rid} active on GitHub Actions 2-vCPU runner"
            # If completed, auto-trigger next verification batch
            d_code, _, _ = get_cmd_output([
                "gh", "workflow", "run", "cloud_notebook_runner.yml",
                "-f", "notebook_path=notebooks/smoke_test.ipynb",
                "-f", 'parameters={"num_trials": 5000, "seed": 2500}'
            ])
            if d_code == 0:
                return "DISPATCHED", f"Previous Run #{rid} ({conc}); triggered new 2-vCPU cloud run"
            return st, f"Run #{rid} (conclusion: {conc})"
    except Exception:
        pass
    return "IDLE", "No active runs"


def run_orchestration_cycle():
    print("=" * 65)
    print("⚡ AUTONOMOUS FLEET ORCHESTRATION HEALTH CHECK")
    print("=" * 65)
    
    # 1. Local CPU
    local_st, local_msg = check_local_strike()
    print(f"🖥️  Local Machine CPU:   [{local_st}] {local_msg}")
    
    # 2. Kaggle GPU
    gpu_st, gpu_msg = check_kaggle_gpu()
    print(f"🚀 Kaggle Tesla T4 GPU: [{gpu_st}] {gpu_msg}")
    
    # 3. Kaggle 5x 4-vCPU Fleet
    kcpu_st, kcpu_msg, kcpu_details = check_and_maintain_kaggle_cpu_fleet()
    print(f"☁️  Kaggle 5x 4-vCPU Fleet: [{kcpu_st}] {kcpu_msg}")
    for inst_name, inst_info in kcpu_details.items():
        print(f"     • {inst_name:<25}: [{inst_info['status']}]")
    
    # 4. GitHub Actions
    gha_st, gha_msg = check_github_actions()
    print(f"🐙 GitHub Actions Cloud: [{gha_st}] {gha_msg}")
    print("=" * 65)

    summary = {
        "local_cpu": {"status": local_st, "details": local_msg},
        "kaggle_gpu": {"status": gpu_st, "details": gpu_msg},
        "kaggle_cpu_5x": {"status": kcpu_st, "details": kcpu_msg, "instances": kcpu_details},
        "github_actions": {"status": gha_st, "details": gha_msg}
    }
    log_event("FLEET_CYCLE", summary)
    return summary


if __name__ == "__main__":
    run_orchestration_cycle()
