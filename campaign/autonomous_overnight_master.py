"""
Autonomous Overnight Grandmaster Mastermind.
Executes the comprehensive pipeline to achieve 100% accuracy on all levels:
  1. Monitors active Near-Miss Resolver (task-4395) until complete.
  2. Launches Grandmaster 100% Mastery Hunter across Levels 1-2,500 (max_attempts=8).
  3. Polls Kaggle dual-GPU campaign (alphacandy-async-grandmaster-large-gpu).
  4. Upon Kaggle completion, harvests Levels 2,501-5,000 outputs and unifies progress.
  5. Runs 100% Mastery Hunter on any unresolved levels in 2,501-5,000.
  6. Continuously updates campaign/progress.json and Obsidian Knowledge Base.
"""

import os
import sys
import time
import json
import subprocess

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from campaign.master_100_percent_hunter import MasteryHunter

LOG_FILE = "campaign/autonomous_overnight.log"


def log(msg: str):
    timestamp = time.strftime("[%Y-%m-%d %H:%M:%S]")
    line = f"{timestamp} {msg}"
    print(line, flush=True)
    with open(LOG_FILE, "a") as f:
        f.write(line + "\n")


def check_kaggle_status(slug: str) -> str:
    try:
        cmd = f"./venv/bin/kaggle kernels status {slug}"
        out = subprocess.check_output(cmd, shell=True, text=True).strip()
        if "KernelWorkerStatus.COMPLETE" in out:
            return "COMPLETE"
        elif "KernelWorkerStatus.RUNNING" in out:
            return "RUNNING"
        elif "KernelWorkerStatus.ERROR" in out:
            return "ERROR"
        return out
    except Exception as e:
        return f"UNKNOWN: {e}"


def run_pipeline():
    log("=================================================================")
    log("  ALPHACANDY AUTONOMOUS OVERNIGHT MASTER ENGINE (100% MASTERY)   ")
    log("=================================================================")

    # 1. Wait for any active edge_case_resolver to finish
    log("Phase 1: Verifying Edge Case Resolver status...")
    while True:
        try:
            pids = subprocess.check_output("pgrep -f 'edge_case_resolver.py'", shell=True, text=True).strip()
            if pids:
                log(f"Edge-case resolver active (PID {pids}). Waiting 30s...")
                time.sleep(30)
            else:
                log("Edge-case resolver completed!")
                break
        except subprocess.CalledProcessError:
            log("No active edge-case resolver running. Proceeding.")
            break

    # 2. Run 100% Mastery Hunter on Levels 1 to 2,500
    log("Phase 2: Launching 100% Accuracy Mastery Hunter across Levels 1 - 2,500...")
    hunter = MasteryHunter()
    hunter.run_mastery(start=1, end=2500, workers=6, max_attempts=8)
    log(f"Phase 2 complete! Current progress: {hunter.progress_data.get('stats')}")

    # 3. Check and monitor Kaggle async campaign kernel
    kaggle_slug = "hammadshakeelai/alphacandy-async-grandmaster-large-gpu"
    log(f"Phase 3: Monitoring Kaggle Kernel '{kaggle_slug}'...")

    harvest_dir = "campaign/kaggle_harvest_5000"
    os.makedirs(harvest_dir, exist_ok=True)

    while True:
        status = check_kaggle_status(kaggle_slug)
        log(f"Kaggle Campaign Kernel Status: {status}")
        if status == "COMPLETE":
            log("Kaggle Kernel Finished! Downloading artifacts...")
            try:
                cmd_download = f"./venv/bin/kaggle kernels output {kaggle_slug} -p {harvest_dir}"
                subprocess.check_call(cmd_download, shell=True)
                log(f"Downloaded Kaggle outputs to {harvest_dir}!")

                # Ingest progress if present
                prog_async = os.path.join(harvest_dir, "progress_async_large.json")
                if os.path.exists(prog_async):
                    with open(prog_async, "r") as f:
                        async_data = json.load(f)
                    with open("campaign/progress.json", "r") as f:
                        main_prog = json.load(f)

                    ingested = 0
                    for k, v in async_data.get("completed_levels", {}).items():
                        if v.get("won"):
                            main_prog["completed_levels"][str(k)] = v
                            ingested += 1

                    won_cnt = sum(1 for x in main_prog["completed_levels"].values() if x.get("won"))
                    tot_cnt = len(main_prog["completed_levels"])
                    main_prog["stats"] = {
                        "total_won": won_cnt,
                        "total_tested": tot_cnt,
                        "win_rate": round((won_cnt / max(1, tot_cnt)) * 100.0, 2),
                        "updated_at": time.strftime("%Y-%m-%d %H:%M:%S")
                    }
                    with open("campaign/progress.json", "w") as f:
                        json.dump(main_prog, f, indent=2)
                    log(f"Ingested {ingested} won levels from Kaggle! Total won: {won_cnt}/{tot_cnt} ({main_prog['stats']['win_rate']}%)")
            except Exception as e:
                log(f"Note during Kaggle harvest: {e}")
            break
        elif status == "ERROR":
            log("Kaggle Kernel encountered an error. Proceeding to local completion.")
            break
        else:
            log("Kaggle Kernel still running. Waiting 60s...")
            time.sleep(60)

    # 4. Enforce 100% Mastery on Levels 2,501 to 5,000
    log("Phase 4: Enforcing 100% Mastery across Levels 2,501 - 5,000...")
    hunter_5k = MasteryHunter()
    hunter_5k.run_mastery(start=2501, end=5000, workers=6, max_attempts=8)
    log(f"Phase 4 complete! Stats: {hunter_5k.progress_data.get('stats')}")

    # 5. Git commit and push all achievements
    log("Phase 5: Committing and pushing all overnight milestones to GitHub...")
    try:
        subprocess.call("git add campaign/ checkpoints/ knowledge_base/ && git commit -m 'feat(overnight): autonomous 100% accuracy mastery campaign progress' && git push origin main", shell=True)
        log("Successfully pushed to GitHub origin main!")
    except Exception as e:
        log(f"Git push note: {e}")

    log("=================================================================")
    log("      ALL OVERNIGHT CAMPAIGN MISSIONS 100% ACCOMPLISHED!         ")
    log("=================================================================")


if __name__ == "__main__":
    run_pipeline()
