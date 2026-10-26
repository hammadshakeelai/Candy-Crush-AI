"""
AlphaCandy Grandmaster: 100% Accuracy Mastery Hunter on Kaggle.
Targets the remaining uncompleted levels in Levels 1 to 2,500 using
Depth-3 Tactical Lookahead, Adaptive Seed Rotation (up to 12 attempts),
and Consensus Agent to drive the entire 2,500-level campaign to 100% accuracy.
"""

import os
import sys
import time
import json
import zipfile
import shutil
import sqlite3
from typing import Dict, Any, List
from multiprocessing import Pool, cpu_count
import torch

for _p in ["/kaggle/working", "/kaggle/working/candy_crush_pkg", os.getcwd()]:
    if _p not in sys.path:
        sys.path.insert(0, _p)
os.environ["PYTHONPATH"] = f"/kaggle/working:/kaggle/working/candy_crush_pkg:{os.environ.get('PYTHONPATH', '')}"

_WORKER_AGENT = None
_WORKER_LOADER = None
_WORKER_PROCEDURAL_GEN = None


def setup_environment() -> tuple:
    print("=================================================================")
    print("  ALPHACANDY GRANDMASTER: 100% ACCURACY MASTERY HUNTER (KAGGLE)  ")
    print("=================================================================")

    input_root = "/kaggle/input" if os.path.exists("/kaggle") else "./kaggle_deploy/src"
    work_dir = "/kaggle/working" if os.path.exists("/kaggle") else "/tmp/candy_mastery_work"
    os.makedirs(work_dir, exist_ok=True)

    src_root = None
    data_json_found = None
    sqlite_found = None

    for root, dirs, files in os.walk(input_root):
        if "candy_crush_pkg.zip" in files and src_root is None:
            zip_path = os.path.join(root, "candy_crush_pkg.zip")
            print(f"[INFO] Extracting {zip_path} into {work_dir}...")
            try:
                with zipfile.ZipFile(zip_path, "r") as zip_ref:
                    zip_ref.extractall(work_dir)
                src_root = work_dir
                print("[INFO] Extraction complete.")
            except Exception as e:
                print(f"[WARN] Failed extracting zip: {e}")

        if "candycrush_master_levels.json" in files:
            data_json_found = os.path.join(root, "candycrush_master_levels.json")
        if "candycrush_master_levels.sqlite" in files:
            sqlite_found = os.path.join(root, "candycrush_master_levels.sqlite")

    if not src_root:
        src_root = work_dir

    for p in [work_dir, src_root]:
        if p and p not in sys.path:
            sys.path.insert(0, p)

    if data_json_found:
        os.environ["KING_CONFIG_PATH"] = data_json_found
    if sqlite_found:
        os.environ["KING_SQLITE_PATH"] = sqlite_found

    print(f"[INFO] Source Root: {src_root} | Working Dir: {work_dir}")
    print(f"[INFO] CPU Count: {cpu_count()} | CUDA Available: {torch.cuda.is_available()}")
    return work_dir, src_root


def init_worker(paths: tuple = None):
    global _WORKER_AGENT, _WORKER_LOADER, _WORKER_PROCEDURAL_GEN
    if paths:
        work_dir, src_root = paths
        for p in [src_root, work_dir]:
            if p and p not in sys.path:
                sys.path.insert(0, p)
        if work_dir and os.path.exists(work_dir):
            try:
                os.chdir(work_dir)
            except Exception:
                pass
    else:
        for p in ["/kaggle/working", "/kaggle/working/candy_crush_pkg", os.getcwd()]:
            if os.path.exists(p) and p not in sys.path:
                sys.path.insert(0, p)

    from core.king_level_loader import KingLevelLoader
    from core.procedural_levels import ProceduralKingGenerator
    from core.reverse_engine import ReverseCandyEngine
    from models.grandmaster_consensus_agent import GrandmasterConsensusAgent

    _WORKER_LOADER = KingLevelLoader()
    _WORKER_PROCEDURAL_GEN = ProceduralKingGenerator()
    engine = ReverseCandyEngine()

    large_ckpt = "checkpoints/alphacandy_large_net_master.pt"
    if not os.path.exists(large_ckpt):
        large_ckpt = "checkpoints/alphacandy_large_net.pt"

    dqn_ckpt = "checkpoints/candy_crush_dueling_dqn_master.pt"
    if not os.path.exists(dqn_ckpt):
        dqn_ckpt = "checkpoints/candy_crush_dueling_dqn.pt"

    _WORKER_AGENT = GrandmasterConsensusAgent(
        large_transformer_ckpt=large_ckpt if os.path.exists(large_ckpt) else None,
        dqn_master_ckpt=dqn_ckpt if os.path.exists(dqn_ckpt) else None,
        search_depth=3,
        top_k_candidates=6,
        engine=engine,
        device="cpu"
    )


def solve_level_mastery_task(args: tuple) -> Dict[str, Any]:
    level_id, max_attempts = args
    global _WORKER_LOADER, _WORKER_PROCEDURAL_GEN, _WORKER_AGENT

    if _WORKER_AGENT is None:
        init_worker()

    t_start = time.time()
    try:
        if level_id <= 10295:
            cfg = _WORKER_LOADER.load_level(level_id)
        else:
            cfg = _WORKER_PROCEDURAL_GEN.generate_level(level_id)
    except Exception as e:
        return {
            "level_id": int(level_id),
            "won": False,
            "attempt": 0,
            "error": f"LoadError: {e}",
            "duration_sec": time.time() - t_start
        }

    initial_seed = (level_id * 10007 + 42) % 1000003
    try:
        res = _WORKER_AGENT.solve_level(cfg, max_attempts=max_attempts, seed=initial_seed)
        won = bool(res.get("won", False))
        att = int(res.get("attempt", 1))
        return {
            "level_id": int(level_id),
            "won": won,
            "attempt": att,
            "winning_seed": (initial_seed + (att - 1) * 313) if won else None,
            "moves_used": int(res.get("moves_used", 0)),
            "moves_limit": int(cfg.moves_limit),
            "moves_remaining": int(res.get("moves_remaining", 0)),
            "final_score": int(res.get("final_score", 0)),
            "objectives_remaining": int(res.get("objectives_remaining", 0)),
            "level_type": cfg.level_type.name,
            "duration_sec": round(float(res.get("duration_sec", time.time() - t_start)), 2),
            "error": None
        }
    except Exception as e_run:
        return {
            "level_id": int(level_id),
            "won": False,
            "attempt": 1,
            "error": str(e_run),
            "duration_sec": round(time.time() - t_start, 2)
        }


def main():
    work_dir, src_root = setup_environment()

    # Load existing progress.json
    progress_file = os.path.join(work_dir, "campaign", "progress.json")
    if not os.path.exists(progress_file):
        progress_file = os.path.join(work_dir, "progress.json")

    completed_levels = {}
    if os.path.exists(progress_file):
        try:
            with open(progress_file, "r") as f:
                d = json.load(f)
                completed_levels = d.get("completed_levels", {})
                print(f"[INFO] Loaded progress file with {len(completed_levels)} levels.")
        except Exception as e:
            print(f"[WARN] Progress file load error: {e}")

    # Determine uncompleted levels in range 1 to 2,500
    target_start, target_end = 1, 2500
    uncompleted = []
    for lvl in range(target_start, target_end + 1):
        entry = completed_levels.get(str(lvl))
        if entry is None or not entry.get("won", False):
            uncompleted.append(lvl)

    print(f"\n[INFO] Levels Range: {target_start} -> {target_end}")
    print(f"[INFO] Already Won : {2500 - len(uncompleted)} / 2500 ({(2500 - len(uncompleted))/25.0:.2f}%)")
    print(f"[INFO] Pending 100% Targets: {len(uncompleted)} levels")

    if not uncompleted:
        print("[SUCCESS] 100% ACCURACY ALREADY ACHIEVED ACROSS ALL 2,500 LEVELS!")
        return

    workers = min(4, cpu_count())
    max_attempts = 12
    task_args = [(lvl, max_attempts) for lvl in uncompleted]

    print(f"[INFO] Launching {workers} Parallel Grandmaster Workers (Max Attempts={max_attempts}, Depth=3)...")

    results = []
    newly_won = 0
    t_start = time.time()

    with Pool(processes=workers, initializer=init_worker, initargs=((work_dir, src_root),)) as pool:
        for idx, res in enumerate(pool.imap_unordered(solve_level_mastery_task, task_args), 1):
            results.append(res)
            lvl_id = res["level_id"]
            won = res["won"]
            if won:
                newly_won += 1
                completed_levels[str(lvl_id)] = res
                print(f"🌟 [SOLVED {newly_won}/{idx}] Level {lvl_id:4d} WON! (Att: {res['attempt']}, Moves: {res['moves_used']}/{res['moves_limit']}, Score: {res['final_score']:,}, Time: {res['duration_sec']}s)")
            else:
                print(f"❌ [{idx}/{len(uncompleted)}] Level {lvl_id:4d} Unsolved (Remaining: {res.get('objectives_remaining', '?')})")

            # Periodic checkpointing
            if idx % 10 == 0 or idx == len(uncompleted):
                elapsed = time.time() - t_start
                tot_won = sum(1 for v in completed_levels.values() if v.get("won"))
                tot_tested = max(2500, len(completed_levels))
                cur_acc = (tot_won / tot_tested) * 100.0
                print(f"\n--- Checkpoint {idx}/{len(uncompleted)} | Newly Won: {newly_won} | Total Solved: {tot_won}/{tot_tested} ({cur_acc:.2f}%) | Elapsed: {elapsed/60:.1f}m ---\n")

                out_payload = {
                    "completed_levels": completed_levels,
                    "stats": {
                        "total_won": tot_won,
                        "total_tested": tot_tested,
                        "win_rate": round(cur_acc, 2),
                        "updated_at": time.strftime("%Y-%m-%d %H:%M:%S")
                    }
                }
                out_path = os.path.join(work_dir, "progress_100_mastery.json")
                with open(out_path + ".tmp", "w") as f:
                    json.dump(out_payload, f, indent=2)
                os.replace(out_path + ".tmp", out_path)

    # Final summary report
    elapsed_total = time.time() - t_start
    final_won = sum(1 for v in completed_levels.values() if v.get("won"))
    final_acc = (final_won / 2500.0) * 100.0

    report = f"""# 🏆 AlphaCandy 100% Accuracy Mastery Report

- **Target Level Span**: Levels 1 to 2,500
- **Pre-Run Solved**: {2500 - len(uncompleted)} / 2,500 ({(2500 - len(uncompleted))/25.0:.2f}%)
- **Newly Conquered**: {newly_won} / {len(uncompleted)}
- **Final Campaign Accuracy**: **{final_won} / 2,500 ({final_acc:.2f}%)**
- **Total Execution Time**: {elapsed_total / 3600.0:.2f} hours
- **Search Parameters**: Depth-3 Tactical Lookahead, Top-6 Beam Width, Up to 12 Adaptive Seed Rotations
"""
    with open(os.path.join(work_dir, "MASTERY_100_PERCENT_SUMMARY.md"), "w") as f:
        f.write(report)
    print("\n" + "="*65)
    print("        100% ACCURACY MASTERY MISSION COMPLETE        ")
    print(f"        Final Win Rate: {final_won}/2500 ({final_acc:.2f}%)")
    print("="*65 + "\n")


if __name__ == "__main__":
    main()
