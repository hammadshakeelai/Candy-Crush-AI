"""
AlphaCandy Grandmaster: Frontier Campaign Runner (Levels 7,501 to 10,000).
Evaluates 2,500 unseen levels up to King's 10,000 Milestone on Kaggle 4 vCPUs.
Features:
  - 4 Parallel Workers with context isolation
  - Master Weights: 14.4M Spatial Attention Transformer + Dueling DQN Master
  - Depth-2 Tactical Lookahead with Top-5 Candidate Beam Pruning
  - Periodic atomic checkpoints to progress_frontier_7501_10000.json
  - Auto-generated Markdown Executive Report
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
    print("==================================================================")
    print("  ALPHACANDY GRANDMASTER: FRONTIER RUNNER (LEVELS 7,501 - 10,000) ")
    print("==================================================================")

    input_root = "/kaggle/input" if os.path.exists("/kaggle") else "./kaggle_deploy/src"
    work_dir = "/kaggle/working" if os.path.exists("/kaggle") else "/tmp/candy_frontier_75k"
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

    print(f"[INFO] CPU Count: {cpu_count()} | Working Dir: {work_dir}")
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
        search_depth=2,
        top_k_candidates=5,
        engine=engine,
        device="cpu"
    )


def solve_frontier_level_task(args: tuple) -> Dict[str, Any]:
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
            "duration_sec": round(time.time() - t_start, 2)
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
    start_lvl, end_lvl = 7501, 10000
    workers = min(4, cpu_count())
    max_attempts = 2
    task_args = [(lvl, max_attempts) for lvl in range(start_lvl, end_lvl + 1)]
    total_levels = len(task_args)

    print(f"\n[INFO] Starting Frontier Campaign: Levels {start_lvl} to {end_lvl} ({total_levels} levels)")
    print(f"[INFO] Workers={workers} | Max Attempts={max_attempts} | Lookahead Depth=2\n")

    results = []
    won_count = 0
    t_start = time.time()

    with Pool(processes=workers, initializer=init_worker, initargs=((work_dir, src_root),)) as pool:
        for idx, res in enumerate(pool.imap(solve_frontier_level_task, task_args, chunksize=4), 1):
            results.append(res)
            if res["won"]:
                won_count += 1

            if idx % 25 == 0 or idx == total_levels:
                elapsed = time.time() - t_start
                rate = (won_count / idx) * 100.0
                speed = idx / elapsed
                remain = (total_levels - idx) / speed if speed > 0 else 0
                print(f"[{idx:4d}/{total_levels}] Solved: {won_count:4d} ({rate:5.1f}%) | "
                      f"Speed: {speed:4.1f} lvl/s | Elapsed: {elapsed/60.0:4.1f}m | ETA: {remain/60.0:4.1f}m")

                # Atomic checkpoint
                out_path = os.path.join(work_dir, "progress_frontier_7501_10000.json")
                with open(out_path + ".tmp", "w") as f:
                    json.dump(results, f, indent=2)
                os.replace(out_path + ".tmp", out_path)

    elapsed_total = time.time() - t_start
    final_rate = (won_count / total_levels) * 100.0
    report = f"""# 🚀 AlphaCandy Frontier Benchmark: Levels 7,501 to 10,000

- **Levels Tested**: {total_levels:,}
- **Levels Won**: {won_count:,} / {total_levels:,} (**{final_rate:.2f}%**)
- **Total Duration**: {elapsed_total / 60.0:.1f} minutes ({elapsed_total / total_levels:.3f} s/level)
- **Model Architecture**: Grandmaster Consensus Agent (14.4M Spatial Attention Transformer + Double Dueling DQN Master)
- **Checkpoints**: Synchronized to `progress_frontier_7501_10000.json`
"""
    with open(os.path.join(work_dir, "FRONTIER_7501_10000_REPORT.md"), "w") as f:
        f.write(report)
    print("\n" + "="*65)
    print(f"   FRONTIER CAMPAIGN 7,501 - 10,000 COMPLETE: {won_count}/{total_levels} ({final_rate:.2f}%)")
    print("="*65 + "\n")


if __name__ == "__main__":
    main()
