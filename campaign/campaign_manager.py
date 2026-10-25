"""
Autonomous Candy Crush 10,000+ Level Campaign Runner.
Orchestrates adaptive multi-attempt solving across all 10,295 official King levels
and transitions to procedural levels up to 25,000+.
Utilizes multi-core parallel processing (12 threads) for high throughput.
"""

import os
import sys

# Ensure project root is in python path for multiprocessing workers
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import json
import time
import argparse
from typing import Dict, Any, List, Optional
from multiprocessing import Pool, cpu_count

from core.king_level_loader import KingLevelLoader
from core.procedural_levels import ProceduralKingGenerator
from core.reverse_engine import ReverseCandyEngine, ReverseEngineState
from models.superhuman_solver import SuperhumanSolver


PROGRESS_FILE = "campaign/progress.json"
_GLOBAL_LOADER = None
_GLOBAL_PROCEDURAL_GEN = None


def get_level_config(level_id: int):
    global _GLOBAL_LOADER, _GLOBAL_PROCEDURAL_GEN
    if level_id <= 10295:
        if _GLOBAL_LOADER is None:
            _GLOBAL_LOADER = KingLevelLoader()
        return _GLOBAL_LOADER.load_level(level_id)
    else:
        if _GLOBAL_PROCEDURAL_GEN is None:
            _GLOBAL_PROCEDURAL_GEN = ProceduralKingGenerator()
        return _GLOBAL_PROCEDURAL_GEN.generate_level(level_id)


def solve_single_level_adaptive(level_arg: Any) -> Dict[str, Any]:
    """
    Solves a single level using adaptive multi-attempt strategies.
    If Attempt 1 does not win, applies seed rerolls and deeper beam lookahead.
    Guarantees maximum win rate without live penalty.
    """
    if isinstance(level_arg, (tuple, list)):
        level_id = int(level_arg[0])
        max_attempts = int(level_arg[1]) if len(level_arg) > 1 else 3
    else:
        level_id = int(level_arg)
        max_attempts = 3

    try:
        cfg = get_level_config(level_id)
    except Exception as e:
        return {
            "level_id": level_id,
            "won": False,
            "attempts": 0,
            "error": str(e)
        }

    engine = ReverseCandyEngine()
    solver = SuperhumanSolver(engine=engine, search_depth=2, top_k_candidates=5)
    res = solver.solve_level(cfg, max_attempts=max_attempts)

    return {
        "level_id": int(level_id),
        "won": bool(res["won"]),
        "attempts": int(res.get("attempts", 1)),
        "moves_used": int(res.get("moves_used", 0)),
        "moves_limit": int(cfg.moves_limit),
        "score": int(res.get("final_score", 0)),
        "level_type": cfg.level_type.name,
        "time_sec": float(res.get("duration_sec", 0.0))
    }


class CampaignManager:
    """Manages the full 10,000 level progression and persistence."""

    def __init__(self, progress_path: str = PROGRESS_FILE):
        self.progress_path = progress_path
        self.progress: Dict[str, Any] = self._load_progress()

    def _load_progress(self) -> Dict[str, Any]:
        if os.path.exists(self.progress_path):
            with open(self.progress_path, "r") as f:
                return json.load(f)
        return {"completed_levels": {}, "stats": {"total_won": 0, "total_tested": 0}}

    def save_progress(self) -> None:
        total_won = sum(1 for v in self.progress.get("completed_levels", {}).values() if v.get("won"))
        total_tested = len(self.progress.get("completed_levels", {}))
        self.progress["stats"] = {
            "total_won": total_won,
            "total_tested": total_tested,
            "win_rate": round((total_won / max(1, total_tested)) * 100.0, 2)
        }
        with open(self.progress_path, "w") as f:
            json.dump(self.progress, f, indent=2)

    def run_campaign(
        self,
        start_lvl: Optional[int] = None,
        end_lvl: Optional[int] = None,
        levels: Optional[List[int]] = None,
        workers: int = 6,
        max_attempts: int = 3,
        force: bool = False
    ):
        print(f"=== LAUNCHING CANDY CRUSH 10,000 CAMPAIGN ===")
        if levels is not None:
            candidate_levels = levels
            print(f"Levels: Custom list of {len(levels)} levels | Max Attempts: {max_attempts} | Parallel Workers: {workers}")
        else:
            start_lvl = start_lvl if start_lvl is not None else 1
            end_lvl = end_lvl if end_lvl is not None else 25
            candidate_levels = list(range(start_lvl, end_lvl + 1))
            print(f"Levels: {start_lvl} to {end_lvl} | Max Attempts: {max_attempts} | Parallel Workers: {workers}")

        levels_to_run = [
            lvl for lvl in candidate_levels
            if force or str(lvl) not in self.progress["completed_levels"] or not self.progress["completed_levels"][str(lvl)]["won"]
        ]

        print(f"Total remaining levels to beat in batch: {len(levels_to_run)}\n")
        if not levels_to_run:
            print("All requested levels in range already completed!")
            return

        t0 = time.time()
        completed_count = 0
        batch_wins = 0

        task_args = [(lvl, max_attempts) for lvl in levels_to_run]

        with Pool(processes=workers) as pool:
            for result in pool.imap_unordered(solve_single_level_adaptive, task_args, chunksize=1):
                lvl_id = result["level_id"]
                won = result["won"]
                self.progress["completed_levels"][str(lvl_id)] = result

                if won:
                    batch_wins += 1

                completed_count += 1
                status = "WON [100% SUCCESS]" if won else "FAILED"
                attempts = result.get("attempts", 1)
                score = result.get("score", 0)
                moves = result.get("moves_used", 0)
                moves_lim = result.get("moves_limit", 0)

                print(f"[{completed_count:4d}/{len(levels_to_run)}] Level {lvl_id:5d} | {status:18s} | Attempts: {attempts} | Moves: {moves:2d}/{moves_lim:2d} | Score: {score:8,d}")

                # Save progress periodically every 25 levels
                if completed_count % 25 == 0:
                    self.save_progress()

        self.save_progress()
        total_time = time.time() - t0
        win_rate = (batch_wins / max(1, completed_count)) * 100.0

        print("\n==============================================================")
        print("                 BATCH CAMPAIGN SUMMARY                       ")
        print("==============================================================")
        print(f"Levels Processed : {completed_count}")
        print(f"Levels Won       : {batch_wins}/{completed_count} ({win_rate:.1f}%)")
        print(f"Elapsed Time     : {total_time:.2f}s (Avg {total_time/max(1, completed_count):.2f}s / level)")
        print(f"Progress Saved To: {self.progress_path}")
        print("==============================================================\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Candy Crush 10,000 Campaign Runner")
    parser.add_argument("--start", type=int, default=None, help="Start level")
    parser.add_argument("--end", type=int, default=None, help="End level")
    parser.add_argument("--levels", type=str, default=None, help="Comma-separated list of level IDs")
    parser.add_argument("--workers", type=int, default=6, help="Parallel worker threads")
    parser.add_argument("--max-attempts", type=int, default=3, help="Max retry attempts per level")
    parser.add_argument("--force", action="store_true", help="Force replay even if already won")
    args = parser.parse_args()

    levels_list = None
    if args.levels:
        levels_list = [int(x.strip()) for x in args.levels.split(",") if x.strip()]

    mgr = CampaignManager()
    mgr.run_campaign(
        start_lvl=args.start or 1 if levels_list is None else None,
        end_lvl=args.end or 25 if levels_list is None else None,
        levels=levels_list,
        workers=args.workers,
        max_attempts=args.max_attempts,
        force=args.force
    )
