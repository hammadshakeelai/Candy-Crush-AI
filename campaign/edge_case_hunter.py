"""
Grandmaster Edge-Case Campaign Hunter (Levels 301 - 1,000+).
Executes high-throughput automated campaign evaluation using the Grandmaster Consensus Agent:
  - AlphaCandy Large Transformer (~14.4M parameters, checkpoints/alphacandy_large_net_master.pt)
  - Double Dueling DQN Master (checkpoints/candy_crush_dueling_dqn_master.pt)
  - CANDYRL Expert Ensemble Heuristics
  - Adaptive Depth-2 Lookahead with Beam Pruning

Tracks edge-case classifications:
  - FIRST_TRY_WIN: Solved on Attempt 1
  - ADAPTIVE_WIN: Overcame bottleneck on Attempt 2
  - NEAR_MISS_EDGE_CASE: Lost with <= 3 objectives remaining
  - HARD_BLOCKER_EDGE_CASE: Lost with > 3 objectives remaining (complex blockers / spawners)

Saves incrementally to:
  - campaign/edge_cases_301_1000.json
  - campaign/progress.json
"""

import os
import sys
import json
import time
import argparse
from typing import Dict, Any, List, Optional
from multiprocessing import Pool, cpu_count
import torch

# Ensure project root is in python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.king_level_loader import KingLevelLoader
from core.procedural_levels import ProceduralKingGenerator
from core.reverse_engine import ReverseCandyEngine, ReverseEngineState
from models.grandmaster_consensus_agent import GrandmasterConsensusAgent

EDGE_CASES_FILE = "campaign/edge_cases_301_1000.json"
PROGRESS_FILE = "campaign/progress.json"

# Process-local singletons for workers
_WORKER_LOADER: Optional[KingLevelLoader] = None
_WORKER_PROCEDURAL_GEN: Optional[ProceduralKingGenerator] = None
_WORKER_AGENT: Optional[GrandmasterConsensusAgent] = None
_WORKER_ENGINE: Optional[ReverseCandyEngine] = None


def init_worker():
    """Initializes per-worker PyTorch thread limits and lazy loaders."""
    torch.set_num_threads(1)
    global _WORKER_LOADER, _WORKER_PROCEDURAL_GEN, _WORKER_AGENT, _WORKER_ENGINE
    _WORKER_LOADER = KingLevelLoader()
    _WORKER_PROCEDURAL_GEN = ProceduralKingGenerator()
    _WORKER_ENGINE = ReverseCandyEngine()
    _WORKER_AGENT = GrandmasterConsensusAgent(
        large_transformer_ckpt="checkpoints/alphacandy_large_net_master.pt",
        dqn_master_ckpt="checkpoints/candy_crush_dueling_dqn_master.pt",
        search_depth=2,
        top_k_candidates=4,
        engine=_WORKER_ENGINE,
        device="cpu"
    )


def solve_level_worker(args: tuple) -> Dict[str, Any]:
    """Worker task function to solve a single level with telemetry."""
    level_id, max_attempts = args
    global _WORKER_LOADER, _WORKER_PROCEDURAL_GEN, _WORKER_AGENT, _WORKER_ENGINE

    if _WORKER_AGENT is None:
        init_worker()

    try:
        if level_id <= 10295:
            cfg = _WORKER_LOADER.load_level(level_id)
        else:
            cfg = _WORKER_PROCEDURAL_GEN.generate_level(level_id)
    except Exception as e:
        return {
            "level_id": int(level_id),
            "won": False,
            "attempts": 0,
            "error": str(e),
            "classification": "ERROR"
        }

    res = _WORKER_AGENT.solve_level(cfg, max_attempts=max_attempts)

    won = bool(res.get("won", False))
    attempts = int(res.get("attempt", 1))
    obj_left = int(res.get("objectives_remaining", 0))

    if won:
        if attempts == 1:
            classification = "FIRST_TRY_WIN"
        else:
            classification = "ADAPTIVE_WIN"
    else:
        if obj_left <= 3:
            classification = "NEAR_MISS_EDGE_CASE"
        else:
            classification = "HARD_BLOCKER_EDGE_CASE"

    return {
        "level_id": int(level_id),
        "won": won,
        "attempts": attempts,
        "moves_used": int(res.get("moves_used", 0)),
        "moves_limit": int(cfg.moves_limit),
        "moves_remaining": int(res.get("moves_remaining", 0)),
        "final_score": int(res.get("final_score", 0)),
        "objectives_remaining": obj_left,
        "level_type": cfg.level_type.name,
        "duration_sec": round(float(res.get("duration_sec", 0.0)), 3),
        "classification": classification
    }


class EdgeCaseHunter:
    """Manages the edge-case campaign execution and persistence."""

    def __init__(
        self,
        edge_cases_path: str = EDGE_CASES_FILE,
        progress_path: str = PROGRESS_FILE
    ):
        self.edge_cases_path = edge_cases_path
        self.progress_path = progress_path
        self.edge_case_data: Dict[str, Any] = self._load_edge_cases()
        self.progress_data: Dict[str, Any] = self._load_progress()

    def _load_edge_cases(self) -> Dict[str, Any]:
        if os.path.exists(self.edge_cases_path):
            try:
                with open(self.edge_cases_path, "r") as f:
                    return json.load(f)
            except Exception:
                pass
        return {
            "metadata": {
                "model": "Grandmaster Consensus (AlphaCandy Large 14.4M + DQN Master + Heuristics + Lookahead)",
                "created_at": time.strftime("%Y-%m-%d %H:%M:%S")
            },
            "levels": {},
            "summary": {}
        }

    def _load_progress(self) -> Dict[str, Any]:
        if os.path.exists(self.progress_path):
            try:
                with open(self.progress_path, "r") as f:
                    return json.load(f)
            except Exception:
                pass
        return {"completed_levels": {}, "stats": {"total_won": 0, "total_tested": 0}}

    def save_records(self) -> None:
        levels = self.edge_case_data.get("levels", {})
        total = len(levels)
        won = sum(1 for v in levels.values() if v.get("won"))
        first_try = sum(1 for v in levels.values() if v.get("classification") == "FIRST_TRY_WIN")
        adaptive = sum(1 for v in levels.values() if v.get("classification") == "ADAPTIVE_WIN")
        near_misses = sum(1 for v in levels.values() if v.get("classification") == "NEAR_MISS_EDGE_CASE")
        hard_blocks = sum(1 for v in levels.values() if v.get("classification") == "HARD_BLOCKER_EDGE_CASE")

        self.edge_case_data["summary"] = {
            "total_tested": total,
            "total_won": won,
            "win_rate": round((won / max(1, total)) * 100.0, 2),
            "first_try_wins": first_try,
            "first_try_rate": round((first_try / max(1, total)) * 100.0, 2),
            "adaptive_wins": adaptive,
            "near_miss_edge_cases": near_misses,
            "hard_blocker_edge_cases": hard_blocks,
            "last_updated": time.strftime("%Y-%m-%d %H:%M:%S")
        }

        # Atomic write to edge cases file
        tmp_edge = self.edge_cases_path + ".tmp"
        with open(tmp_edge, "w") as f:
            json.dump(self.edge_case_data, f, indent=2)
        os.replace(tmp_edge, self.edge_cases_path)

        # Update and atomic write to progress file
        prog_levels = self.progress_data.get("completed_levels", {})
        p_won = sum(1 for v in prog_levels.values() if v.get("won"))
        p_tot = len(prog_levels)
        self.progress_data["stats"] = {
            "total_won": p_won,
            "total_tested": p_tot,
            "win_rate": round((p_won / max(1, p_tot)) * 100.0, 2)
        }
        tmp_prog = self.progress_path + ".tmp"
        with open(tmp_prog, "w") as f:
            json.dump(self.progress_data, f, indent=2)
        os.replace(tmp_prog, self.progress_path)

    def run(
        self,
        start_lvl: int = 301,
        end_lvl: int = 1000,
        workers: int = 4,
        max_attempts: int = 2,
        force: bool = False
    ):
        print("=================================================================")
        print("    ALPHACANDY GRANDMASTER EDGE-CASE CAMPAIGN HUNTER             ")
        print(f"    Levels {start_lvl} -> {end_lvl} | Workers: {workers} | Max Attempts: {max_attempts}")
        print("=================================================================")

        candidate_levels = list(range(start_lvl, end_lvl + 1))
        existing = self.edge_case_data.get("levels", {})

        levels_to_run = [
            lvl for lvl in candidate_levels
            if force or str(lvl) not in existing or not existing[str(lvl)].get("won", False)
        ]

        print(f"Levels in requested range: {len(candidate_levels)}")
        print(f"Levels to evaluate/solve: {len(levels_to_run)}\n")

        if not levels_to_run:
            print("All levels in this range have already been won!")
            return

        t0 = time.time()
        completed = 0
        batch_won = 0
        task_args = [(lvl, max_attempts) for lvl in levels_to_run]

        with Pool(processes=workers, initializer=init_worker) as pool:
            for res in pool.imap_unordered(solve_level_worker, task_args, chunksize=1):
                lvl_id = res["level_id"]
                won = res["won"]
                cls_tag = res.get("classification", "UNKNOWN")
                moves = res.get("moves_used", 0)
                lim = res.get("moves_limit", 0)
                score = res.get("final_score", 0)
                t_sec = res.get("duration_sec", 0.0)
                ltype = res.get("level_type", "UNKNOWN")

                # Store in edge_case_data
                self.edge_case_data["levels"][str(lvl_id)] = res

                # Store in progress_data
                self.progress_data["completed_levels"][str(lvl_id)] = {
                    "level_id": lvl_id,
                    "won": won,
                    "attempts": res.get("attempts", 1),
                    "moves_used": moves,
                    "moves_limit": lim,
                    "score": score,
                    "level_type": ltype,
                    "time_sec": t_sec
                }

                completed += 1
                if won:
                    batch_won += 1

                # Visual status tag
                if won:
                    status_str = f"[\033[92mWON - {cls_tag}\033[0m]"
                else:
                    status_str = f"[\033[91mLOST - {cls_tag}\033[0m]"

                print(
                    f"[{completed:4d}/{len(levels_to_run)}] Lvl {lvl_id:5d} ({ltype:11s}) | "
                    f"{status_str:30s} | Moves: {moves:2d}/{lim:2d} | "
                    f"Score: {score:8,d} | {t_sec:5.2f}s"
                )

                if completed % 10 == 0:
                    self.save_records()
                    elapsed = time.time() - t0
                    cur_wr = (batch_won / completed) * 100.0
                    avg_speed = elapsed / completed
                    print(
                        f"--> Milestone [{completed}/{len(levels_to_run)}] | "
                        f"Current Win Rate: {cur_wr:.1f}% | Avg Speed: {avg_speed:.2f}s/lvl\n"
                    )

        self.save_records()
        total_time = time.time() - t0
        final_wr = (batch_won / max(1, completed)) * 100.0

        print("\n=================================================================")
        print("               EDGE-CASE CAMPAIGN BATCH COMPLETE                 ")
        print("=================================================================")
        print(f"Total Evaluated : {completed}")
        print(f"Total Won       : {batch_won}/{completed} ({final_wr:.1f}%)")
        print(f"Total Time      : {total_time:.1f}s ({total_time/max(1, completed):.2f}s/level)")
        print(f"Saved Edge Cases: {self.edge_cases_path}")
        print(f"Saved Progress  : {self.progress_path}")
        print("=================================================================\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Grandmaster Edge-Case Campaign Hunter")
    parser.add_argument("--start", type=int, default=301, help="Start level (default: 301)")
    parser.add_argument("--end", type=int, default=1000, help="End level (default: 1000)")
    parser.add_argument("--workers", type=int, default=4, help="Parallel worker processes (default: 4)")
    parser.add_argument("--max-attempts", type=int, default=2, help="Max retry attempts per level (default: 2)")
    parser.add_argument("--output", type=str, default=None, help="Output JSON path for edge cases")
    parser.add_argument("--force", action="store_true", help="Re-evaluate already won levels")
    args = parser.parse_args()

    out_file = args.output or f"campaign/edge_cases_{args.start}_{args.end}.json"
    hunter = EdgeCaseHunter(edge_cases_path=out_file)
    hunter.run(
        start_lvl=args.start,
        end_lvl=args.end,
        workers=args.workers,
        max_attempts=args.max_attempts,
        force=args.force
    )
