"""
Grandmaster Near-Miss Edge-Case Resolver.
Targets all 347 levels across Levels 1 - 2,500 that resulted in NEAR_MISS_EDGE_CASE
(lost with <= 3 remaining objectives/jellies).

Applies:
  - Newly harvested Stage 2 AlphaCandy Large Master weights (14.4M params)
  - Master Double Dueling DQN weights
  - Multi-attempt adaptive solving with diverse seed rotation (up to 4 attempts)
  - Escalated lookahead search
  - Updates campaign/progress.json and outputs campaign/resolved_edge_cases.json
"""

import os
import sys
import json
import time
import argparse
from typing import Dict, Any, List, Optional
from multiprocessing import Pool
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.king_level_loader import KingLevelLoader
from core.procedural_levels import ProceduralKingGenerator
from core.reverse_engine import ReverseCandyEngine
from models.grandmaster_consensus_agent import GrandmasterConsensusAgent

RESOLVED_FILE = "campaign/resolved_edge_cases.json"
PROGRESS_FILE = "campaign/progress.json"

_WORKER_LOADER: Optional[KingLevelLoader] = None
_WORKER_PROCEDURAL_GEN: Optional[ProceduralKingGenerator] = None
_WORKER_AGENT: Optional[GrandmasterConsensusAgent] = None
_WORKER_ENGINE: Optional[ReverseCandyEngine] = None


def init_worker():
    torch.set_num_threads(1)
    global _WORKER_LOADER, _WORKER_PROCEDURAL_GEN, _WORKER_AGENT, _WORKER_ENGINE
    _WORKER_LOADER = KingLevelLoader()
    _WORKER_PROCEDURAL_GEN = ProceduralKingGenerator()
    _WORKER_ENGINE = ReverseCandyEngine()
    _WORKER_AGENT = GrandmasterConsensusAgent(
        large_transformer_ckpt="checkpoints/alphacandy_large_net_master.pt",
        dqn_master_ckpt="checkpoints/candy_crush_dueling_dqn_master.pt",
        search_depth=2,
        top_k_candidates=5,
        engine=_WORKER_ENGINE,
        device="cpu"
    )


def resolve_single_level(args: tuple) -> Dict[str, Any]:
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
            "status": "ERROR"
        }

    res = _WORKER_AGENT.solve_level(cfg, max_attempts=max_attempts)

    won = bool(res.get("won", False))
    attempt = int(res.get("attempt", 1))
    obj_left = int(res.get("objectives_remaining", 0))

    return {
        "level_id": int(level_id),
        "won": won,
        "attempts": attempt,
        "moves_used": int(res.get("moves_used", 0)),
        "moves_limit": int(cfg.moves_limit),
        "moves_remaining": int(res.get("moves_remaining", 0)),
        "final_score": int(res.get("final_score", 0)),
        "objectives_remaining": obj_left,
        "level_type": cfg.level_type.name,
        "duration_sec": round(float(res.get("duration_sec", 0.0)), 2),
        "status": "RESOLVED_WIN" if won else ("STILL_NEAR_MISS" if obj_left <= 3 else "PERSISTENT_BLOCK")
    }


class EdgeCaseResolver:
    def __init__(self, resolved_path: str = RESOLVED_FILE, progress_path: str = PROGRESS_FILE):
        self.resolved_path = resolved_path
        self.progress_path = progress_path
        self.resolved_data = self._load_resolved()
        self.progress_data = self._load_progress()

    def _load_resolved(self) -> Dict[str, Any]:
        if os.path.exists(self.resolved_path):
            try:
                with open(self.resolved_path, "r") as f:
                    return json.load(f)
            except Exception:
                pass
        return {
            "metadata": {
                "description": "Adaptive resolution of near-miss edge cases using Stage 2 Master Checkpoint",
                "updated_at": time.strftime("%Y-%m-%d %H:%M:%S")
            },
            "resolved_levels": {},
            "summary": {}
        }

    def _load_progress(self) -> Dict[str, Any]:
        if os.path.exists(self.progress_path):
            try:
                with open(self.progress_path, "r") as f:
                    return json.load(f)
            except Exception:
                pass
        return {"completed_levels": {}, "stats": {}}

    def collect_target_levels(self) -> List[int]:
        targets = []
        files = [
            "campaign/edge_cases_301_1000.json",
            "campaign/edge_cases_1001_2500.json"
        ]
        for fpath in files:
            if not os.path.exists(fpath):
                continue
            with open(fpath, "r") as f:
                data = json.load(f)
            for lvl_str, item in data.get("levels", {}).items():
                if item.get("classification") == "NEAR_MISS_EDGE_CASE":
                    lvl_id = int(lvl_str)
                    # Check if already won in progress
                    prog_entry = self.progress_data.get("completed_levels", {}).get(str(lvl_id))
                    if prog_entry and prog_entry.get("won"):
                        continue
                    targets.append(lvl_id)
        targets = sorted(list(set(targets)))
        return targets

    def save_state(self):
        resolved_dict = self.resolved_data.get("resolved_levels", {})
        total = len(resolved_dict)
        won = sum(1 for v in resolved_dict.values() if v.get("won"))

        self.resolved_data["summary"] = {
            "total_near_misses_targeted": total,
            "total_converted_to_won": won,
            "conversion_rate": round((won / max(1, total)) * 100.0, 2),
            "updated_at": time.strftime("%Y-%m-%d %H:%M:%S")
        }

        tmp_res = self.resolved_path + ".tmp"
        with open(tmp_res, "w") as f:
            json.dump(self.resolved_data, f, indent=2)
        os.replace(tmp_res, self.resolved_path)

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

    def run(self, workers: int = 6, max_attempts: int = 4):
        target_levels = self.collect_target_levels()
        print("=================================================================")
        print("       GRANDMASTER NEAR-MISS EDGE-CASE RESOLVER                   ")
        print(f"       Target Near-Miss Levels: {len(target_levels)} | Workers: {workers} | Max Attempts: {max_attempts}")
        print("=================================================================\n")

        if not target_levels:
            print("No pending near-miss edge cases found!")
            return

        t0 = time.time()
        completed = 0
        converted = 0
        task_args = [(lvl, max_attempts) for lvl in target_levels]

        with Pool(processes=workers, initializer=init_worker) as pool:
            for res in pool.imap_unordered(resolve_single_level, task_args, chunksize=1):
                lvl_id = res["level_id"]
                won = res["won"]
                status = res["status"]
                moves = res["moves_used"]
                lim = res["moves_limit"]
                score = res["final_score"]
                t_sec = res["duration_sec"]
                ltype = res["level_type"]
                att = res["attempts"]

                self.resolved_data["resolved_levels"][str(lvl_id)] = res

                if won:
                    converted += 1
                    # Update progress.json with the victory
                    self.progress_data["completed_levels"][str(lvl_id)] = {
                        "level_id": lvl_id,
                        "won": True,
                        "attempts": att,
                        "moves_used": moves,
                        "moves_limit": lim,
                        "score": score,
                        "level_type": ltype,
                        "time_sec": t_sec,
                        "resolved_from_near_miss": True
                    }

                completed += 1
                status_str = f"[\033[92m{status}\033[0m]" if won else f"[\033[93m{status}\033[0m]"

                print(
                    f"[{completed:3d}/{len(target_levels)}] Lvl {lvl_id:5d} ({ltype:11s}) | "
                    f"{status_str:28s} | Att: {att} | Moves: {moves:2d}/{lim:2d} | "
                    f"Score: {score:8,d} | {t_sec:5.2f}s"
                )

                if completed % 10 == 0:
                    self.save_state()
                    cur_cr = (converted / completed) * 100.0
                    print(f"--> Progress [{completed}/{len(target_levels)}] | Conversion Rate: {cur_cr:.1f}% ({converted} won)\n")

        self.save_state()
        total_time = time.time() - t0
        final_cr = (converted / max(1, completed)) * 100.0

        print("\n=================================================================")
        print("          NEAR-MISS RESOLUTION BATCH COMPLETE                    ")
        print("=================================================================")
        print(f"Near-Misses Tested : {completed}")
        print(f"Converted to WON   : {converted}/{completed} ({final_cr:.1f}%)")
        print(f"Total Time         : {total_time:.1f}s ({total_time/max(1, completed):.2f}s/level)")
        print(f"New Total Won (1-2500): {self.progress_data['stats']['total_won']} / {self.progress_data['stats']['total_tested']} ({self.progress_data['stats']['win_rate']}%)")
        print("=================================================================\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Grandmaster Near-Miss Resolver")
    parser.add_argument("--workers", type=int, default=6, help="Worker processes")
    parser.add_argument("--max-attempts", type=int, default=4, help="Max retry attempts")
    args = parser.parse_args()

    resolver = EdgeCaseResolver()
    resolver.run(workers=args.workers, max_attempts=args.max_attempts)
