#!/usr/bin/env python3
"""
AlphaCandy Frontier Campaign Invader (Levels 10,296 to 23,355)
=============================================================
High-throughput multi-core solver targeting the remaining 13,060 frontier levels
in `data/candycrush_master_levels.sqlite`.
Solves procedural King episodes, proves solvability, and harvests winning trajectories.
"""

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from core.king_level_loader import KingLevelLoader
from alphacandy_zero.oracle import solve_with_oracle


def _solve_single_frontier_level(
    level_id: int,
    beam_width: int = 12,
    branch_per_node: int = 6,
    max_seeds: int = 4,
    base_seed: int = 42,
) -> Dict[str, Any]:
    """Worker task executed in sub-process pool."""
    loader = KingLevelLoader()
    t0 = time.time()
    try:
        cfg = loader.load_level(level_id)
        res = solve_with_oracle(
            cfg,
            beam_width=beam_width,
            branch_per_node=branch_per_node,
            max_seeds=max_seeds,
            base_seed=base_seed,
        )
        won = bool(res.get("won", False))
        moves = int(res.get("moves_used", 0))
        rem = int(res.get("objectives_remaining", 0))
        score = int(res.get("final_score", 0))
        duration = time.time() - t0
        return {
            "level_id": level_id,
            "won": won,
            "moves_used": moves,
            "moves_limit": int(getattr(cfg, "moves_limit", 25)),
            "objectives_remaining": rem,
            "final_score": score,
            "duration_sec": round(duration, 3),
            "error": None,
        }
    except Exception as exc:
        return {
            "level_id": level_id,
            "won": False,
            "moves_used": 0,
            "moves_limit": 0,
            "objectives_remaining": 999,
            "final_score": 0,
            "duration_sec": round(time.time() - t0, 3),
            "error": str(exc),
        }


def run_frontier_campaign(
    start_level: int = 10296,
    end_level: int = 12500,
    workers: int = 4,
    beam_width: int = 12,
    branch_per_node: int = 6,
    max_seeds: int = 4,
    output_json: Optional[Path] = None,
) -> Dict[str, Any]:
    output_json = output_json or (REPO_ROOT / "campaign" / f"frontier_{start_level}_{end_level}_results.json")
    output_json.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print(f"🚀 ALPHACANDY FRONTIER INVASION: Levels {start_level:,} to {end_level:,}")
    print(f"   Workers: {workers} | Beam Width: {beam_width} | Shard Range: {end_level - start_level + 1} levels")
    print("=" * 80)

    # Load existing progress if resumed
    results_map: Dict[int, Dict[str, Any]] = {}
    if output_json.exists():
        try:
            with open(output_json, "r", encoding="utf-8") as f:
                prev = json.load(f)
                if isinstance(prev, dict) and "levels" in prev:
                    results_map = {int(k): v for k, v in prev["levels"].items()}
                    print(f"[RESUME] Loaded {len(results_map)} previously solved frontier levels.")
        except Exception as e:
            print(f"[WARN] Could not parse existing results: {e}")

    levels_to_run = [lvl for lvl in range(start_level, end_level + 1) if lvl not in results_map]
    print(f"[TARGET] {len(levels_to_run)} levels remaining in shard {start_level}-{end_level}.")

    t_start = time.time()
    solved_in_run = 0
    won_in_run = 0

    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(
                _solve_single_frontier_level,
                lvl,
                beam_width=beam_width,
                branch_per_node=branch_per_node,
                max_seeds=max_seeds,
            ): lvl
            for lvl in levels_to_run
        }

        for fut in as_completed(futures):
            res = fut.result()
            lvl = res["level_id"]
            results_map[lvl] = res
            solved_in_run += 1
            if res["won"]:
                won_in_run += 1

            status_tag = "WON" if res["won"] else "FAIL"
            if solved_in_run % 10 == 0 or not res["won"] or solved_in_run == len(levels_to_run):
                rate = (won_in_run / solved_in_run) * 100.0
                elapsed = time.time() - t_start
                lps = solved_in_run / max(0.1, elapsed)
                print(
                    f"[{solved_in_run:04d}/{len(levels_to_run):04d}] Lvl {lvl:05d}: {status_tag} | "
                    f"Moves: {res['moves_used']}/{res['moves_limit']} | "
                    f"WinRate: {rate:.1f}% | Rate: {lps:.1f} lvl/s",
                    flush=True,
                )

            # Checkpoint results every 25 levels
            if solved_in_run % 25 == 0:
                payload = {
                    "shard_start": start_level,
                    "shard_end": end_level,
                    "total_solved": len(results_map),
                    "total_won": sum(1 for v in results_map.values() if v.get("won")),
                    "levels": results_map,
                }
                with open(output_json, "w", encoding="utf-8") as f:
                    json.dump(payload, f, indent=2)

    # Final save
    total_won = sum(1 for v in results_map.values() if v.get("won"))
    final_payload = {
        "status": "COMPLETED",
        "shard_start": start_level,
        "shard_end": end_level,
        "total_solved": len(results_map),
        "total_won": total_won,
        "win_rate_pct": round(100.0 * total_won / max(1, len(results_map)), 2),
        "total_elapsed_sec": round(time.time() - t_start, 2),
        "levels": results_map,
    }
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(final_payload, f, indent=2)

    print("=" * 80)
    print(f"🎉 FRONTIER SHARD FINISHED: {total_won}/{len(results_map)} ({final_payload['win_rate_pct']}%)")
    print(f"   Saved -> {output_json}")
    print("=" * 80)
    return final_payload


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AlphaCandy Frontier Campaign Invader")
    parser.add_argument("--start", type=int, default=10296)
    parser.add_argument("--end", type=int, default=10350)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--beam-width", type=int, default=8)
    parser.add_argument("--branch-per-node", type=int, default=6)
    parser.add_argument("--max-seeds", type=int, default=2)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    out_p = Path(args.output) if args.output else None
    run_frontier_campaign(
        start_level=args.start,
        end_level=args.end,
        workers=args.workers,
        beam_width=args.beam_width,
        branch_per_node=args.branch_per_node,
        max_seeds=args.max_seeds,
        output_json=out_p,
    )
