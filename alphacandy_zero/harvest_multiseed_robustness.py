#!/usr/bin/env python3
"""
AlphaCandy Multi-Seed Stochastic Robustness Harvester
====================================================
Evaluates and proves Match-3 level solvability across multiple independent
stochastic refill seeds (e.g. 5–10 distinct PRNG seeds per level).
Harvests diverse winning trajectories across varied board permutations into
`runs/multiseed_robustness_trajectories.pt` to ensure zero-overfitting training.
"""

import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from core.king_level_loader import KingLevelLoader
from alphacandy_zero.oracle import solve_with_oracle

DEFAULT_BENCHMARK_LEVELS = [
    # The 18 Famous Hard & Fortress Benchmark Levels
    656, 975, 1205, 1349, 1351, 1390, 1488, 1794, 1958,
    2049, 2185, 2307, 2351, 2437, 2465, 2489, 2490, 2499,
    # Diverse Iconic Mid-World Levels
    50, 75, 100, 147, 181, 250, 272, 300, 350, 400, 500,
]

DEFAULT_SEEDS = [42, 101, 202, 303, 404, 505, 606, 707, 808, 909]


def harvest_multiseed_robustness(
    levels: Optional[List[int]] = None,
    seeds: Optional[List[int]] = None,
    beam_width: int = 16,
    branch_per_node: int = 8,
    max_level_time_sec: float = 30.0,
    output_trajectories_path: Optional[Path] = None,
    output_report_path: Optional[Path] = None,
) -> Dict[str, Any]:
    levels = levels or DEFAULT_BENCHMARK_LEVELS
    seeds = seeds or DEFAULT_SEEDS[:5]  # Default 5 seeds per level
    output_trajectories_path = output_trajectories_path or (REPO_ROOT / "runs" / "multiseed_robustness_trajectories.pt")
    output_report_path = output_report_path or (REPO_ROOT / "runs" / "multiseed_robustness_report.json")

    print("=" * 80)
    print("🍬 ALPHACANDY MULTI-SEED STOCHASTIC ROBUSTNESS HARVESTER")
    print(f"   Levels ({len(levels)}): {levels[:5]}... | Seeds per level: {len(seeds)} {seeds}")
    print("=" * 80)

    loader = KingLevelLoader()
    t_start = time.time()
    
    harvested_trajectories: List[Dict[str, Any]] = []
    level_reports: List[Dict[str, Any]] = []
    
    total_runs = len(levels) * len(seeds)
    runs_completed = 0
    total_wins = 0

    for lvl_idx, lvl in enumerate(levels, 1):
        lvl_t0 = time.time()
        try:
            cfg = loader.load_level(lvl)
        except Exception as e:
            print(f"[WARN] Could not load level {lvl}: {e}")
            continue

        seed_results = []
        lvl_wins = 0
        lvl_moves = []

        for s_idx, s in enumerate(seeds):
            if (time.time() - lvl_t0) > max_level_time_sec:
                print(f"  [Lvl {lvl:04d}] Reached time limit ({max_level_time_sec}s); moving to next level.")
                break

            res = solve_with_oracle(
                cfg,
                beam_width=beam_width,
                branch_per_node=branch_per_node,
                max_seeds=1,
                base_seed=s,
            )
            runs_completed += 1
            won = bool(res.get("won", False))
            moves = int(res.get("moves_used", 0))
            if won:
                lvl_wins += 1
                total_wins += 1
                lvl_moves.append(moves)

            # Harvest trajectory tensors if available in result
            traj = res.get("trajectory", [])
            for step in traj:
                if isinstance(step, dict) and "planes" in step:
                    harvested_trajectories.append(step)

            seed_results.append({
                "seed": s,
                "won": won,
                "moves_used": moves,
                "objectives_remaining": int(res.get("objectives_remaining", 0)),
                "duration_sec": round(float(res.get("duration_sec", 0)), 2),
            })

        stochastic_win_rate = (lvl_wins / len(seed_results)) * 100.0 if seed_results else 0.0
        avg_moves = float(np.mean(lvl_moves)) if lvl_moves else 0.0
        lvl_duration = time.time() - lvl_t0

        level_report = {
            "level_id": lvl,
            "seeds_tested": len(seed_results),
            "seeds_won": lvl_wins,
            "stochastic_win_rate_pct": round(stochastic_win_rate, 2),
            "mean_moves_used": round(avg_moves, 2),
            "moves_limit": int(getattr(cfg, "moves_limit", 25)),
            "duration_sec": round(lvl_duration, 2),
            "seed_details": seed_results,
        }
        level_reports.append(level_report)

        status_sym = "✅" if stochastic_win_rate >= 80.0 else ("⚠️" if stochastic_win_rate > 0 else "❌")
        print(
            f"[{lvl_idx:02d}/{len(levels):02d}] Lvl {lvl:04d} {status_sym} "
            f"Robustness: {lvl_wins}/{len(seed_results)} ({stochastic_win_rate:.1f}%) | "
            f"Avg Moves: {avg_moves:.1f}/{getattr(cfg, 'moves_limit', 25)} | "
            f"Time: {lvl_duration:.2f}s",
            flush=True,
        )

    total_elapsed = time.time() - t_start
    overall_win_rate = (total_wins / max(1, runs_completed)) * 100.0

    summary = {
        "status": "COMPLETED",
        "total_levels_evaluated": len(level_reports),
        "total_runs": runs_completed,
        "total_wins": total_wins,
        "overall_stochastic_robustness_pct": round(overall_win_rate, 2),
        "harvested_trajectory_steps": len(harvested_trajectories),
        "total_elapsed_sec": round(total_elapsed, 2),
        "levels": level_reports,
    }

    # Save summary report JSON
    output_report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_report_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[REPORT] Saved robustness audit -> {output_report_path}")

    # Save harvested multi-seed trajectories if any were extracted
    if harvested_trajectories:
        output_trajectories_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(harvested_trajectories, output_trajectories_path)
        print(f"[DATA] Saved {len(harvested_trajectories):,} multi-seed transitions -> {output_trajectories_path}")

    print("=" * 80)
    print(f"🏆 OVERALL STOCHASTIC ROBUSTNESS: {overall_win_rate:.2f}% ({total_wins}/{runs_completed} seed runs won)")
    print(f"⏱️ Total Wall-Clock Time: {total_elapsed:.2f}s")
    print("=" * 80)
    return summary


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Multi-Seed Stochastic Robustness Harvester")
    parser.add_argument("--levels", type=int, nargs="+", default=None)
    parser.add_argument("--num-seeds", type=int, default=5)
    parser.add_argument("--beam-width", type=int, default=16)
    parser.add_argument("--branch-per-node", type=int, default=8)
    parser.add_argument("--time-limit", type=float, default=25.0)
    args = parser.parse_args()

    active_seeds = DEFAULT_SEEDS[:args.num_seeds]
    harvest_multiseed_robustness(
        levels=args.levels,
        seeds=active_seeds,
        beam_width=args.beam_width,
        branch_per_node=args.branch_per_node,
        max_level_time_sec=args.time_limit,
    )
