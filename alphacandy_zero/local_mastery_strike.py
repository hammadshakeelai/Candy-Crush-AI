#!/usr/bin/env python3
"""
Local Mastery Strike Solver
===========================
Executes multi-seed Clairvoyant Zobrist Beam Search across 4 local CPU cores
targeting the 11 closest near-miss fortresses in Levels 1–2,500.

Features:
- Direct Striped Laser Snipe Heuristic (+550.0)
- Color Bomb Snipe Alignment (+350.0)
- King Candy Frog token '036' homing support
- Multi-seed exploration (up to 20 deterministic seeds per level)
- Incremental JSON logging to runs/local_mastery_strike_results.json
"""

import sys
import time
import json
from pathlib import Path
import multiprocessing as mp

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.king_level_loader import KingLevelLoader
from alphacandy_zero.oracle import solve_with_oracle

TARGET_LEVELS = [975, 953, 1349, 1351, 656, 1800, 1205, 1794, 2245, 606]
BEAM_WIDTH = 26
BRANCH_PER_NODE = 8
MAX_SEEDS = 48
OUT_PATH = Path("runs/local_mastery_strike_results.json")


def _solve_single(lid: int):
    t0 = time.time()
    try:
        loader = KingLevelLoader()
        cfg = loader.load_level(lid)
        res = solve_with_oracle(
            cfg,
            beam_width=BEAM_WIDTH,
            branch_per_node=BRANCH_PER_NODE,
            max_seeds=MAX_SEEDS,
            base_seed=100 + lid * 3,
        )
        elapsed = time.time() - t0
        return {
            "level_id": lid,
            "won": bool(res.get("won", False)),
            "moves_used": res.get("moves_used"),
            "moves_limit": res.get("moves_limit"),
            "remaining_objectives": res.get("objectives_remaining"),
            "initial_objectives": res.get("initial_objectives"),
            "winning_seed": res.get("winning_seed"),
            "elapsed_sec": round(elapsed, 2),
            "error": None,
        }
    except Exception as e:
        return {
            "level_id": lid,
            "won": False,
            "elapsed_sec": round(time.time() - t0, 2),
            "error": str(e),
        }


def main():
    print(f"[MasteryStrike] Launching local multi-core assault on {len(TARGET_LEVELS)} fortresses: {TARGET_LEVELS}")
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    results = []
    won_levels = []

    with mp.Pool(processes=min(4, mp.cpu_count())) as pool:
        for r in pool.imap_unordered(_solve_single, TARGET_LEVELS):
            results.append(r)
            lid = r["level_id"]
            if r["won"]:
                won_levels.append(lid)
                print(f"🎉 [WON] Level {lid} in {r['elapsed_sec']}s! Moves: {r['moves_used']}/{r['moves_limit']}", flush=True)
            else:
                print(f"❌ [LOSS] Level {lid} in {r['elapsed_sec']}s. Rem: {r.get('remaining_objectives')}/{r.get('initial_objectives')}", flush=True)

            with open(OUT_PATH, "w") as f:
                json.dump({
                    "total_targets": len(TARGET_LEVELS),
                    "evaluated_count": len(results),
                    "won_count": len(won_levels),
                    "won_levels": won_levels,
                    "results": results
                }, f, indent=2)

    print(f"\n[MasteryStrike] Assault complete! Conquered {len(won_levels)}/{len(TARGET_LEVELS)} fortresses: {won_levels}")


if __name__ == "__main__":
    main()
