#!/usr/bin/env python3
"""
Test AlphaCandyTitanAgent (79.30M Flagship Model) on Hard King Levels.
Evaluates the newly trained GPU checkpoint (`checkpoints/alphacandy_titan_master.pt`)
across known hard benchmark levels and remaining fortress levels.
"""

import os
import sys
import time
import json
from pathlib import Path
from typing import List, Dict, Any

from core.king_level_loader import KingLevelLoader
from models.alphacandy_titan_agent import AlphaCandyTitanAgent

HARD_BENCHMARK_LEVELS = [65, 125, 147, 181, 235, 272, 279, 300, 350, 400]
FORTRESS_LEVELS = [606, 656, 953, 975, 1205, 1349, 1351]
ALL_TEST_LEVELS = HARD_BENCHMARK_LEVELS + FORTRESS_LEVELS


def main():
    print("=" * 78)
    print("   ALPHACANDY-TITAN (79.30M) HARD LEVEL BENCHMARK ASSAULT")
    print(f"   Checkpoint: checkpoints/alphacandy_titan_master.pt")
    print(f"   Total Levels: {len(ALL_TEST_LEVELS)} ({len(HARD_BENCHMARK_LEVELS)} hard + {len(FORTRESS_LEVELS)} fortress)")
    print("=" * 78)

    loader = KingLevelLoader()
    agent = AlphaCandyTitanAgent(checkpoint_path="checkpoints/alphacandy_titan_master.pt")

    results: List[Dict[str, Any]] = []
    won_count = 0
    t_start = time.time()

    for lvl in ALL_TEST_LEVELS:
        cfg = loader.load_level(lvl)
        category = "FORTRESS" if lvl in FORTRESS_LEVELS else "HARD_BENCHMARK"
        print(f"\n---> Testing Level {lvl:04d} [{category}] (Moves Limit: {cfg.moves_limit})...", flush=True)

        res = agent.solve_level(cfg, max_attempts=5, seed=100 + lvl * 17)
        won = bool(res.get("won", False))
        if won:
            won_count += 1
            status_str = f"🎉 [WON in attempt {res.get('attempt')}]"
        else:
            status_str = f"❌ [LOSS]"

        moves_used = res.get("moves_used")
        moves_limit = res.get("moves_limit")
        rem_obj = res.get("objectives_remaining")
        conf = res.get("mean_epistemic_confidence", 0.0)
        regimes = res.get("regime_counts", {})
        dur = res.get("duration_sec", 0.0)

        print(
            f"   {status_str} Level {lvl:04d} | Moves: {moves_used}/{moves_limit} | "
            f"Rem: {rem_obj} | Conf: {conf:.3f} | Dur: {dur:.2f}s | Regimes: {regimes}",
            flush=True,
        )

        results.append({
            "level_id": lvl,
            "category": category,
            "won": won,
            "attempt": res.get("attempt"),
            "moves_used": moves_used,
            "moves_limit": moves_limit,
            "objectives_remaining": rem_obj,
            "final_score": res.get("final_score"),
            "mean_epistemic_confidence": conf,
            "duration_sec": dur,
            "regime_counts": regimes,
        })

    elapsed = time.time() - t_start
    win_rate = 100.0 * won_count / len(ALL_TEST_LEVELS)

    print("\n" + "=" * 78)
    print(f"   HARD LEVEL BENCHMARK SUMMARY")
    print(f"   Levels Won: {won_count} / {len(ALL_TEST_LEVELS)} ({win_rate:.1f}%)")
    print(f"   Total Time: {elapsed:.1f}s ({elapsed / len(ALL_TEST_LEVELS):.2f}s per level)")
    print("=" * 78)

    out_file = Path("runs/titan_hard_level_evaluation.json")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump({
            "model": "AlphaCandyTitanNet (79.30M)",
            "checkpoint": "checkpoints/alphacandy_titan_master.pt",
            "total_levels": len(ALL_TEST_LEVELS),
            "won_count": won_count,
            "win_rate_pct": win_rate,
            "elapsed_sec": round(elapsed, 2),
            "results": results,
        }, f, indent=2)
    print(f"[REPORT] Saved detailed results to {out_file}")


if __name__ == "__main__":
    main()
