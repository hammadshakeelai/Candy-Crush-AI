#!/usr/bin/env python3
"""
Local Near-Miss Finisher
========================
Runs AlphaCandy-Zero Clairvoyant Oracle locally on the 8 highest-priority near-miss fortresses:
[953, 1660, 1656, 2243, 2245, 2445, 2482, 1951].
Features:
- Direct Striped Laser Snipe Heuristics
- Color Bomb Isolated Alignment Rewards
- Multiprocessing across local CPU cores
- Incremental JSON logging to runs/near_miss_finisher_results.json
"""

import sys
import time
import json
from pathlib import Path
import multiprocessing as mp

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.king_level_loader import KingLevelLoader
from alphacandy_zero.oracle import solve_with_oracle

TARGET_LEVELS = [1660, 2245, 2243, 1656, 2445, 2482, 953, 1951]
BEAM_WIDTH = 12
BRANCH_PER_NODE = 8
MAX_SEEDS = 6
OUT_PATH = Path("runs/near_miss_finisher_results.json")


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
            base_seed=42 + lid * 7,
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
    print(f"[LocalFinisher] Starting evaluation on {len(TARGET_LEVELS)} near-miss fortresses: {TARGET_LEVELS}")
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
                json.dump({"won_count": len(won_levels), "won_levels": won_levels, "results": results}, f, indent=2)

    print(f"\n[LocalFinisher] Completed! Won {len(won_levels)}/{len(TARGET_LEVELS)} levels: {won_levels}")


if __name__ == "__main__":
    main()
