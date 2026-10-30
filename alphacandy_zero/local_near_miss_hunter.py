#!/usr/bin/env python3
"""
Dedicated Deep-Beam Solver for Near-Miss Levels (<= 3 objectives remaining)
Targets: [953, 1292, 1349, 1205, 1351]
Uses high-precision beam=30, branch=18, seeds=20.
"""

import sys
import time
import json
import multiprocessing as mp
from pathlib import Path
from typing import Dict, Any

WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from core.king_level_loader import KingLevelLoader
from alphacandy_zero.oracle import solve_with_oracle, diagnose_level

TARGET_LEVELS = [953, 1292, 1349, 1205, 1351]
BEAM_WIDTH = 30
BRANCH_PER_NODE = 18
MAX_SEEDS = 20
NUM_WORKERS = min(5, mp.cpu_count())

_LOADER = None

def _init_worker():
    global _LOADER
    _LOADER = KingLevelLoader()
    _LOADER._ensure_loaded()

def _solve_level(lid: int) -> Dict[str, Any]:
    global _LOADER
    if _LOADER is None:
        _LOADER = KingLevelLoader()
        _LOADER._ensure_loaded()
    
    t0 = time.time()
    cfg = _LOADER.load_level(lid)
    diag = diagnose_level(cfg)
    
    res = solve_with_oracle(
        cfg,
        beam_width=BEAM_WIDTH,
        branch_per_node=BRANCH_PER_NODE,
        max_seeds=MAX_SEEDS,
        base_seed=5000 + lid * 13,
    )
    
    elapsed = time.time() - t0
    won = bool(res.get("won", False))
    rem = int(res.get("objectives_remaining", -1))
    score = int(res.get("final_score", 0))
    moves_left = int(res.get("moves_remaining", 0))
    
    status_str = "WON!" if won else f"rem={rem}/{diag.initial_objectives}"
    print(f"  [Level {lid:4d}]: {status_str:<15} score={score:<7d} in {elapsed:.1f}s", flush=True)
    
    return {
        "level_id": lid,
        "won": won,
        "score": score,
        "objectives_remaining": rem,
        "initial_objectives": diag.initial_objectives,
        "moves_remaining": moves_left,
        "elapsed_sec": round(elapsed, 2),
    }

def main():
    print(f"[Near-Miss Deep Hunter] Launching {NUM_WORKERS} workers on {TARGET_LEVELS}...", flush=True)
    print(f"Config: Beam={BEAM_WIDTH}, Branch={BRANCH_PER_NODE}, Seeds={MAX_SEEDS}\n", flush=True)
    
    t_start = time.time()
    ctx = mp.get_context("fork")
    with ctx.Pool(processes=NUM_WORKERS, initializer=_init_worker) as pool:
        results = pool.map(_solve_level, TARGET_LEVELS)
    
    total_time = time.time() - t_start
    won_levels = [r["level_id"] for r in results if r["won"]]
    print(f"\n[Complete in {total_time:.1f}s] Conquered {len(won_levels)}/{len(TARGET_LEVELS)} levels: {won_levels}", flush=True)
    
    out_path = Path("runs/near_miss_hunter_results.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"won_levels": won_levels, "results": results}, f, indent=2)
    print(f"Saved results to {out_path}", flush=True)

if __name__ == "__main__":
    main()
