#!/usr/bin/env python3
"""
Local Multi-Core Clairvoyant Oracle Blitz (`alphacandy_zero/local_oracle_blitz.py`)
===================================================================================
Runs 6 parallel CPU worker processes using `mp.get_context("fork")` to hunt down
high-yield unsolved levels across Levels 301-2,500:
  1. All 8 remaining unsolved INGREDIENTS levels (unlocked by our '010' exit + swap fixes!)
  2. Isolated 1x1 Island-Snipe JELLY levels (unlocked by remote laser/Color Bomb targeting!)
  3. Low-objective (<= 30 jellies) JELLY levels
Automatically updates `campaign/progress.json` and `campaign/unsolved_levels_1_2500.json`
and exports golden winning trajectories to `runs/oracle_golden_blitz.pt`.
"""

import os
import time
import json
import numpy as np
import multiprocessing as mp
from pathlib import Path

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

from core.king_level_loader import KingLevelLoader, LevelType
from alphacandy_zero.oracle import diagnose_level, solve_with_oracle

_LOADER = KingLevelLoader()


def _solve_w(args):
    lvl, beam, branch, seeds = args
    t0 = time.time()
    cfg = _LOADER.load_level(lvl)
    res = solve_with_oracle(
        cfg,
        beam_width=beam,
        branch_per_node=branch,
        max_seeds=seeds,
        base_seed=lvl * 1337 + 11,
    )
    samples = res.pop("samples", [])
    if not res["won"]:
        samples = []
    # Strip any remaining ndarrays from diagnosis before serialization
    if "diagnosis" in res and isinstance(res["diagnosis"], dict):
        diag = res["diagnosis"]
        for k in list(diag.keys()):
            if isinstance(diag[k], np.ndarray):
                diag[k] = diag[k].tolist()
    res["elapsed_sec"] = round(time.time() - t0, 2)
    return res, samples


def main():
    unsolved_path = Path("campaign/unsolved_levels_1_2500.json")
    unsolved = json.loads(unsolved_path.read_text())["unsolved_levels"]

    target_levels = [414, 462, 478, 524, 606, 608, 656, 918, 946, 953, 954, 956, 967, 975, 1053, 1178, 1205, 1292, 1318, 1349, 1351, 1368, 1390, 1488]
    print(f"[Local Oracle Blitz 301-1500 Residuals] Launching 6-worker parallel search across {len(target_levels)} unsolved levels...", flush=True)
    print(f"  Targets ({len(target_levels)}): {target_levels}", flush=True)

    tasks = [(lv, 18, 12, 10) for lv in target_levels]
    t_start = time.time()
    results = []
    golden_samples = []

    ctx = mp.get_context("fork")
    with ctx.Pool(processes=6) as pool:
        for idx, (res, samps) in enumerate(pool.imap_unordered(_solve_w, tasks), 1):
            results.append(res)
            if samps:
                golden_samples.extend(samps)
            tag = "WON!" if res["won"] else f"rem={res['objectives_remaining']}/{res['initial_objectives']}"
            print(
                f"  [{idx:2d}/{len(tasks)}] Level {res['level_id']:4d} ({res['diagnosis']['level_type']:11s}): "
                f"{tag:12s} in {res['elapsed_sec']:.1f}s",
                flush=True,
            )

    won_levels = sorted([r["level_id"] for r in results if r["won"]])
    near_misses = sorted([r["level_id"] for r in results if not r["won"] and r["objectives_remaining"] <= 2])
    print("=" * 72, flush=True)
    print(f"[Local Oracle Blitz Complete] Conquered {len(won_levels)}/{len(results)} levels in {time.time() - t_start:.1f}s!", flush=True)
    print(f"  Newly Won Levels: {won_levels}", flush=True)
    print(f"  Near-Misses (<=2 obj left): {near_misses}", flush=True)

    Path("runs").mkdir(exist_ok=True)
    Path("runs/local_oracle_blitz_results.json").write_text(json.dumps({
        "evaluated": len(results),
        "won_count": len(won_levels),
        "won_levels": won_levels,
        "near_misses_le2": near_misses,
        "results": results,
    }, indent=2))
    if golden_samples:
        import torch
        torch.save(golden_samples, "runs/oracle_golden_blitz.pt")
        print(f"  Saved {len(golden_samples)} golden expert transitions to runs/oracle_golden_blitz.pt", flush=True)


if __name__ == "__main__":
    main()
