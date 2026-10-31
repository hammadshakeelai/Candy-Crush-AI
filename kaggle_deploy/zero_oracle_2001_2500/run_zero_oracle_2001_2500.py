#!/usr/bin/env python3
"""
AlphaCandy-Zero Clairvoyant Oracle Cloud Runner — Shard 2001-2500
=======================================================================
100% Crash-Proof Pure Python Kaggle CPU Script (4 vCPUs, 0 GPU Quota).
- Clones latest GitHub repository into /tmp/alphacandy_runtime/Candy-Crush-AI
- Discovers King SQLite & JSON configs in /kaggle/input without copying bulky files
- Runs AlphaCandy-Zero Clairvoyant Zobrist Beam Search (`solve_with_oracle`)
- Enforces a strict 18-minute internal wall-clock budget (MAX_WALLTIME_SEC = 1080)
- Writes incremental JSON & Markdown telemetry after every single level
- Wraps all workers and main() in zero-crash defensive guards (sys.exit(0))
"""

import os
import sys
import time
import json
import zipfile
import traceback
import subprocess
import multiprocessing as mp
from pathlib import Path

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

SHARD_LABEL = "2001-2500"
TARGET_LEVELS = [2445, 2482, 2245, 2007, 2035, 2080, 2242, 2273, 2358, 2359, 2384, 2406, 2411, 2437, 2454, 2464, 2471, 2475, 2486]
BEAM_WIDTH = 18
MAX_BRANCH = 10
NUM_SEEDS = 16
MAX_WALLTIME_SEC = 1080.0  # 18 minutes strict safety cutoff

KAGGLE_INPUT = Path("/kaggle/input")
KAGGLE_WORKING = Path("/kaggle/working")
RUNTIME_ROOT = Path("/tmp/alphacandy_runtime")

_WORKER_LOADER = None


def bootstrap_environment():
    """Clone latest GitHub repo into /tmp and wire up /kaggle/input data paths."""
    RUNTIME_ROOT.mkdir(parents=True, exist_ok=True)
    repo_dir = RUNTIME_ROOT / "Candy-Crush-AI"

    if KAGGLE_INPUT.exists():
        if not (repo_dir / "alphacandy_zero").exists():
            try:
                subprocess.run(
                    ["git", "clone", "--depth", "1", "https://github.com/hammadshakeelai/Candy-Crush-AI.git", str(repo_dir)],
                    check=True,
                    timeout=90,
                )
                print(f"[Bootstrap] Cloned latest GitHub repo into {repo_dir}", flush=True)
            except Exception as e:
                print(f"[Bootstrap] Git clone warning: {e}; falling back to input zip", flush=True)
                for p in KAGGLE_INPUT.rglob("candy_crush_pkg.zip"):
                    with zipfile.ZipFile(p, "r") as zf:
                        zf.extractall(repo_dir)
                    break

        for sq in KAGGLE_INPUT.rglob("*.sqlite"):
            os.environ["KING_SQLITE_PATH"] = str(sq)
            break
        for cfg in KAGGLE_INPUT.rglob("configs_v1.208.json"):
            os.environ["KING_CONFIG_PATH"] = str(cfg)
            break
    else:
        repo_dir = Path(__file__).resolve().parents[2]

    if str(repo_dir) not in sys.path:
        sys.path.insert(0, str(repo_dir))
    return str(repo_dir), os.environ.get("KING_SQLITE_PATH", ""), os.environ.get("KING_CONFIG_PATH", "")


def _init_worker(repo_dir_str: str, sqlite_path: str, config_path: str):
    global _WORKER_LOADER
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    if sqlite_path:
        os.environ["KING_SQLITE_PATH"] = sqlite_path
    if config_path:
        os.environ["KING_CONFIG_PATH"] = config_path
    if repo_dir_str not in sys.path:
        sys.path.insert(0, repo_dir_str)
    from core.king_level_loader import KingLevelLoader
    _WORKER_LOADER = KingLevelLoader()


def _solve_single_level_worker(level_id: int) -> dict:
    global _WORKER_LOADER
    t0 = time.time()
    try:
        from alphacandy_zero.oracle import solve_with_oracle
        if _WORKER_LOADER is None:
            from core.king_level_loader import KingLevelLoader
            _WORKER_LOADER = KingLevelLoader()

        cfg = _WORKER_LOADER.load_level(level_id)
        oracle_res = solve_with_oracle(
            cfg=cfg,
            beam_width=BEAM_WIDTH,
            branch_per_node=MAX_BRANCH,
            max_seeds=NUM_SEEDS,
            base_seed=level_id * 1009 + 7,
        )
        diag_dict = oracle_res.get("diagnosis", {})
        return {
            "level_id": int(level_id),
            "level_type": str(diag_dict.get("level_type", cfg.level_type.name)),
            "moves_limit": int(oracle_res.get("moves_limit", cfg.moves_limit)),
            "initial_objectives": int(oracle_res.get("initial_objectives", 0)),
            "isolated_islands": len(diag_dict.get("isolated_objective_cells", [])),
            "won": bool(oracle_res["won"]),
            "winning_seed": oracle_res.get("winning_seed"),
            "attempts_used": int(oracle_res.get("attempts_used", 0)),
            "moves_used": int(oracle_res.get("moves_used", 0)),
            "moves_remaining": int(oracle_res.get("moves_remaining", 0)),
            "final_score": int(oracle_res.get("final_score", 0)),
            "best_remaining_objectives": int(oracle_res.get("objectives_remaining", 999)),
            "elapsed_sec": round(time.time() - t0, 2),
            "error": None,
        }
    except BaseException as e:
        return {
            "level_id": int(level_id),
            "level_type": "UNKNOWN",
            "moves_limit": 0,
            "initial_objectives": 0,
            "isolated_islands": 0,
            "won": False,
            "winning_seed": None,
            "attempts_used": 0,
            "moves_used": 0,
            "moves_remaining": 0,
            "final_score": 0,
            "best_remaining_objectives": 999,
            "elapsed_sec": round(time.time() - t0, 2),
            "error": f"{type(e).__name__}: {e}",
        }


def save_telemetry(out_dir: Path, results: list, start_time: float, finished: bool):
    won_list = [r["level_id"] for r in results if r.get("won")]
    still_unsolved = [r["level_id"] for r in results if not r.get("won")]
    near_misses = [r for r in results if not r.get("won") and r.get("best_remaining_objectives", 999) <= 2]

    summary = {
        "shard": SHARD_LABEL,
        "target_count": len(TARGET_LEVELS),
        "evaluated_count": len(results),
        "newly_won_count": len(won_list),
        "newly_won_levels": sorted(won_list),
        "near_miss_le2_count": len(near_misses),
        "still_unsolved_levels": sorted(still_unsolved),
        "win_rate_on_shard": round(100.0 * len(won_list) / max(1, len(results)), 2),
        "elapsed_sec": round(time.time() - start_time, 2),
        "finished": finished,
        "results": results,
    }
    json_path = out_dir / f"oracle_shard_{SHARD_LABEL.replace('-', '_')}_results.json"
    md_path = out_dir / f"ORACLE_SHARD_{SHARD_LABEL.replace('-', '_')}_REPORT.md"
    json_path.write_text(json.dumps(summary, indent=2))

    lines = [
        f"# AlphaCandy-Zero Clairvoyant Oracle Report — Shard {SHARD_LABEL}",
        "",
        f"- **Evaluated Unsolved Levels**: `{len(results)} / {len(TARGET_LEVELS)}`",
        f"- **Newly Conquered Levels**: **`{len(won_list)}`** (`{summary['win_rate_on_shard']:.2f}%` recovery rate)",
        f"- **Near-Misses (<= 2 Objectives Left)**: `{len(near_misses)}`",
        f"- **Wall-Clock Runtime**: `{summary['elapsed_sec']:.1f}s`",
        f"- **Newly Won Level IDs**: `{sorted(won_list)}`",
        "",
        "## Per-Level Telemetry",
        "",
        "| Level | Type | Moves Limit | Init Obj | Status | Moves Used | Rem Obj | Time (s) |",
        "|---:|:---|---:|---:|:---|---:|---:|---:|",
    ]
    for r in sorted(results, key=lambda x: x["level_id"]):
        status = "WON" if r.get("won") else "UNSOLVED"
        lines.append(
            f"| {r['level_id']} | {r['level_type']} | {r['moves_limit']} | {r['initial_objectives']} | "
            f"**{status}** | {r['moves_used']} | {r['best_remaining_objectives']} | {r['elapsed_sec']:.1f} |"
        )
    md_path.write_text("\n".join(lines) + "\n")


def main():
    start_time = time.time()
    out_dir = KAGGLE_WORKING if KAGGLE_WORKING.exists() else Path(".")
    repo_dir_str, sqlite_path, config_path = bootstrap_environment()

    print("=" * 72, flush=True)
    print(f"ALPHACANDY-ZERO CLAIRVOYANT ORACLE — SHARD {SHARD_LABEL}", flush=True)
    print(f"Target Unsolved Levels: {len(TARGET_LEVELS)} | Beam={BEAM_WIDTH}, Branch={MAX_BRANCH}, Seeds={NUM_SEEDS}", flush=True)
    print("=" * 72, flush=True)

    results = []
    num_workers = min(4, max(1, os.cpu_count() or 2))

    pool = mp.Pool(
        processes=num_workers,
        initializer=_init_worker,
        initargs=(repo_dir_str, sqlite_path, config_path),
    )
    try:
        for res in pool.imap_unordered(_solve_single_level_worker, TARGET_LEVELS):
            results.append(res)
            status_tag = "WON!" if res["won"] else f"rem={res['best_remaining_objectives']}/{res['initial_objectives']}"
            print(
                f"  [Shard {SHARD_LABEL}] ({len(results)}/{len(TARGET_LEVELS)}) "
                f"Level {res['level_id']:4d} ({res['level_type']:11s}): {status_tag:12s} "
                f"in {res['elapsed_sec']:.1f}s",
                flush=True,
            )
            save_telemetry(out_dir, results, start_time, finished=False)

            if time.time() - start_time >= MAX_WALLTIME_SEC:
                print(f"[Safeguard] Reached {MAX_WALLTIME_SEC}s wall-clock cap; saving cleanly!", flush=True)
                break
    finally:
        pool.terminate()
        pool.join()

    save_telemetry(out_dir, results, start_time, finished=True)
    won_cnt = sum(1 for r in results if r["won"])
    print(f"[Complete] Shard {SHARD_LABEL}: {won_cnt}/{len(results)} levels conquered in {time.time() - start_time:.1f}s", flush=True)


if __name__ == "__main__":
    try:
        main()
    except BaseException as exc:
        print(f"[Zero-Crash Guard] Caught top-level exception: {exc}", flush=True)
        traceback.print_exc()
    sys.exit(0)
