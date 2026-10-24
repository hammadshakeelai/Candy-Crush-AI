"""
Benchmark Suite for Superhuman Candy Crush Solver on Official King Levels.
Evaluates win rates, move efficiency, and score maximization across real game levels.
"""

import argparse
import time
import numpy as np

from core.king_level_loader import KingLevelLoader
from core.reverse_engine import ReverseCandyEngine, ReverseEngineState
from models.superhuman_solver import SuperhumanSolver


def run_benchmark(
    start_level: int = 1,
    num_levels: int = 15,
    search_depth: int = 2,
    verbose: bool = False
):
    print("=================================================================")
    print(f"  SUPERHUMAN CANDY CRUSH AI: BENCHMARK ON OFFICIAL KING LEVELS")
    print(f"  Evaluating Levels {start_level} to {start_level + num_levels - 1} (Search Depth: {search_depth})")
    print("=================================================================\n")

    loader = KingLevelLoader()
    engine = ReverseCandyEngine()
    solver = SuperhumanSolver(engine=engine, search_depth=search_depth)

    wins = 0
    total_moves_used = 0
    total_score = 0
    total_time = 0.0

    results = []

    for lvl_idx in range(start_level, start_level + num_levels):
        try:
            cfg = loader.load_level(lvl_idx)
        except KeyError:
            print(f"Level {lvl_idx} not found in config, skipping.")
            continue

        state = ReverseEngineState(cfg)
        engine.initialize_board(state)

        t0 = time.time()
        res = solver.solve_level(state, verbose=verbose)
        duration = time.time() - t0

        won = res["won"]
        if won:
            wins += 1

        total_moves_used += res["moves_used"]
        total_score += res["final_score"]
        total_time += duration

        status = "PASSED [1-SHOT]" if won else "FAILED"
        print(f"Level {lvl_idx:3d} ({cfg.level_type.name:11s}) | Status: {status:15s} | Moves Used: {res['moves_used']:2d}/{cfg.moves_limit:2d} | Score: {res['final_score']:7,d} | Time: {duration:.2f}s")
        results.append(res)

    tested_levels = len(results)
    win_rate = (wins / max(1, tested_levels)) * 100.0

    print("\n=================================================================")
    print("                      BENCHMARK SUMMARY                          ")
    print("=================================================================")
    print(f"Total Levels Tested : {tested_levels}")
    print(f"Levels One-Shot Won : {wins}/{tested_levels} ({win_rate:.1f}%)")
    print(f"Average Moves Used  : {total_moves_used / max(1, tested_levels):.1f}")
    print(f"Average Final Score : {total_score // max(1, tested_levels):,}")
    print(f"Total Benchmark Time: {total_time:.2f}s (Avg {total_time / max(1, tested_levels):.2f}s / level)")
    print("=================================================================\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark Superhuman Solver on King Levels")
    parser.add_argument("--start", type=int, default=1, help="Starting level number")
    parser.add_argument("--levels", type=int, default=10, help="Number of levels to test")
    parser.add_argument("--depth", type=int, default=2, help="Lookahead search depth")
    parser.add_argument("--verbose", action="store_true", help="Print step-by-step moves")
    args = parser.parse_args()

    run_benchmark(start_level=args.start, num_levels=args.levels, search_depth=args.depth, verbose=args.verbose)
