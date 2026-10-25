"""
Head-to-Head Comparative Benchmark:
Pure AlphaCandy Large (14.4M Standalone) vs Grandmaster Consensus (Large + Master DQN + Lookahead).
Evaluates solve rates, move efficiencies, score distributions, and move latencies
across official King milestone levels.
"""

import os
import sys
import time
import json
import argparse
from typing import List, Dict, Any

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.king_level_loader import KingLevelLoader
from models.alphacandy_large_agent import AlphaCandyLargeAgent
from models.grandmaster_consensus_agent import GrandmasterConsensusAgent


def run_comparative_benchmark(levels: List[int], max_attempts: int = 2) -> Dict[str, Any]:
    print("=" * 72)
    print("  HEAD-TO-HEAD BENCHMARK: ALPHACANDY LARGE vs GRANDMASTER CONSENSUS   ")
    print("=" * 72)

    loader = KingLevelLoader()
    large_agent = AlphaCandyLargeAgent()
    consensus_agent = GrandmasterConsensusAgent()

    results_large = []
    results_consensus = []

    print(f"\nEvaluating {len(levels)} milestone levels: {levels}\n")

    for lvl in levels:
        cfg = loader.load_level(lvl)
        mode_name = cfg.level_type.name

        # 1. Pure AlphaCandy Large
        t0 = time.time()
        res_l = large_agent.solve_level(cfg, max_attempts=max_attempts, seed=42 + lvl * 13)
        dt_l = time.time() - t0
        results_large.append({
            "level": lvl,
            "mode": mode_name,
            "won": res_l["won"],
            "moves_used": res_l["moves_used"],
            "moves_limit": cfg.moves_limit,
            "score": res_l["final_score"],
            "duration": dt_l
        })

        # 2. Grandmaster Consensus
        t0 = time.time()
        res_c = consensus_agent.solve_level(cfg, max_attempts=max_attempts, seed=42 + lvl * 13)
        dt_c = time.time() - t0
        results_consensus.append({
            "level": lvl,
            "mode": mode_name,
            "won": res_c["won"],
            "moves_used": res_c["moves_used"],
            "moves_limit": cfg.moves_limit,
            "score": res_c["final_score"],
            "duration": dt_c
        })

        status_l = f"WON ({res_l['moves_used']}/{cfg.moves_limit} m)" if res_l["won"] else "LOST"
        status_c = f"WON ({res_c['moves_used']}/{cfg.moves_limit} m)" if res_c["won"] else "LOST"
        print(f"Level {lvl:3d} ({mode_name:11s}) | AlphaCandy Large: {status_l:14s} | Consensus: {status_c:14s}")

    # Aggregations
    total = len(levels)
    won_l = sum(1 for r in results_large if r["won"])
    won_c = sum(1 for r in results_consensus if r["won"])

    avg_moves_l = sum(r["moves_used"] for r in results_large) / total
    avg_moves_c = sum(r["moves_used"] for r in results_consensus) / total

    avg_score_l = sum(r["score"] for r in results_large) / total
    avg_score_c = sum(r["score"] for r in results_consensus) / total

    print("\n" + "=" * 72)
    print("                        COMPARATIVE SUMMARY                            ")
    print("=" * 72)
    print(f"Metric                      | Pure AlphaCandy Large | Grandmaster Consensus")
    print("-" * 72)
    print(f"Win Rate                    | {won_l}/{total} ({won_l/total*100:.1f}%)         | {won_c}/{total} ({won_c/total*100:.1f}%)")
    print(f"Average Moves Used          | {avg_moves_l:.1f} moves           | {avg_moves_c:.1f} moves")
    print(f"Average Final Score         | {avg_score_l:,.0f} pts         | {avg_score_c:,.0f} pts")
    print("=" * 72)

    return {
        "levels": levels,
        "pure_large": results_large,
        "grandmaster_consensus": results_consensus,
        "summary": {
            "pure_large_win_rate": won_l / total,
            "consensus_win_rate": won_c / total,
            "avg_moves_large": avg_moves_l,
            "avg_moves_consensus": avg_moves_c,
            "avg_score_large": avg_score_l,
            "avg_score_consensus": avg_score_c
        }
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Head-to-head model benchmark")
    parser.add_argument("--levels", type=int, nargs="+", default=[1, 5, 10, 20, 23, 25, 50],
                        help="List of milestone level IDs to evaluate")
    parser.add_argument("--attempts", type=int, default=2, help="Max attempts per level")
    args = parser.parse_args()

    run_comparative_benchmark(levels=args.levels, max_attempts=args.attempts)
