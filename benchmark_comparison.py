"""
Benchmark Comparison: Pure DQN vs Pure Transformer vs Grandmaster Unified Agent.
Evaluates agents on official King milestone levels: [1, 10, 23, 25, 50, 100, 250].
Generates detailed comparative tables and performance metrics.
"""

from typing import List, Dict, Any
import os
import sys
import time
import json
import numpy as np

from core.king_level_loader import KingLevelLoader
from core.reverse_engine import ReverseCandyEngine, ReverseEngineState
from models.grandmaster_unified_agent import (
    GrandmasterUnifiedAgent, PureDQNAgent, PureTransformerAgent
)

MILESTONE_LEVELS = [1, 10, 23, 25, 50, 100, 250]


def run_comparative_benchmark(
    levels: List[int] = MILESTONE_LEVELS,
    output_json: str = "analysis/benchmark_grandmaster_unified.json"
) -> Dict[str, Any]:
    print("===============================================================================")
    print("   GRANDMASTER MULTI-PARADIGM BENCHMARK ON OFFICIAL KING MILESTONES")
    print(f"   Levels to Evaluate: {levels}")
    print("   Paradigms:")
    print("     1. Pure Double Dueling DQN (Trained on King Levels)")
    print("     2. Pure AlphaCandy Transformer (Untrained / Pre-convergence baseline)")
    print("     3. Grandmaster Unified Agent (DQN + GRPO + Transformer + Ensemble + Expectimax)")
    print("===============================================================================\n")

    loader = KingLevelLoader()
    engine = ReverseCandyEngine()

    print("Initializing Agents...")
    t_init = time.time()
    dqn_agent = PureDQNAgent(engine=engine)
    trans_agent = PureTransformerAgent(engine=engine)
    # Enable transformer in unified agent, note parent reminder
    gm_agent = GrandmasterUnifiedAgent(engine=engine)
    print(f"Agents initialized in {time.time() - t_init:.2f}s\n")

    agents = {
        "Pure DQN": dqn_agent,
        "Pure Transformer (Untrained)": trans_agent,
        "Grandmaster Unified": gm_agent
    }

    benchmark_data: Dict[str, List[Dict[str, Any]]] = {
        name: [] for name in agents
    }

    summary: Dict[str, Dict[str, Any]] = {}

    for lvl_id in levels:
        try:
            cfg = loader.load_level(lvl_id)
        except Exception as e:
            print(f"Failed to load Level {lvl_id}: {e}")
            continue

        print(f"--- Evaluating Level {lvl_id:3d} ({cfg.level_type.name:11s} | Moves Limit: {cfg.moves_limit:2d}) ---")

        for name, agent in agents.items():
            t0 = time.time()
            res = agent.solve_level(cfg, seed=42)
            duration = time.time() - t0

            won = res["won"]
            status = "WON" if won else "LOST"
            moves_used = res["moves_used"]
            score = res["final_score"]
            obj_left = res["objectives_remaining"]

            print(f"  [{name:28s}] Status: {status:4s} | Moves: {moves_used:2d}/{cfg.moves_limit:2d} | Score: {score:7,d} | Obj Left: {obj_left:3d} | Time: {duration:5.2f}s")

            benchmark_data[name].append({
                "level_id": int(lvl_id),
                "level_type": str(cfg.level_type.name),
                "moves_limit": int(cfg.moves_limit),
                "won": bool(won),
                "moves_used": int(moves_used),
                "final_score": int(score),
                "objectives_remaining": int(obj_left),
                "duration_sec": round(float(duration), 3)
            })

        print()

    # Calculate aggregate metrics
    for name, results in benchmark_data.items():
        total_tested = len(results)
        wins = sum(1 for r in results if r["won"])
        win_rate = (wins / max(1, total_tested)) * 100.0
        avg_moves = np.mean([r["moves_used"] for r in results])
        avg_score = np.mean([r["final_score"] for r in results])
        avg_time = np.mean([r["duration_sec"] for r in results])

        summary[name] = {
            "total_tested": total_tested,
            "wins": wins,
            "win_rate_pct": round(win_rate, 2),
            "avg_moves_used": round(float(avg_moves), 2),
            "avg_score": round(float(avg_score), 1),
            "avg_time_sec": round(float(avg_time), 3)
        }

    # Print Summary Table
    print("\n" + "=" * 85)
    print("                                BENCHMARK SUMMARY TABLE")
    print("=" * 85)
    header = f"{'Agent / Paradigm':<30} | {'Win Rate':<10} | {'Wins/Total':<10} | {'Avg Moves':<10} | {'Avg Score':<12} | {'Avg Time':<10}"
    print(header)
    print("-" * 85)
    for name, stats in summary.items():
        row = f"{name:<30} | {stats['win_rate_pct']:>8.1f}% | {stats['wins']:>3d}/{stats['total_tested']:<4d} | {stats['avg_moves_used']:>9.1f} | {int(stats['avg_score']):>11,d} | {stats['avg_time_sec']:>8.2f}s"
        print(row)
    print("=" * 85 + "\n")

    payload = {
        "milestone_levels": levels,
        "summary": summary,
        "details": benchmark_data
    }

    os.makedirs(os.path.dirname(os.path.abspath(output_json)), exist_ok=True)
    with open(output_json, "w") as f:
        json.dump(payload, f, indent=2)

    print(f"Benchmark results successfully exported to {output_json}")
    return payload


if __name__ == "__main__":
    run_comparative_benchmark()
