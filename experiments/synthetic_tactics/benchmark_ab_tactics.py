"""
A/B Tactical Curriculum Generalization Benchmark.
Evaluates whether injecting 20% synthetic tactical puzzle demonstrations
into the training buffer improves win rate on stubborn King levels
(Levels 148, 201, 235, 272, 279, 300) compared to pure natural gameplay.
"""

import os
import sys
import time
import json
from typing import Dict, Any, List

sys.path.insert(0, os.getcwd())
from core.king_level_loader import KingLevelLoader
from models.grandmaster_consensus_agent import GrandmasterConsensusAgent

STUBBORN_BENCHMARK_LEVELS = [148, 201, 235, 272, 279, 300]


def run_ab_benchmark(agent: GrandmasterConsensusAgent, max_attempts: int = 5) -> Dict[str, Any]:
    loader = KingLevelLoader()
    results = {}
    won_count = 0

    print("=" * 65)
    print("   A/B TACTICAL BENCHMARK: EVALUATING STUBBORN LEVELS")
    print(f"   Target Levels: {STUBBORN_BENCHMARK_LEVELS} | Max Attempts: {max_attempts}")
    print("=" * 65)

    t0 = time.time()
    for lvl in STUBBORN_BENCHMARK_LEVELS:
        cfg = loader.load_level(lvl)
        t_lvl = time.time()
        res = agent.solve_level(cfg, max_attempts=max_attempts)
        duration = round(time.time() - t_lvl, 2)

        won = bool(res.get("won", False))
        if won:
            won_count += 1
            print(f"🌟 Level {lvl:3d}: WON on attempt {res.get('attempt')} ({duration}s)")
        else:
            print(f"❌ Level {lvl:3d}: FAILED after {max_attempts} attempts ({duration}s)")

        results[lvl] = {
            "won": won,
            "attempt": res.get("attempt"),
            "moves_used": res.get("moves_used"),
            "score": res.get("final_score"),
            "duration": duration
        }

    total_time = round(time.time() - t0, 2)
    win_rate = (won_count / len(STUBBORN_BENCHMARK_LEVELS)) * 100.0

    print("\n" + "=" * 65)
    print(f"   A/B BENCHMARK RESULT: {won_count}/{len(STUBBORN_BENCHMARK_LEVELS)} Won ({win_rate:.1f}%) in {total_time}s")
    print("=" * 65 + "\n")

    return {
        "won_count": won_count,
        "total": len(STUBBORN_BENCHMARK_LEVELS),
        "win_rate": win_rate,
        "duration": total_time,
        "details": results
    }


if __name__ == "__main__":
    agent = GrandmasterConsensusAgent(
        large_transformer_ckpt="checkpoints/alphacandy_large_net_master.pt",
        dqn_master_ckpt="checkpoints/candy_crush_dueling_dqn_master.pt",
        search_depth=2,
        top_k_candidates=5,
        device="cpu"
    )
    run_ab_benchmark(agent, max_attempts=2)
