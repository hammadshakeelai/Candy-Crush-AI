"""
Superhuman One-Shot Solver for Candy Crush Saga.
Combines CANDYRL multi-skill ensemble with Expectimax lookahead tree search.
Capable of one-shot beating levels across all game modes (Jelly, Ingredients, Score, Blockers).
"""

from typing import List, Tuple, Dict, Optional, Any
import time
import numpy as np

from core.reverse_engine import (
    ReverseEngineState, ReverseCandyEngine, Direction, LevelType
)
from .candyrl_ensemble import CandyRLEnsemble


class SuperhumanSolver:
    """
    Master One-Shot AI Solver for Candy Crush Saga.
    """

    def __init__(
        self,
        engine: Optional[ReverseCandyEngine] = None,
        search_depth: int = 2,
        top_k_candidates: int = 4
    ):
        self.engine = engine or ReverseCandyEngine()
        self.ensemble = CandyRLEnsemble(self.engine)
        self.search_depth = search_depth
        self.top_k_candidates = top_k_candidates

    def select_best_move(
        self, state: ReverseEngineState
    ) -> Optional[Tuple[int, int, Direction, float]]:
        """
        Selects the optimal move using Expectimax Lookahead with CANDYRL heuristic pruning.
        """
        legal_moves = self.engine.get_legal_moves(state)
        if not legal_moves:
            return None
        if len(legal_moves) == 1:
            r, c, d = legal_moves[0]
            return (r, c, d, 1.0)

        # 1. 1-ply candidate evaluation with CANDYRL ensemble
        scored_candidates = []
        for r, c, direction in legal_moves:
            score = self.ensemble.evaluate_move(state, r, c, direction)
            scored_candidates.append((score, r, c, direction))

        scored_candidates.sort(key=lambda item: item[0], reverse=True)

        # If search depth is 1, return the top candidate immediately
        if self.search_depth <= 1:
            best_score, r, c, direction = scored_candidates[0]
            return (r, c, direction, best_score)

        # 2. Deep Lookahead on Top-K Candidates (Expectimax Search)
        best_overall_score = -float("inf")
        best_overall_move = (scored_candidates[0][1], scored_candidates[0][2], scored_candidates[0][3])

        top_candidates = scored_candidates[:self.top_k_candidates]

        for base_score, r, c, direction in top_candidates:
            # Simulate chance node (sample 2 stochastic refill rollouts)
            total_future_value = 0.0
            samples = 2

            for _ in range(samples):
                sim_state = state.copy()
                self.engine.execute_move(sim_state, r, c, direction)

                if sim_state.has_won():
                    total_future_value += 100000.0
                    continue
                if sim_state.moves_left <= 0:
                    continue

                # 2nd ply: Find best response
                next_legal = self.engine.get_legal_moves(sim_state)
                if next_legal:
                    best_next = max(
                        self.ensemble.evaluate_move(sim_state, nr, nc, nd)
                        for nr, nc, nd in next_legal[:6]
                    )
                    total_future_value += best_next

            avg_future = total_future_value / samples
            total_expected_utility = base_score + (0.75 * avg_future)

            if total_expected_utility > best_overall_score:
                best_overall_score = total_expected_utility
                best_overall_move = (r, c, direction)

        return (best_overall_move[0], best_overall_move[1], best_overall_move[2], best_overall_score)

    def solve_level(
        self, state: ReverseEngineState, max_moves: Optional[int] = None, verbose: bool = False
    ) -> Dict[str, Any]:
        """
        Plays a complete level until win or loss.
        Returns detailed performance telemetry.
        """
        moves_limit = max_moves or state.moves_left
        step_count = 0
        shuffle_count = 0
        t0 = time.time()

        if verbose:
            print(f"=== Starting Level {state.config.level_id} ({state.config.level_type.name}) ===")
            print(f"Moves: {state.moves_left} | Jellies: {np.sum(state.jellies)} | Target Score: {state.config.target_score}")

        while not state.has_won() and not state.has_lost() and step_count < moves_limit:
            best = self.select_best_move(state)
            if best is None:
                shuffle_count += 1
                if shuffle_count > 3:
                    # Board deadlocked with no valid swaps possible
                    break
                # Reshuffle board
                self.engine.initialize_board(state)
                continue

            shuffle_count = 0
            r, c, direction, expected_score = best
            score_gained, stats = self.engine.execute_move(state, r, c, direction)
            step_count += 1

            if verbose:
                print(f"Step {step_count:2d} | Move: ({r}, {c}) {direction.name:5s} | Score +{score_gained:5d} | Cascades: {stats.get('cascades', 0)} | Moves Left: {state.moves_left:2d} | Obj Left: {state.remaining_objectives_count()}")

        duration = time.time() - t0
        won = state.has_won()

        result = {
            "level_id": state.config.level_id,
            "won": won,
            "moves_used": step_count,
            "moves_remaining": state.moves_left,
            "final_score": state.score,
            "duration_sec": duration,
            "steps_per_sec": step_count / max(1e-4, duration)
        }

        if verbose:
            status_str = "WON (ONE-SHOT SUCCESS)!" if won else "FAILED"
            print(f"=== Result: {status_str} in {step_count} moves ({duration:.2f}s, Score: {state.score:,}) ===")

        return result
