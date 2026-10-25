"""
Superhuman Grandmaster Solver for Candy Crush Saga.
Combines Deep Dual-Head ResNet (Policy P(a|s) + Value V(s)),
CANDYRL expert tactical ensemble, and Adaptive Expectimax Lookahead.
Capable of beating levels across all game modes (Jelly, Ingredients, Score, Blockers).
"""

from typing import List, Tuple, Dict, Optional, Any, Union
import os
import time
import numpy as np
import torch

from core.reverse_engine import (
    ReverseEngineState, ReverseCandyEngine, Direction, LevelType, LevelConfig
)
from .candyrl_ensemble import CandyRLEnsemble
from .network import CandyCrushPolicyValueNet, encode_19ch_tensor


class SuperhumanSolver:
    """
    Master Grandmaster AI Solver for Candy Crush Saga.
    Fuses deep neural pattern recognition with multi-skill tactical heuristics.
    """

    def __init__(
        self,
        engine: Optional[ReverseCandyEngine] = None,
        search_depth: int = 2,
        top_k_candidates: int = 6,
        neural_checkpoint: Optional[str] = "checkpoints/candy_crush_grandmaster_superhuman.pt",
        use_neural: bool = True
    ):
        self.engine = engine or ReverseCandyEngine()
        self.ensemble = CandyRLEnsemble(self.engine)
        self.search_depth = search_depth
        self.top_k_candidates = top_k_candidates
        self.use_neural = use_neural
        self.model: Optional[CandyCrushPolicyValueNet] = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if self.use_neural:
            ckpt_path = neural_checkpoint or "checkpoints/candy_crush_grandmaster_superhuman.pt"
            if not os.path.exists(ckpt_path) and os.path.exists("checkpoints/candy_crush_grandmaster_v2.pt"):
                ckpt_path = "checkpoints/candy_crush_grandmaster_v2.pt"

            if os.path.exists(ckpt_path):
                try:
                    self.model = CandyCrushPolicyValueNet(in_channels=19).to(self.device)
                    self.model.load_state_dict(
                        torch.load(ckpt_path, map_location=self.device, weights_only=True)
                    )
                    self.model.eval()
                except Exception as e:
                    self.model = None

    def select_best_move(
        self, state: ReverseEngineState
    ) -> Optional[Tuple[int, int, Direction, float]]:
        """
        Selects optimal move using Hybrid Neural-Tactical Expectimax search.
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

        top_candidates = scored_candidates[:self.top_k_candidates]

        # 2. Neural Value Network Evaluation & Reranking (Batched Inference)
        if self.model is not None and len(top_candidates) > 0:
            try:
                sim_tensors = []
                for base_score, r, c, direction in top_candidates:
                    sim_s = state.copy()
                    self.engine.execute_move(sim_s, r, c, direction)
                    sim_tensors.append(encode_19ch_tensor(sim_s))

                batch_x = torch.from_numpy(np.array(sim_tensors, dtype=np.float32)).to(self.device)
                with torch.no_grad():
                    _, values = self.model(batch_x)
                    val_scores = values.squeeze(-1).cpu().numpy()

                # Blend heuristic score with neural value prediction V(s')
                reranked = []
                for idx, (base_score, r, c, direction) in enumerate(top_candidates):
                    v_val = float(val_scores[idx])
                    blended = base_score + (120000.0 * v_val)
                    reranked.append((blended, r, c, direction))

                reranked.sort(key=lambda item: item[0], reverse=True)
                top_candidates = reranked
            except Exception:
                pass

        # Adaptive search depth: expand when in endgame or low moves
        effective_depth = self.search_depth
        if state.moves_left <= 8 or state.remaining_objectives_count() <= 6:
            effective_depth = max(self.search_depth, 3)

        if effective_depth <= 1:
            best_score, r, c, direction = top_candidates[0]
            return (r, c, direction, best_score)

        # 3. Deep Lookahead on Top Candidates (Expectimax Search)
        best_overall_score = -float("inf")
        best_overall_move = (top_candidates[0][1], top_candidates[0][2], top_candidates[0][3])

        # Evaluate top 5 pruned moves with stochastic chance sampling
        eval_pool = top_candidates[:5]

        for base_score, r, c, direction in eval_pool:
            total_future_value = 0.0
            samples = 2

            for _ in range(samples):
                sim_state = state.copy()
                self.engine.execute_move(sim_state, r, c, direction)

                if sim_state.has_won():
                    total_future_value += 200000.0
                    continue
                if sim_state.moves_left <= 0:
                    continue

                # 2nd ply: Find best response across legal moves
                next_legal = self.engine.get_legal_moves(sim_state)
                if next_legal:
                    best_next = -float("inf")
                    for nr, nc, nd in next_legal[:16]:
                        u_next = self.ensemble.evaluate_move(sim_state, nr, nc, nd)
                        if effective_depth >= 3 and u_next > 50000.0:
                            # 3rd ply mini-lookahead on promising branches
                            sim_state3 = sim_state.copy()
                            self.engine.execute_move(sim_state3, nr, nc, nd)
                            if sim_state3.has_won():
                                u_next += 150000.0
                        if u_next > best_next:
                            best_next = u_next
                    total_future_value += best_next

            avg_future = total_future_value / samples
            total_expected_utility = base_score + (0.75 * avg_future)

            if total_expected_utility > best_overall_score:
                best_overall_score = total_expected_utility
                best_overall_move = (r, c, direction)

        return (best_overall_move[0], best_overall_move[1], best_overall_move[2], best_overall_score)

    def _play_single_attempt(
        self, state: ReverseEngineState, max_moves: Optional[int] = None, verbose: bool = False
    ) -> Dict[str, Any]:
        """Runs a single play attempt on a state."""
        moves_limit = max_moves or state.moves_left
        step_count = 0
        shuffle_count = 0
        t0 = time.time()

        if verbose:
            print(f"=== Starting Level {state.config.level_id} ({state.config.level_type.name}) ===")
            print(f"Moves: {state.moves_left} | Objectives: {state.remaining_objectives_count()} | Target Score: {state.config.target_score}")

        while not state.has_won() and not state.has_lost() and step_count < moves_limit:
            best = self.select_best_move(state)
            if best is None:
                shuffle_count += 1
                if shuffle_count > 3:
                    break
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

        return {
            "level_id": state.config.level_id,
            "won": won,
            "moves_used": step_count,
            "moves_remaining": state.moves_left,
            "final_score": state.score,
            "duration_sec": duration,
            "steps_per_sec": step_count / max(1e-4, duration)
        }

    def solve_level(
        self,
        level_or_state: Union[ReverseEngineState, LevelConfig],
        max_moves: Optional[int] = None,
        max_attempts: int = 1,
        verbose: bool = False
    ) -> Dict[str, Any]:
        """
        Plays a complete level with adaptive multi-attempt support.
        If initial attempt fails due to unlucky starting board RNG, retries with fresh seeds.
        """
        # If already a state, play single attempt
        if isinstance(level_or_state, ReverseEngineState):
            return self._play_single_attempt(level_or_state, max_moves=max_moves, verbose=verbose)

        config: LevelConfig = level_or_state
        best_res = None

        for attempt in range(1, max_attempts + 1):
            seed = 42 + (attempt - 1) * 1013
            state = ReverseEngineState(config)
            self.engine.initialize_board(state, seed=seed)

            if verbose and attempt > 1:
                print(f"\n--> Attempt {attempt}/{max_attempts} for Level {config.level_id} (Seed: {seed})")

            res = self._play_single_attempt(state, max_moves=max_moves, verbose=verbose)
            res["attempts"] = attempt

            if res["won"]:
                if verbose:
                    print(f"=== Level {config.level_id}: WON on Attempt {attempt}/{max_attempts}! ({res['duration_sec']:.2f}s, Score: {res['final_score']:,}) ===")
                return res

            if best_res is None or res["final_score"] > best_res["final_score"]:
                best_res = res

        return best_res or res
