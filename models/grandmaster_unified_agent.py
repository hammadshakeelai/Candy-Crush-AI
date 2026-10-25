"""
Grandmaster Unified Multi-Paradigm Agent for Candy Crush Saga.
Integrates:
  1. Double Dueling DQN (models/candy_dqn.py): Q-values over all 144 canonical swaps.
  2. AlphaCandy Spatial Attention Transformer (models/spatial_transformer_net.py):
     Policy distribution P(a|s) and win probability P(Win|s).
  3. GRPO Policy-Value Net (checkpoints/candy_crush_grandmaster_superhuman.pt):
     Group Relative Policy Optimization value and action distribution.
  4. CANDYRL Expert Ensemble & Adaptive Expectimax (models/candyrl_ensemble.py):
     Multi-skill tactical heuristics, macro combo maneuvering, and chance-node search.
"""

from typing import List, Tuple, Dict, Optional, Any, Union
import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from core.reverse_engine import (
    ReverseEngineState, ReverseCandyEngine, Direction, LevelType, LevelConfig,
    SpecialCandy, Blocker, Ingredient, OFFSETS
)
from core.king_level_loader import KingLevelLoader
from models.candy_dqn import DuelingQNetwork
from models.spatial_transformer_net import (
    AlphaCandySpatialNet, encode_spatial_board_tensor,
    encode_action_index, decode_action_index, NUM_CANONICAL_ACTIONS
)
from models.network import CandyCrushPolicyValueNet, encode_19ch_tensor
from models.candyrl_ensemble import GrandmasterCandyRLEnsemble, CandyRLEnsemble


class PureDQNAgent:
    """
    Pure Model-Free Deep Q-Learning Agent.
    Selects moves strictly based on Q-values predicted by the Double Dueling DQN.
    """

    def __init__(
        self,
        checkpoint_path: str = "checkpoints/candy_crush_dueling_dqn.pt",
        engine: Optional[ReverseCandyEngine] = None,
        device: Optional[torch.device] = None
    ):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.engine = engine or ReverseCandyEngine()
        self.net = DuelingQNetwork(in_channels=20, hidden_dim=128, num_actions=144).to(self.device)

        if os.path.exists(checkpoint_path):
            state_dict = torch.load(checkpoint_path, map_location=self.device)
            self.net.load_state_dict(state_dict)
            print(f"[PureDQNAgent] Loaded weights from {checkpoint_path}")
        else:
            print(f"[PureDQNAgent] Warning: Checkpoint not found at {checkpoint_path}, using initialized weights.")
        self.net.eval()

    def select_best_move(self, state: ReverseEngineState) -> Optional[Tuple[int, int, Direction, float]]:
        legal_moves = self.engine.get_legal_moves(state)
        if not legal_moves:
            return None
        if len(legal_moves) == 1:
            r, c, d = legal_moves[0]
            return (r, c, d, 0.0)

        s_np = encode_spatial_board_tensor(state)
        s_t = torch.from_numpy(s_np).unsqueeze(0).to(self.device)

        a_idx, (r, c, d), q_val = self.net.select_action(s_t, legal_moves, epsilon=0.0)
        return (r, c, d, q_val)

    def solve_level(
        self,
        config_or_state: Union[LevelConfig, ReverseEngineState],
        max_moves: Optional[int] = None,
        seed: int = 42
    ) -> Dict[str, Any]:
        if isinstance(config_or_state, ReverseEngineState):
            state = config_or_state
        else:
            state = ReverseEngineState(config_or_state)
            self.engine.initialize_board(state, seed=seed)

        t0 = time.time()
        moves_limit = max_moves or state.moves_left
        step = 0
        shuffles = 0

        while not state.has_won() and not state.has_lost() and step < moves_limit:
            best = self.select_best_move(state)
            if best is None:
                shuffles += 1
                if shuffles > 3:
                    break
                self.engine.initialize_board(state)
                continue
            shuffles = 0
            r, c, d, _ = best
            self.engine.execute_move(state, r, c, d)
            step += 1

        duration = time.time() - t0
        return {
            "level_id": state.config.level_id,
            "won": state.has_won(),
            "moves_used": step,
            "moves_remaining": state.moves_left,
            "final_score": state.score,
            "objectives_remaining": state.remaining_objectives_count(),
            "duration_sec": duration
        }


class PureTransformerAgent:
    """
    Pure AlphaCandy Spatial Attention Transformer Agent.
    Selects moves strictly based on policy distribution P(a|s) output by the Transformer.
    """

    def __init__(
        self,
        checkpoint_path: str = "checkpoints/alphacandy_spatial_transformer.pt",
        engine: Optional[ReverseCandyEngine] = None,
        device: Optional[torch.device] = None
    ):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.engine = engine or ReverseCandyEngine()
        self.net = AlphaCandySpatialNet(in_channels=20, hidden_dim=128, num_actions=144).to(self.device)

        if os.path.exists(checkpoint_path):
            state_dict = torch.load(checkpoint_path, map_location=self.device)
            self.net.load_state_dict(state_dict)
            print(f"[PureTransformerAgent] Loaded weights from {checkpoint_path}")
        else:
            print(f"[PureTransformerAgent] Warning: Checkpoint not found at {checkpoint_path}, using initialized weights.")
        self.net.eval()

    def select_best_move(self, state: ReverseEngineState) -> Optional[Tuple[int, int, Direction, float]]:
        legal_moves = self.engine.get_legal_moves(state)
        if not legal_moves:
            return None
        if len(legal_moves) == 1:
            r, c, d = legal_moves[0]
            return (r, c, d, 1.0)

        s_np = encode_spatial_board_tensor(state)
        s_t = torch.from_numpy(s_np).unsqueeze(0).to(self.device)

        _, probs, val, win_p = self.net.forward_with_mask(s_t, legal_moves)

        # Greedily pick legal move with highest transformer probability
        best_prob = -1.0
        best_move = legal_moves[0]
        for r, c, d in legal_moves:
            a_idx = encode_action_index(r, c, d)
            p = float(probs[a_idx].item())
            if p > best_prob:
                best_prob = p
                best_move = (r, c, d)

        return (best_move[0], best_move[1], best_move[2], best_prob)

    def solve_level(
        self,
        config_or_state: Union[LevelConfig, ReverseEngineState],
        max_moves: Optional[int] = None,
        seed: int = 42
    ) -> Dict[str, Any]:
        if isinstance(config_or_state, ReverseEngineState):
            state = config_or_state
        else:
            state = ReverseEngineState(config_or_state)
            self.engine.initialize_board(state, seed=seed)

        t0 = time.time()
        moves_limit = max_moves or state.moves_left
        step = 0
        shuffles = 0

        while not state.has_won() and not state.has_lost() and step < moves_limit:
            best = self.select_best_move(state)
            if best is None:
                shuffles += 1
                if shuffles > 3:
                    break
                self.engine.initialize_board(state)
                continue
            shuffles = 0
            r, c, d, _ = best
            self.engine.execute_move(state, r, c, d)
            step += 1

        duration = time.time() - t0
        return {
            "level_id": state.config.level_id,
            "won": state.has_won(),
            "moves_used": step,
            "moves_remaining": state.moves_left,
            "final_score": state.score,
            "objectives_remaining": state.remaining_objectives_count(),
            "duration_sec": duration
        }


class GrandmasterUnifiedAgent:
    """
    Unified Grandmaster Multi-Paradigm Superhuman Agent.
    Synthesizes:
      1. Double Dueling DQN: Action-value advantages and Q-values across 144 swaps.
      2. AlphaCandy Spatial Attention Transformer: Graph self-attention policy P(a|s) and P(Win|s).
      3. GRPO Policy-Value Net: High-order reinforcement learned policy & distributional value.
      4. CANDYRL Expert Ensemble: Domain heuristics & tactical combo maneuvering.
      5. Adaptive Expectimax Lookahead: Chance-node simulation and consensus value backpropagation.
    """

    def __init__(
        self,
        dqn_checkpoint: str = "checkpoints/candy_crush_dueling_dqn.pt",
        transformer_checkpoint: str = "checkpoints/alphacandy_spatial_transformer.pt",
        grpo_checkpoint: str = "checkpoints/candy_crush_grandmaster_superhuman.pt",
        engine: Optional[ReverseCandyEngine] = None,
        search_depth: int = 2,
        top_k_candidates: int = 6,
        device: Optional[torch.device] = None
    ):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.engine = engine or ReverseCandyEngine()
        self.ensemble = GrandmasterCandyRLEnsemble(self.engine)
        self.search_depth = search_depth
        self.top_k_candidates = top_k_candidates

        # 1. Double Dueling DQN
        self.dqn = DuelingQNetwork(in_channels=20, hidden_dim=128, num_actions=144).to(self.device)
        if os.path.exists(dqn_checkpoint):
            self.dqn.load_state_dict(torch.load(dqn_checkpoint, map_location=self.device))
            print(f"[GrandmasterUnifiedAgent] Loaded Double Dueling DQN: {dqn_checkpoint}")
        self.dqn.eval()

        # 2. AlphaCandy Spatial Transformer
        self.transformer = AlphaCandySpatialNet(in_channels=20, hidden_dim=128, num_actions=144).to(self.device)
        if os.path.exists(transformer_checkpoint):
            self.transformer.load_state_dict(torch.load(transformer_checkpoint, map_location=self.device))
            print(f"[GrandmasterUnifiedAgent] Loaded AlphaCandy Transformer: {transformer_checkpoint}")
        self.transformer.eval()

        # 3. GRPO Policy-Value Net
        self.grpo = CandyCrushPolicyValueNet(in_channels=19).to(self.device)
        if os.path.exists(grpo_checkpoint):
            self.grpo.load_state_dict(torch.load(grpo_checkpoint, map_location=self.device))
            print(f"[GrandmasterUnifiedAgent] Loaded GRPO Superhuman Net: {grpo_checkpoint}")
        elif os.path.exists("checkpoints/candy_crush_grandmaster_v2.pt"):
            self.grpo.load_state_dict(torch.load("checkpoints/candy_crush_grandmaster_v2.pt", map_location=self.device))
            print("[GrandmasterUnifiedAgent] Loaded GRPO Baseline v2")
        self.grpo.eval()

    def _encode_grpo_action(self, r: int, c: int, direction: Direction, width: int = 9) -> int:
        """Converts move to GRPO 162-action index."""
        if direction == Direction.LEFT:
            r, c = r, c - 1
            direction = Direction.RIGHT
        elif direction == Direction.UP:
            r, c = r - 1, c
            direction = Direction.DOWN

        dir_idx = 0 if direction == Direction.RIGHT else 1
        return (r * width + c) * 2 + dir_idx

    def evaluate_multi_paradigm(
        self, state: ReverseEngineState
    ) -> List[Dict[str, Any]]:
        """
        Runs comprehensive multi-network evaluation for all legal moves in current state.
        Returns a list of structured candidate dictionaries.
        """
        legal_moves = self.engine.get_legal_moves(state)
        if not legal_moves:
            return []

        # 1. Prepare State Tensors
        s_spatial_np = encode_spatial_board_tensor(state)
        s_spatial_t = torch.from_numpy(s_spatial_np).unsqueeze(0).to(self.device)

        s_19ch_np = encode_19ch_tensor(state)
        s_19ch_t = torch.from_numpy(s_19ch_np).unsqueeze(0).to(self.device)

        # 2. AlphaCandy Transformer Evaluation
        with torch.no_grad():
            _, trans_probs, trans_val, win_prob = self.transformer.forward_with_mask(
                s_spatial_t, legal_moves
            )
            trans_probs_np = trans_probs.cpu().numpy()

        # 3. Double Dueling DQN Evaluation
        with torch.no_grad():
            mask_144 = torch.zeros((1, 144), dtype=torch.bool, device=self.device)
            for r, c, d in legal_moves:
                a_idx = encode_action_index(r, c, d)
                if 0 <= a_idx < 144:
                    mask_144[0, a_idx] = True
            dqn_q_all = self.dqn(s_spatial_t, legal_action_mask=mask_144).squeeze(0).cpu().numpy()

        # 4. GRPO Policy-Value Evaluation
        with torch.no_grad():
            grpo_logits, grpo_val = self.grpo(s_19ch_t)
            grpo_logits_1d = grpo_logits.squeeze(0)
            grpo_mask = torch.full_like(grpo_logits_1d, -float("inf"))
            for r, c, d in legal_moves:
                g_idx = self._encode_grpo_action(r, c, d, width=state.width)
                if 0 <= g_idx < 162:
                    grpo_mask[g_idx] = 0.0
            grpo_probs = F.softmax(grpo_logits_1d + grpo_mask, dim=-1).cpu().numpy()
            grpo_val_scalar = float(grpo_val.item())

        # 5. Extract Raw Candidate Scores
        raw_candidates = []
        for r, c, direction in legal_moves:
            a_idx_144 = encode_action_index(r, c, direction)
            g_idx_162 = self._encode_grpo_action(r, c, direction, width=state.width)

            q_val = float(dqn_q_all[a_idx_144]) if 0 <= a_idx_144 < 144 else 0.0
            p_trans = float(trans_probs_np[a_idx_144]) if 0 <= a_idx_144 < 144 else 0.0
            p_grpo = float(grpo_probs[g_idx_162]) if 0 <= g_idx_162 < 162 else 0.0
            tactical_score = self.ensemble.evaluate_move(state, r, c, direction)

            raw_candidates.append({
                "move": (r, c, direction),
                "action_idx_144": a_idx_144,
                "dqn_q": q_val,
                "trans_prob": p_trans,
                "trans_val": trans_val,
                "win_prob": win_prob,
                "grpo_prob": p_grpo,
                "grpo_val": grpo_val_scalar,
                "tactical_score": tactical_score
            })

        # 6. Normalize Scores for Balanced Multi-Paradigm Synthesis
        q_vals = [cand["dqn_q"] for cand in raw_candidates]
        min_q, max_q = min(q_vals), max(q_vals)
        range_q = max(1e-5, max_q - min_q)

        tactical_vals = [cand["tactical_score"] for cand in raw_candidates]
        min_t, max_t = min(tactical_vals), max(tactical_vals)
        range_t = max(1.0, max_t - min_t)

        # Dynamic paradigm weight modulation based on win confidence and game phase
        obj_left = state.remaining_objectives_count()
        moves_left = max(1, state.moves_left)
        urgency = obj_left / float(moves_left)

        # In crunch mode (low win prob or high urgency): bias towards explosive Transformer & Ensemble combos
        if win_prob < 0.4 or urgency > 3.0:
            w_trans = 0.35
            w_ens = 0.35
            w_dqn = 0.15
            w_grpo = 0.15
        elif win_prob > 0.75:  # Endgame closing: bias towards precise DQN & GRPO execution
            w_trans = 0.20
            w_ens = 0.30
            w_dqn = 0.25
            w_grpo = 0.25
        else:  # Standard balanced grandmaster operation
            w_trans = 0.25
            w_ens = 0.35
            w_dqn = 0.20
            w_grpo = 0.20

        for cand in raw_candidates:
            norm_q = (cand["dqn_q"] - min_q) / range_q
            norm_tactical = (cand["tactical_score"] - min_t) / range_t

            # Composite multi-paradigm score
            fused_score = (
                w_ens * norm_tactical +
                w_trans * cand["trans_prob"] +
                w_dqn * norm_q +
                w_grpo * cand["grpo_prob"]
            )
            cand["norm_q"] = norm_q
            cand["norm_tactical"] = norm_tactical
            cand["fused_score"] = fused_score

        raw_candidates.sort(key=lambda item: item["fused_score"], reverse=True)
        return raw_candidates

    def select_best_move(
        self, state: ReverseEngineState, escalate: bool = False
    ) -> Optional[Tuple[int, int, Direction, float]]:
        """
        Executes Unified Grandmaster Move Selection:
          Multi-Paradigm Prior -> Top-K Expectimax Lookahead -> Consensus Action.
        """
        candidates = self.evaluate_multi_paradigm(state)
        if not candidates:
            return None
        if len(candidates) == 1:
            r, c, d = candidates[0]["move"]
            return (r, c, d, candidates[0]["fused_score"])

        top_candidates = candidates[:self.top_k_candidates]

        # Dynamic depth: in endgame or tight constraints, deepen lookahead if escalate is True
        effective_depth = self.search_depth
        if escalate and (state.moves_left <= 8 or state.remaining_objectives_count() <= 6):
            effective_depth = max(self.search_depth, 3)

        if effective_depth <= 1:
            best = top_candidates[0]
            r, c, d = best["move"]
            return (r, c, d, best["fused_score"])

        # Deep Lookahead (Adaptive Expectimax over Top Candidates)
        best_overall_utility = -float("inf")
        best_move = top_candidates[0]["move"]

        # Evaluate top 5 moves with chance-node sampling
        eval_pool = top_candidates[:5]
        samples = 2

        for cand in eval_pool:
            r, c, direction = cand["move"]
            base_tactical = cand["tactical_score"]
            total_future_value = 0.0

            for _ in range(samples):
                sim_state = state.copy()
                score_gained, stats = self.engine.execute_move(sim_state, r, c, direction)

                if sim_state.has_won():
                    total_future_value += 200000.0
                    continue
                if sim_state.moves_left <= 0:
                    continue

                # 2nd ply: evaluate responses
                next_legal = self.engine.get_legal_moves(sim_state)
                if next_legal:
                    best_next = -float("inf")
                    for nr, nc, nd in next_legal[:14]:
                        u_next = self.ensemble.evaluate_move(sim_state, nr, nc, nd)
                        if effective_depth >= 3 and u_next > 60000.0:
                            sim_state3 = sim_state.copy()
                            self.engine.execute_move(sim_state3, nr, nc, nd)
                            if sim_state3.has_won():
                                u_next += 150000.0
                        if u_next > best_next:
                            best_next = u_next
                    total_future_value += best_next

            avg_future = total_future_value / samples
            total_utility = base_tactical + (0.75 * avg_future) + (cand["fused_score"] * 50000.0)

            if total_utility > best_overall_utility:
                best_overall_utility = total_utility
                best_move = (r, c, direction)

        return (best_move[0], best_move[1], best_move[2], best_overall_utility)

    def solve_level(
        self,
        config_or_state: Union[LevelConfig, ReverseEngineState],
        max_moves: Optional[int] = None,
        max_attempts: int = 1,
        seed: int = 42,
        verbose: bool = False
    ) -> Dict[str, Any]:
        """Plays full level with unified grandmaster agent."""
        if isinstance(config_or_state, ReverseEngineState):
            cfg = config_or_state.config
        else:
            cfg = config_or_state

        best_res = None
        for attempt in range(1, max_attempts + 1):
            run_seed = seed + (attempt - 1) * 733
            state = ReverseEngineState(cfg)
            self.engine.initialize_board(state, seed=run_seed)

            t0 = time.time()
            moves_limit = max_moves or cfg.moves_limit
            step = 0
            shuffles = 0

            while not state.has_won() and not state.has_lost() and step < moves_limit:
                best = self.select_best_move(state, escalate=(attempt > 1))
                if best is None:
                    shuffles += 1
                    if shuffles > 3:
                        break
                    self.engine.initialize_board(state)
                    continue
                shuffles = 0
                r, c, d, _ = best
                score_gained, stats = self.engine.execute_move(state, r, c, d)
                step += 1

                if verbose:
                    print(f"Step {step:2d} | Move: ({r}, {c}) {d.name:5s} | Gained: {score_gained:4d} | Cascades: {stats.get('cascades', 0)} | Left: {state.moves_left:2d} | Obj: {state.remaining_objectives_count()}")

            duration = time.time() - t0
            won = state.has_won()

            res = {
                "level_id": cfg.level_id,
                "won": won,
                "moves_used": step,
                "moves_remaining": state.moves_left,
                "final_score": state.score,
                "objectives_remaining": state.remaining_objectives_count(),
                "duration_sec": duration,
                "attempt": attempt
            }

            if won:
                return res
            if best_res is None or res["final_score"] > best_res["final_score"]:
                best_res = res

        return best_res or res
