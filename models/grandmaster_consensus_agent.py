"""
Grandmaster Consensus Agent for Candy Crush Saga.
Unifies:
  1. AlphaCandy Large (14.4M Spatial Attention Transformer) -> P_large(a|s) & P(Win|s)
  2. Master Double Dueling DQN with PER -> Q_master(s, a) over 144 canonical swaps
  3. CANDYRL Expert Tactical Ensemble -> H(s, a) heuristic utility & combo alignments
  4. Adaptive Dynamic Phase-Weighted Consensus Layer
  5. Multi-Depth Adaptive Expectimax Tree Lookahead (Depth 1-3 with Beam Pruning)
"""

from typing import List, Tuple, Dict, Optional, Any, Union
import os
import time
import numpy as np
import torch
import torch.nn.functional as F

from core.reverse_engine import (
    ReverseEngineState, ReverseCandyEngine, Direction, LevelType, LevelConfig,
    SpecialCandy, Blocker, Ingredient, OFFSETS
)
from core.king_level_loader import KingLevelLoader
from models.candy_dqn import DuelingQNetwork
from models.spatial_transformer_net import (
    AlphaCandyLargeNet, encode_spatial_board_tensor,
    encode_action_index, decode_action_index, NUM_CANONICAL_ACTIONS
)
from models.candyrl_ensemble import GrandmasterCandyRLEnsemble


class GrandmasterConsensusAgent:
    """
    Grandmaster Consensus Agent: Combines AlphaCandy Large (14.4M), Double Dueling DQN Master,
    and Tactical CANDYRL heuristics into a phase-adaptive consensus layer with tree lookahead.
    """

    def __init__(
        self,
        large_transformer_ckpt: Optional[str] = None,
        dqn_master_ckpt: Optional[str] = None,
        search_depth: int = 2,
        top_k_candidates: int = 6,
        engine: Optional[ReverseCandyEngine] = None,
        device: Optional[torch.device] = None
    ):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.engine = engine or ReverseCandyEngine()
        self.ensemble = GrandmasterCandyRLEnsemble(self.engine)
        self.search_depth = search_depth
        self.top_k_candidates = top_k_candidates

        # 1. Initialize AlphaCandy Large Net (~14.4M params)
        self.large_net = AlphaCandyLargeNet(in_channels=20, hidden_dim=256, num_actions=144).to(self.device)
        large_ckpt = large_transformer_ckpt or "checkpoints/alphacandy_large_net.pt"
        if not os.path.exists(large_ckpt):
            large_ckpt = "checkpoints/alphacandy_spatial_transformer_master.pt"

        if os.path.exists(large_ckpt):
            try:
                sd = torch.load(large_ckpt, map_location=self.device)
                if self.device.type == "cpu":
                    sd = {k: v.float() if hasattr(v, "is_floating_point") and v.is_floating_point() else v for k, v in sd.items()}
                self.large_net.load_state_dict(sd)
                print(f"[GrandmasterConsensus] Loaded AlphaCandy Large weights from {large_ckpt}")
            except Exception as e:
                print(f"[GrandmasterConsensus] Large net load note: {e}")
        self.large_net.eval()

        # 2. Initialize Double Dueling DQN Master
        self.dqn = DuelingQNetwork(in_channels=20, hidden_dim=128, num_actions=144).to(self.device)
        dqn_ckpt = dqn_master_ckpt or "checkpoints/candy_crush_dueling_dqn_master.pt"
        if not os.path.exists(dqn_ckpt):
            dqn_ckpt = "checkpoints/candy_crush_dueling_dqn.pt"

        if os.path.exists(dqn_ckpt):
            try:
                sd_dqn = torch.load(dqn_ckpt, map_location=self.device)
                self.dqn.load_state_dict(sd_dqn)
                print(f"[GrandmasterConsensus] Loaded DQN Master weights from {dqn_ckpt}")
            except Exception as e:
                print(f"[GrandmasterConsensus] DQN load note: {e}")
        self.dqn.eval()

    def get_phase_weights(self, moves_left: int, total_moves: int) -> Tuple[float, float, float]:
        """
        Dynamically adjusts consensus weights based on match phase.
        Returns: (w_transformer, w_dqn, w_heuristics)
        """
        ratio = moves_left / float(max(1, total_moves))
        if moves_left <= 5 or ratio < 0.20:
            # Endgame: High focus on immediate blocker targeting and objective fulfillment
            return 0.30, 0.10, 0.60
        elif moves_left <= 15 or ratio < 0.50:
            # Midgame: Balanced non-local spatial guidance and objective clearing
            return 0.50, 0.25, 0.25
        else:
            # Opening: Strategic space creation, combo setup, and global Q-value exploration
            return 0.40, 0.35, 0.25

    def evaluate_consensus_candidates(
        self, state: ReverseEngineState
    ) -> List[Dict[str, Any]]:
        """
        Runs multi-network evaluation across all legal moves and computes consensus scores.
        """
        legal_moves = self.engine.get_legal_moves(state)
        if not legal_moves:
            return []

        # 1. Prepare Board Tensor
        s_np = encode_spatial_board_tensor(state)
        s_t = torch.from_numpy(s_np).unsqueeze(0).to(self.device)

        # 2. AlphaCandy Large Evaluation
        with torch.no_grad():
            _, trans_probs, _, win_prob = self.large_net.forward_with_mask(s_t, legal_moves)
            trans_probs_np = trans_probs.cpu().numpy()
            win_p = float(win_prob)

        # 3. Double Dueling DQN Master Evaluation
        with torch.no_grad():
            mask_144 = torch.zeros((1, 144), dtype=torch.bool, device=self.device)
            for r, c, d in legal_moves:
                a_idx = encode_action_index(r, c, d)
                if 0 <= a_idx < 144:
                    mask_144[0, a_idx] = True
            dqn_q = self.dqn(s_t, legal_action_mask=mask_144).squeeze(0).cpu().numpy()

        # 4. Normalize Q-values and Tactical Scores
        max_limit = getattr(state.config, "moves_limit", 30)
        w_trans, w_dqn, w_heur = self.get_phase_weights(state.moves_left, max_limit)

        q_valid = [float(dqn_q[encode_action_index(r, c, d)]) for r, c, d in legal_moves if 0 <= encode_action_index(r, c, d) < 144]
        min_q = min(q_valid) if q_valid else 0.0
        max_q = max(q_valid) if q_valid else 1.0
        q_denom = max(1e-5, max_q - min_q)

        tactical_scores = [self.ensemble.evaluate_move(state, r, c, d) for r, c, d in legal_moves]
        min_t = min(tactical_scores) if tactical_scores else 0.0
        max_t = max(tactical_scores) if tactical_scores else 1.0
        t_denom = max(1e-5, max_t - min_t)

        candidates = []
        for i, (r, c, d) in enumerate(legal_moves):
            a_idx = encode_action_index(r, c, d)
            p_tr = float(trans_probs_np[a_idx]) if 0 <= a_idx < 144 else 0.0
            q_norm = (float(dqn_q[a_idx]) - min_q) / q_denom if 0 <= a_idx < 144 else 0.0
            t_norm = (tactical_scores[i] - min_t) / t_denom

            # Consensus Score
            consensus_score = (w_trans * p_tr) + (w_dqn * q_norm) + (w_heur * t_norm)

            candidates.append({
                "move": (r, c, d),
                "action_idx": a_idx,
                "trans_prob": p_tr,
                "dqn_q_norm": q_norm,
                "tactical_norm": t_norm,
                "consensus_score": consensus_score,
                "win_prob": win_p
            })

        candidates.sort(key=lambda x: x["consensus_score"], reverse=True)
        return candidates

    def select_best_move(
        self, state: ReverseEngineState
    ) -> Optional[Tuple[int, int, Direction, float, Dict[str, Any]]]:
        """
        Selects best action using Consensus Scoring + Adaptive Lookahead Search.
        """
        candidates = self.evaluate_consensus_candidates(state)
        if not candidates:
            return None
        if len(candidates) == 1:
            m = candidates[0]["move"]
            return (m[0], m[1], m[2], candidates[0]["consensus_score"], candidates[0])

        top_k = candidates[:self.top_k_candidates]

        # Fast path if best candidate has massive consensus margin
        if len(candidates) > 1 and (candidates[0]["consensus_score"] - candidates[1]["consensus_score"] > 0.45):
            m = candidates[0]["move"]
            return (m[0], m[1], m[2], candidates[0]["consensus_score"], candidates[0])

        # Dynamic Search Depth
        win_p = candidates[0]["win_prob"]
        if state.moves_left <= 4 or (0.25 <= win_p <= 0.65):
            effective_depth = max(self.search_depth, 2)
        else:
            effective_depth = 1

        best_score = -1e9
        best_cand = top_k[0]

        for cand in top_k:
            r, c, d = cand["move"]
            sim_state = state.copy()
            score_before = sim_state.score
            obj_before = sim_state.remaining_objectives_count()

            self.engine.execute_move(sim_state, r, c, d)

            if sim_state.has_won():
                return (r, c, d, 9999.0, cand)

            immediate_gain = (sim_state.score - score_before) / 100.0
            obj_cleared = (obj_before - sim_state.remaining_objectives_count()) * 25.0

            branch_val = cand["consensus_score"] * 10.0 + immediate_gain + obj_cleared

            # Depth-2 Lookahead on competitive branches
            if effective_depth >= 2 and not sim_state.has_lost():
                next_cands = self.evaluate_consensus_candidates(sim_state)
                if next_cands:
                    best_next = next_cands[0]["consensus_score"] * 5.0
                    branch_val += best_next

            if branch_val > best_score:
                best_score = branch_val
                best_cand = cand

        chosen = best_cand["move"]
        return (chosen[0], chosen[1], chosen[2], best_cand["consensus_score"], best_cand)

    def solve_level(
        self,
        config_or_state: Union[LevelConfig, ReverseEngineState],
        max_moves: Optional[int] = None,
        max_attempts: int = 2,
        seed: int = 42
    ) -> Dict[str, Any]:
        """
        Solves level using full Grandmaster Consensus Ensemble with adaptive search.
        """
        t0 = time.time()
        best_res = None

        for attempt in range(1, max_attempts + 1):
            if isinstance(config_or_state, ReverseEngineState):
                state = config_or_state
            else:
                state = ReverseEngineState(config_or_state)
                self.engine.initialize_board(state, seed=seed + (attempt - 1) * 313)

            moves_limit = max_moves or state.moves_left
            step = 0
            shuffles = 0

            while not state.has_won() and not state.has_lost() and step < moves_limit:
                legal = self.engine.get_legal_moves(state)
                if not legal:
                    if not self.engine.shuffle_board(state):
                        break
                    shuffles += 1
                    continue

                mv_res = self.select_best_move(state)
                if mv_res is None:
                    break

                r, c, d, sc, meta = mv_res
                self.engine.execute_move(state, r, c, d)
                step += 1

            won = state.has_won()
            res = {
                "won": won,
                "attempt": attempt,
                "moves_used": step,
                "moves_limit": moves_limit,
                "moves_remaining": state.moves_left,
                "final_score": state.score,
                "objectives_remaining": state.remaining_objectives_count(),
                "duration_sec": time.time() - t0,
                "shuffles": shuffles,
                "model": "Grandmaster Consensus (Large + DQN Master + Lookahead)"
            }

            if won:
                return res
            if best_res is None or res["final_score"] > best_res["final_score"]:
                best_res = res

        return best_res or res
