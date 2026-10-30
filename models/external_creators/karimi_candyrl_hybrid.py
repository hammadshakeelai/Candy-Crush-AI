"""
Reproduction of Karimi et al. (ICMLA 2020 / IEEE ToG 2021):
"CANDYRL: A Hybrid Reinforcement Learning Model for Gameplay" &
"BaziGooshi: A Hybrid Model of Reinforcement Learning for Generalization in Gameplay"
Authors: Sara Karimi, Sahar Asadi, Francesco Lorenzo, Amir H. Payberah (KTH & King Digital Entertainment).

Core Architecture (`KarimiCandyRL2020`):
  - Decomposes Match-3 gameplay into 4 modular intrinsic-skill Q-estimators:
      1. Q_combo(s, a): Special candy creation & detonation skill
      2. Q_blocker(s, a): Blocker/frosting/chocolate demolition skill
      3. Q_objective(s, a): Direct jelly / ingredient clearing skill
      4. Q_cascade(s, a): Lower-board cascade & board circulation skill
  - Combines the 4 skill estimators via:
      * Average Bagging (AB): Dynamic weighted combination of normalized skill Q-values
      * High-Speed Hierarchy (HSH): Priority gating when critical objectives or combos are available
  - Note: Faithful to the 2020 paper, this model evaluates immediate 1-step intrinsic skills
    and neural sub-heads WITHOUT non-Euclidean graph attention, detached-island artillery routing,
    or multi-depth expectimax lookahead (which are later innovations in AlphaCandy).
"""

from typing import List, Tuple, Dict, Optional, Any, Union
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from core.reverse_engine import (
    ReverseEngineState, ReverseCandyEngine, Direction, LevelConfig, LevelType,
    SpecialCandy, Blocker, Ingredient, OFFSETS
)
from models.spatial_transformer_net import (
    encode_spatial_board_tensor, encode_action_index, decode_action_index, NUM_CANONICAL_ACTIONS
)


class SkillSubNetwork(nn.Module):
    """Lightweight 3-layer Conv Q-head for a single intrinsic skill in CANDYRL."""

    def __init__(self, in_channels: int = 20, hidden_dim: int = 32, num_actions: int = NUM_CANONICAL_ACTIONS):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(hidden_dim * 9 * 9, num_actions)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class KarimiCandyRL2020(nn.Module):
    """
    Reproduction of Karimi et al. (2020) CANDYRL Hybrid RL Agent.
    Combines 4 specialized intrinsic-skill neural heads + 1-ply intrinsic reward evaluators
    using Average Bagging (AB) and High-Speed Hierarchy (HSH).
    """

    def __init__(
        self,
        in_channels: int = 20,
        skill_hidden_dim: int = 32,
        num_actions: int = NUM_CANONICAL_ACTIONS,
        mode: str = "HSH_AB",  # "AB" (Average Bagging) or "HSH_AB" (High-Speed Hierarchy + AB)
        engine: Optional[ReverseCandyEngine] = None
    ):
        super().__init__()
        self.in_channels = in_channels
        self.num_actions = num_actions
        self.mode = mode
        self.engine = engine or ReverseCandyEngine()

        # 4 Pre-trained / Specialized Intrinsic Skill Networks
        self.skill_combo_net = SkillSubNetwork(in_channels, skill_hidden_dim, num_actions)
        self.skill_blocker_net = SkillSubNetwork(in_channels, skill_hidden_dim, num_actions)
        self.skill_objective_net = SkillSubNetwork(in_channels, skill_hidden_dim, num_actions)
        self.skill_cascade_net = SkillSubNetwork(in_channels, skill_hidden_dim, num_actions)

        # Meta-controller network predicting dynamic skill weights w(s) in Simplex^4
        self.meta_controller = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((3, 3)),
            nn.Flatten(),
            nn.Linear(16 * 9, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 4),
            nn.Softmax(dim=-1)
        )

    def compute_intrinsic_skill_vectors(
        self, state: ReverseEngineState, legal_moves: List[Tuple[int, int, Direction]]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Computes the 4 intrinsic reward vectors over all 144 actions as defined in Karimi et al. (2020):
          r_combo, r_blocker, r_objective, r_cascade
        """
        r_combo = np.full(self.num_actions, -1e9, dtype=np.float32)
        r_blocker = np.full(self.num_actions, -1e9, dtype=np.float32)
        r_objective = np.full(self.num_actions, -1e9, dtype=np.float32)
        r_cascade = np.full(self.num_actions, -1e9, dtype=np.float32)

        for r, c, d in legal_moves:
            a_idx = encode_action_index(r, c, d)
            dr, dc = OFFSETS[d]
            nr, nc = r + dr, c + dc

            sim_state = state.copy()
            score_gained, stats = self.engine.execute_move(sim_state, r, c, d)
            if not stats["success"]:
                continue

            # 1. Special Candy Intrinsic Skill (Karimi Eq. 1)
            s1, s2 = state.specials[r, c], state.specials[nr, nc]
            used_special = 1.0 if (s1 != SpecialCandy.NONE or s2 != SpecialCandy.NONE) else 0.0
            new_bombs = max(0, int(np.sum(sim_state.specials == SpecialCandy.COLOR_BOMB)) - int(np.sum(state.specials == SpecialCandy.COLOR_BOMB)))
            new_wrapped = max(0, int(np.sum(sim_state.specials == SpecialCandy.WRAPPED)) - int(np.sum(state.specials == SpecialCandy.WRAPPED)))
            new_striped = max(0, int(np.sum((sim_state.specials == SpecialCandy.STRIPED_HORIZONTAL) | (sim_state.specials == SpecialCandy.STRIPED_VERTICAL))) -
                              int(np.sum((state.specials == SpecialCandy.STRIPED_HORIZONTAL) | (state.specials == SpecialCandy.STRIPED_VERTICAL))))
            r_combo[a_idx] = (2.5 * new_bombs) + (1.2 * new_wrapped) + (0.6 * new_striped) + (1.5 * used_special)

            # 2. Blocker Demolition Intrinsic Skill
            blockers_cleared = max(0, int(np.sum(state.blockers != Blocker.NONE)) - int(np.sum(sim_state.blockers != Blocker.NONE)))
            r_blocker[a_idx] = float(blockers_cleared) * 1.0

            # 3. Objective (Jelly / Ingredient / Score) Intrinsic Skill
            jellies_cleared = max(0, int(np.sum(state.jellies)) - int(np.sum(sim_state.jellies)))
            ing_cleared = max(0, sim_state.ingredients_collected - state.ingredients_collected)
            r_objective[a_idx] = (jellies_cleared * 1.5) + (ing_cleared * 3.0) + (score_gained / 2000.0)

            # 4. Bottom Cascade Intrinsic Skill
            cascades = stats.get("cascades", 0)
            row_norm = max(r, nr) / float(max(1, state.height - 1))
            r_cascade[a_idx] = (row_norm * 0.8) + (min(cascades, 6) * 0.35)

        return r_combo, r_blocker, r_objective, r_cascade

    def forward(
        self,
        state_or_tensor: Union[ReverseEngineState, torch.Tensor, np.ndarray],
        legal_action_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Computes hybrid (B, 144) Q-scores combining the 4 skill estimators via Average Bagging & HSH.
        """
        device = next(self.parameters()).device
        single_state: Optional[ReverseEngineState] = None

        if isinstance(state_or_tensor, ReverseEngineState):
            single_state = state_or_tensor
            arr = encode_spatial_board_tensor(state_or_tensor)
            x = torch.from_numpy(arr).unsqueeze(0).to(device=device, dtype=torch.float32)
        elif isinstance(state_or_tensor, np.ndarray):
            x = torch.from_numpy(state_or_tensor)
            if x.dim() == 3:
                x = x.unsqueeze(0)
            x = x.to(device=device, dtype=torch.float32)
        else:
            x = state_or_tensor
            if x.dim() == 3:
                x = x.unsqueeze(0)
            x = x.to(device=device, dtype=torch.float32)

        q1 = self.skill_combo_net(x)
        q2 = self.skill_blocker_net(x)
        q3 = self.skill_objective_net(x)
        q4 = self.skill_cascade_net(x)
        weights = self.meta_controller(x)  # (B, 4)

        q_bagged = (
            weights[:, 0:1] * torch.tanh(q1)
            + weights[:, 1:2] * torch.tanh(q2)
            + weights[:, 2:3] * torch.tanh(q3)
            + weights[:, 3:4] * torch.tanh(q4)
        ) * 0.15

        if single_state is not None:
            legal_moves = self.engine.get_legal_moves(single_state)
            if legal_moves:
                rc, rb, ro, rcas = self.compute_intrinsic_skill_vectors(single_state, legal_moves)
                # High-Speed Hierarchy (HSH) + Average Bagging (AB) rules from Karimi et al. (2020)
                num_blockers = int(np.sum(single_state.blockers != Blocker.NONE))
                urgency = single_state.remaining_objectives_count() / float(max(1, single_state.moves_left))

                if self.mode == "HSH_AB":
                    w_obj = 0.45 if urgency > 1.2 else 0.30
                    w_blk = 0.30 if num_blockers > 8 else 0.15
                    w_cmb = 0.25
                    w_cas = max(0.05, 1.0 - (w_obj + w_blk + w_cmb))
                else:
                    w_cmb, w_blk, w_obj, w_cas = 0.25, 0.25, 0.25, 0.25

                hybrid_scores = np.full(self.num_actions, -1e9, dtype=np.float32)
                for r, c, d in legal_moves:
                    idx = encode_action_index(r, c, d)
                    hybrid_scores[idx] = (
                        w_cmb * rc[idx]
                        + w_blk * rb[idx]
                        + w_obj * ro[idx]
                        + w_cas * rcas[idx]
                    )
                hybrid_t = torch.from_numpy(hybrid_scores).unsqueeze(0).to(device)
                q_bagged = q_bagged + hybrid_t

        if legal_action_mask is not None:
            q_bagged = q_bagged.masked_fill(~legal_action_mask.to(device), -1e9)

        return q_bagged

    def select_best_move(self, state: ReverseEngineState) -> Optional[Tuple[int, int, Direction, float]]:
        legal_moves = self.engine.get_legal_moves(state)
        if not legal_moves:
            return None
        if len(legal_moves) == 1:
            r, c, d = legal_moves[0]
            return (r, c, d, 0.0)

        device = next(self.parameters()).device
        mask = torch.zeros((1, self.num_actions), dtype=torch.bool, device=device)
        for r, c, d in legal_moves:
            mask[0, encode_action_index(r, c, d)] = True

        with torch.no_grad():
            scores = self.forward(state, legal_action_mask=mask).squeeze(0)

        best_score = -1e9
        best_move = legal_moves[0]
        for r, c, d in legal_moves:
            idx = encode_action_index(r, c, d)
            s = float(scores[idx].item())
            if s > best_score:
                best_score = s
                best_move = (r, c, d)

        return (best_move[0], best_move[1], best_move[2], best_score)

    def solve_level(
        self,
        config_or_state: Union[LevelConfig, ReverseEngineState],
        max_moves: Optional[int] = None,
        seed: int = 42
    ) -> Dict[str, Any]:
        if isinstance(config_or_state, ReverseEngineState):
            state = config_or_state.copy()
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

        return {
            "model_name": "KarimiCandyRL2020 (Hybrid Intrinsic RL)",
            "level_id": state.config.level_id,
            "won": state.has_won(),
            "moves_used": step,
            "moves_remaining": state.moves_left,
            "final_score": state.score,
            "objectives_remaining": state.remaining_objectives_count(),
            "duration_sec": time.time() - t0,
            "parameters": self.count_parameters()
        }

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


if __name__ == "__main__":
    model = KarimiCandyRL2020()
    dummy = torch.randn(2, 20, 9, 9)
    out = model(dummy)
    print(f"KarimiCandyRL2020 initialized | Params: {model.count_parameters():,} | Output shape: {out.shape}")
