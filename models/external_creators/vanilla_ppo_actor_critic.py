"""
Reproduction of Standard Open-Source GitHub Match-3 PPO Actor-Critic Baseline (`VanillaPPOMatch3Net`).

Background:
  Standard Proximal Policy Optimization (PPO) Actor-Critic network architecture commonly used
  in open-source Match-3 RL repositories (e.g., `kamildar/gym-match3`, `maomao0819/Match3-game-with-RL`,
  `AliA74/match-3-RL-Environment`, CleanRL discrete-action PPO with invalid action masking).
  Features:
    - Shared 4-layer 3x3 CNN feature trunk with orthogonal weight initialization
    - Actor Head: Linear projection to 144 swap logits with invalid action masking
    - Critic Head: Scalar state-value estimation V(s)
"""

from typing import List, Tuple, Dict, Optional, Any, Union
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from core.reverse_engine import (
    ReverseEngineState, ReverseCandyEngine, Direction, LevelConfig, OFFSETS
)
from models.spatial_transformer_net import (
    encode_spatial_board_tensor, encode_action_index, decode_action_index, NUM_CANONICAL_ACTIONS
)


def _layer_init(layer: nn.Module, std: float = np.sqrt(2), bias_const: float = 0.0) -> nn.Module:
    if hasattr(layer, "weight") and layer.weight is not None:
        nn.init.orthogonal_(layer.weight, std)
    if hasattr(layer, "bias") and layer.bias is not None:
        nn.init.constant_(layer.bias, bias_const)
    return layer


class VanillaPPOMatch3Net(nn.Module):
    """
    Standard 4-Layer Conv + Actor-Critic PPO Network for Match-3 Environments.
    Outputs action logits (B, 144) and scalar value V(s) (B, 1).
    """

    def __init__(
        self,
        in_channels: int = 20,
        hidden_channels: int = 64,
        num_actions: int = NUM_CANONICAL_ACTIONS
    ):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.num_actions = num_actions

        # Shared 4-Layer CNN Feature Extractor (CleanRL / gym-match3 convention)
        self.shared_cnn = nn.Sequential(
            _layer_init(nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1)),
            nn.ReLU(inplace=True),
            _layer_init(nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1)),
            nn.ReLU(inplace=True),
            _layer_init(nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1)),
            nn.ReLU(inplace=True),
            _layer_init(nn.Conv2d(hidden_channels, 32, kernel_size=3, padding=1)),
            nn.ReLU(inplace=True),
            nn.Flatten()
        )

        flat_dim = 32 * 9 * 9
        self.shared_fc = nn.Sequential(
            _layer_init(nn.Linear(flat_dim, 256)),
            nn.ReLU(inplace=True)
        )

        # Actor Head: outputs 144 action logits (initialized with small std=0.01 per PPO best practice)
        self.actor_head = _layer_init(nn.Linear(256, num_actions), std=0.01)

        # Critic Head: outputs state value V(s) (initialized with std=1.0)
        self.critic_head = _layer_init(nn.Linear(256, 1), std=1.0)

    def forward_actor_critic(
        self,
        state_or_tensor: Union[ReverseEngineState, torch.Tensor, np.ndarray],
        legal_action_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns (action_logits: (B, 144), state_value: (B, 1))."""
        if isinstance(state_or_tensor, ReverseEngineState):
            arr = encode_spatial_board_tensor(state_or_tensor)
            x = torch.from_numpy(arr).unsqueeze(0)
        elif isinstance(state_or_tensor, np.ndarray):
            x = torch.from_numpy(state_or_tensor)
            if x.dim() == 3:
                x = x.unsqueeze(0)
        else:
            x = state_or_tensor
            if x.dim() == 3:
                x = x.unsqueeze(0)

        device = next(self.parameters()).device
        x = x.to(device=device, dtype=torch.float32)

        latent = self.shared_fc(self.shared_cnn(x))
        logits = self.actor_head(latent)
        value = self.critic_head(latent)

        if legal_action_mask is not None:
            logits = logits.masked_fill(~legal_action_mask.to(device), -1e9)

        return logits, value

    def forward(
        self,
        state_or_tensor: Union[ReverseEngineState, torch.Tensor, np.ndarray],
        legal_action_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Default forward returning (B, 144) action score logits."""
        logits, _ = self.forward_actor_critic(state_or_tensor, legal_action_mask=legal_action_mask)
        return logits

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class VanillaPPOAgent:
    """
    Agent wrapper for VanillaPPOMatch3Net.
    """

    def __init__(self, engine: Optional[ReverseCandyEngine] = None, device: str = "cpu"):
        self.device = torch.device(device)
        self.engine = engine or ReverseCandyEngine()
        self.net = VanillaPPOMatch3Net().to(self.device)
        self.net.eval()

    def select_best_move(self, state: ReverseEngineState) -> Optional[Tuple[int, int, Direction, float]]:
        legal_moves = self.engine.get_legal_moves(state)
        if not legal_moves:
            return None
        if len(legal_moves) == 1:
            r, c, d = legal_moves[0]
            return (r, c, d, 1.0)

        mask = torch.zeros((1, NUM_CANONICAL_ACTIONS), dtype=torch.bool, device=self.device)
        for r, c, d in legal_moves:
            mask[0, encode_action_index(r, c, d)] = True

        with torch.no_grad():
            logits, _ = self.net.forward_actor_critic(state, legal_action_mask=mask)
            probs = F.softmax(logits.squeeze(0), dim=-1)

        best_prob = -1.0
        best_move = legal_moves[0]
        for r, c, d in legal_moves:
            idx = encode_action_index(r, c, d)
            p = float(probs[idx].item())
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
            "model_name": "VanillaPPOMatch3Net (Actor-Critic PPO)",
            "level_id": state.config.level_id,
            "won": state.has_won(),
            "moves_used": step,
            "moves_remaining": state.moves_left,
            "final_score": state.score,
            "objectives_remaining": state.remaining_objectives_count(),
            "duration_sec": time.time() - t0,
            "parameters": self.net.count_parameters()
        }


if __name__ == "__main__":
    net = VanillaPPOMatch3Net()
    dummy = torch.randn(2, 20, 9, 9)
    logits, val = net.forward_actor_critic(dummy)
    print(f"VanillaPPOMatch3Net initialized | Params: {net.count_parameters():,} | Logits: {logits.shape} | Val: {val.shape}")
