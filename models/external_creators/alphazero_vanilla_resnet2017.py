"""
Reproduction of Silver et al. (Nature 2017 / Science 2018) Vanilla AlphaZero Architecture:
"Mastering the game of Go without human knowledge" & "A general reinforcement learning algorithm
that masters chess, shogi, and Go through self-play" — David Silver, Thomas Hubert, Julian Schrittwieser,
Ioannis Antonoglou, Matthew Lai, Arthur Guez, Marc Lanctot, Laurent Sifre, Dharshan Kumaran,
Thore Graepel, Timothy Lillicrap, Karen Simonyan, Demis Hassabis (DeepMind).

Architectural Specifications (faithful to Silver et al., 2017, adapted to 9x9 Match-3 grid):
  - Input: Canonical (B, 20, 9, 9) multi-channel spatial board tensor (or ReverseEngineState).
  - Stem: 3x3 Convolution (20 -> 256 channels, stride=1, padding=1, bias=False) + BatchNorm2d + ReLU.
  - Residual Tower: 10 Residual Blocks, each consisting of:
      * 3x3 Conv2d(256, 256, padding=1, bias=False) + BatchNorm2d(256) + ReLU
      * 3x3 Conv2d(256, 256, padding=1, bias=False) + BatchNorm2d(256) + skip connection + ReLU
  - Policy Head:
      * 1x1 Conv2d(256, 2, bias=False) + BatchNorm2d(2) + ReLU
      * Flatten (2 * 9 * 9 = 162) -> Linear(162, 144) canonical swap action logits
  - Value Head:
      * 1x1 Conv2d(256, 1, bias=False) + BatchNorm2d(1) + ReLU
      * Flatten (1 * 9 * 9 = 81) -> Linear(81, 256) + ReLU -> Linear(256, 1) + Tanh (or Sigmoid)
  - Ablation Role: Pure Euclidean 3x3 convolutional ResNet with NO portal graph attention,
    NO S_6 color permutation equivariance, and NO distributional quantile risk heads.
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


class AlphaZeroResidualBlock2017(nn.Module):
    """
    Canonical Silver et al. (2017) Residual Tower Block:
    Two 3x3 convolutions with 256 filters, batch normalization, and ReLU skip connection.
    """

    def __init__(self, channels: int = 256):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        return F.relu(out + residual, inplace=True)


class AlphaZeroVanilla2017Net(nn.Module):
    """
    Faithful reproduction of Silver et al. (Nature 2017) Vanilla AlphaZero Dual-Head ResNet
    adapted to 9x9 Candy Crush (10 Residual Blocks, 256 channels, 1x1 Policy & Value Heads).
    Serves as the canonical Euclidean AlphaZero ablation baseline (no portal attention,
    no color equivariance, no quantile risk distribution).
    """

    def __init__(
        self,
        in_channels: int = 20,
        num_blocks: int = 10,
        channels: int = 256,
        num_actions: int = NUM_CANONICAL_ACTIONS,
        value_activation: str = "tanh"
    ):
        super().__init__()
        self.in_channels = in_channels
        self.num_blocks = num_blocks
        self.channels = channels
        self.num_actions = num_actions
        self.value_activation = value_activation

        # Initial Convolutional Block (Silver et al., 2017)
        self.conv_input = nn.Sequential(
            nn.Conv2d(in_channels, channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )

        # 10-Block Residual Tower (256 channels)
        self.residual_tower = nn.ModuleList([
            AlphaZeroResidualBlock2017(channels) for _ in range(num_blocks)
        ])

        # Standard 1x1 Conv Policy Head (2 filters -> FC to 144 actions)
        self.policy_conv = nn.Sequential(
            nn.Conv2d(channels, 2, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(2),
            nn.ReLU(inplace=True)
        )
        self.policy_fc = nn.Linear(2 * 9 * 9, num_actions)

        # Standard 1x1 Conv Scalar Value Head (1 filter -> 256 hidden -> 1 scalar)
        self.value_conv = nn.Sequential(
            nn.Conv2d(channels, 1, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(1),
            nn.ReLU(inplace=True)
        )
        self.value_fc1 = nn.Linear(1 * 9 * 9, 256)
        self.value_fc2 = nn.Linear(256, 1)

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _prepare_input(
        self, state_or_tensor: Union[ReverseEngineState, torch.Tensor, np.ndarray]
    ) -> torch.Tensor:
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
        return x.to(device=device, dtype=torch.float32)

    def forward(
        self,
        state_or_tensor: Union[ReverseEngineState, torch.Tensor, np.ndarray],
        legal_action_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass returning:
          - policy_logits: (B, 144) action logits
          - value: (B, 1) scalar state value estimate
        """
        x = self._prepare_input(state_or_tensor)

        h = self.conv_input(x)
        for block in self.residual_tower:
            h = block(h)

        # Policy Head
        p = self.policy_conv(h).flatten(start_dim=1)
        policy_logits = self.policy_fc(p)

        if legal_action_mask is not None:
            policy_logits = policy_logits.masked_fill(~legal_action_mask.to(policy_logits.device), -1e9)

        # Value Head
        v = self.value_conv(h).flatten(start_dim=1)
        v = F.relu(self.value_fc1(v), inplace=True)
        v_raw = self.value_fc2(v)
        if self.value_activation == "sigmoid":
            value = torch.sigmoid(v_raw)
        else:
            value = torch.tanh(v_raw)

        return policy_logits, value

    def forward_actor_critic(
        self,
        state_or_tensor: Union[ReverseEngineState, torch.Tensor, np.ndarray],
        legal_action_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Alias returning (policy_logits: (B, 144), value: (B, 1))."""
        return self.forward(state_or_tensor, legal_action_mask=legal_action_mask)

    def get_action_logits(
        self,
        state_or_tensor: Union[ReverseEngineState, torch.Tensor, np.ndarray],
        legal_action_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Returns only (B, 144) policy logits."""
        logits, _ = self.forward(state_or_tensor, legal_action_mask=legal_action_mask)
        return logits

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class AlphaZeroVanilla2017Agent:
    """
    Agent wrapper around AlphaZeroVanilla2017Net for benchmarking and level evaluation.
    """

    def __init__(self, engine: Optional[ReverseCandyEngine] = None, device: str = "cpu"):
        self.device = torch.device(device)
        self.engine = engine or ReverseCandyEngine()
        self.net = AlphaZeroVanilla2017Net().to(self.device)
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
            logits, _ = self.net(state, legal_action_mask=mask)
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
            "model_name": "AlphaZeroVanilla2017Net (Silver et al., 2017)",
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
    net = AlphaZeroVanilla2017Net()
    dummy = torch.randn(2, 20, 9, 9)
    logits, val = net(dummy)
    print(f"AlphaZeroVanilla2017Net | Params: {net.count_parameters():,} | Logits: {logits.shape} | Value: {val.shape}")
