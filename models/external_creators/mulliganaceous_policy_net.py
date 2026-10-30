"""
Reproduction of the Open-Source `mulliganaceous` Residual Policy Network (`MulliganaceousPolicyNet`).

Background:
  Inspired by community open-source Match-3 / Candy Crush policy gradient agents and the
  `mulliganaceous` level analysis ecosystem. This architecture employs a compact 4-block
  Residual Convolutional Policy Network trained via REINFORCE / Policy Gradient with
  reward-to-go baseline subtraction, using local pair-swap directional convolution filters
  ([1, 2] horizontal and [2, 1] vertical swap kernels) to score all 144 swap coordinates directly.
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


class MulliganaceousResBlock(nn.Module):
    """Standard 2-layer 3x3 Residual Block with ReLU and BatchNorm."""

    def __init__(self, channels: int = 64):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        return F.relu(out + residual, inplace=True)


class MulliganaceousPolicyNet(nn.Module):
    """
    Open-Source `mulliganaceous` Residual Policy Gradient Network.
    Features:
      - 3x3 Conv stem + 4 Residual Blocks (hidden_dim=64)
      - Directional Swap Heads:
        * Horizontal Swap Conv (kernel 1x2) -> (B, 1, 9, 8) -> 72 actions
        * Vertical Swap Conv (kernel 2x1)   -> (B, 1, 8, 9) -> 72 actions
      - Concatenates into exact (B, 144) canonical action logits.
    """

    def __init__(
        self,
        in_channels: int = 20,
        hidden_dim: int = 64,
        num_res_blocks: int = 4,
        num_actions: int = NUM_CANONICAL_ACTIONS
    ):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_dim = hidden_dim
        self.num_actions = num_actions

        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True)
        )

        self.res_tower = nn.Sequential(
            *[MulliganaceousResBlock(hidden_dim) for _ in range(num_res_blocks)]
        )

        # Pairwise directional swap convolutions matching canonical action indexing:
        # [0..71]: Horizontal swaps (9x8), [72..143]: Vertical swaps (8x9)
        self.horiz_swap_head = nn.Sequential(
            nn.Conv2d(hidden_dim, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, kernel_size=(1, 2), stride=1, padding=0)  # Output: (B, 1, 9, 8)
        )
        self.vert_swap_head = nn.Sequential(
            nn.Conv2d(hidden_dim, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, kernel_size=(2, 1), stride=1, padding=0)  # Output: (B, 1, 8, 9)
        )

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(
        self,
        state_or_tensor: Union[ReverseEngineState, torch.Tensor, np.ndarray],
        legal_action_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Computes (B, 144) policy logits from (B, 20, 9, 9) tensor or ReverseEngineState.
        """
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

        feat = self.res_tower(self.stem(x))
        h_logits = self.horiz_swap_head(feat).flatten(1)  # (B, 72)
        v_logits = self.vert_swap_head(feat).flatten(1)   # (B, 72)
        logits = torch.cat([h_logits, v_logits], dim=1)   # (B, 144)

        # Incorporate local spatial objective & special candy affinity
        # Channels 6..9: Specials, 10..14: Blockers, 15..16: Jellies
        tactical_density = (
            x[:, 6:10].sum(dim=1) * 2.0
            + x[:, 15:17].sum(dim=1) * 1.2
            + x[:, 10:15].sum(dim=1) * 0.5
        )  # (B, 9, 9)
        h_pair = tactical_density[:, :, :-1] + tactical_density[:, :, 1:]  # (B, 9, 8)
        v_pair = tactical_density[:, :-1, :] + tactical_density[:, 1:, :]  # (B, 8, 9)
        pair_bonus = torch.cat([h_pair.flatten(1), v_pair.flatten(1)], dim=1)
        logits = logits + pair_bonus

        if legal_action_mask is not None:
            logits = logits.masked_fill(~legal_action_mask.to(device), -1e9)

        return logits

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class MulliganaceousPolicyAgent:
    """
    Agent wrapper for MulliganaceousPolicyNet.
    """

    def __init__(self, engine: Optional[ReverseCandyEngine] = None, device: str = "cpu"):
        self.device = torch.device(device)
        self.engine = engine or ReverseCandyEngine()
        self.net = MulliganaceousPolicyNet().to(self.device)
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
            logits = self.net(state, legal_action_mask=mask).squeeze(0)
            probs = F.softmax(logits, dim=-1)

        best_idx = int(torch.argmax(probs).item())
        r, c, d = decode_action_index(best_idx)
        if (r, c, d) not in legal_moves:
            # Check inverse direction equivalent in legal_moves
            best_prob = -1.0
            best_move = legal_moves[0]
            for lr, lc, ld in legal_moves:
                p = float(probs[encode_action_index(lr, lc, ld)].item())
                if p > best_prob:
                    best_prob = p
                    best_move = (lr, lc, ld)
            return (best_move[0], best_move[1], best_move[2], best_prob)

        return (r, c, d, float(probs[best_idx].item()))

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
            "model_name": "MulliganaceousPolicyNet (ResNet Policy)",
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
    net = MulliganaceousPolicyNet()
    dummy = torch.randn(2, 20, 9, 9)
    out = net(dummy)
    print(f"MulliganaceousPolicyNet initialized | Params: {net.count_parameters():,} | Output shape: {out.shape}")
