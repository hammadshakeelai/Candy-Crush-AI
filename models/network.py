"""
Dual-Head Policy-Value ResNet for Candy Crush.
Similar to AlphaZero / MuZero architectures adapted for stochastic match-3 grids.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
import numpy as np


class ResidualBlock(nn.Module):
    """Convolutional residual block preserving spatial dimensions."""

    def __init__(self, channels: int):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        return F.relu(out)


class CandyCrushPolicyValueNet(nn.Module):
    """
    Dual-Head Residual Network.
    
    Inputs:
      - board_tensor: (B, 19, H, W)
    Outputs:
      - policy_logits: (B, H * W * 2) - Probability distribution over legal moves
      - value: (B, 1) - Expected win probability / game utility [-1, 1]
    """

    def __init__(
        self,
        in_channels: int = 19,
        num_res_blocks: int = 4,
        channels: int = 64,
        height: int = 9,
        width: int = 9
    ):
        super(CandyCrushPolicyValueNet, self).__init__()
        self.height = height
        self.width = width
        self.num_actions = height * width * 2

        # Initial feature extractor
        self.initial_conv = nn.Sequential(
            nn.Conv2d(in_channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU()
        )

        # Residual tower
        self.res_blocks = nn.ModuleList([
            ResidualBlock(channels) for _ in range(num_res_blocks)
        ])

        # Policy Head
        self.policy_conv = nn.Sequential(
            nn.Conv2d(channels, 32, kernel_size=1),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        self.policy_fc = nn.Linear(32 * height * width, self.num_actions)

        # Value Head
        self.value_conv = nn.Sequential(
            nn.Conv2d(channels, 16, kernel_size=1),
            nn.BatchNorm2d(16),
            nn.ReLU()
        )
        self.value_fc = nn.Sequential(
            nn.Linear(16 * height * width, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Tanh()  # Output range [-1.0, 1.0]
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Feature representation
        h = self.initial_conv(x)
        for block in self.res_blocks:
            h = block(h)

        # Policy logits
        p = self.policy_conv(h)
        p = p.flatten(start_dim=1)
        policy_logits = self.policy_fc(p)

        # Value scalar
        v = self.value_conv(h)
        v = v.flatten(start_dim=1)
        value = self.value_fc(v)

        return policy_logits, value

    def forward_with_mask(
        self, x: torch.Tensor, action_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass with action masking over illegal match-3 swaps.
        Returns:
          - masked_logits: (B, num_actions) with -1e9 for illegal moves
          - action_probs: (B, num_actions) normalized probability distribution
          - value: (B, 1) expected win probability in [-1, 1]
        """
        policy_logits, value = self.forward(x)
        if action_mask is not None:
            # Mask out illegal actions with large negative value
            masked_logits = torch.where(action_mask.bool(), policy_logits, torch.full_like(policy_logits, -1e9))
        else:
            masked_logits = policy_logits

        action_probs = F.softmax(masked_logits, dim=-1)
        return masked_logits, action_probs, value


def encode_19ch_tensor(state) -> np.ndarray:
    """Encodes state to (19, H, W) tensor."""
    channels = 19
    h, w = state.height, state.width
    tensor = np.zeros((channels, h, w), dtype=np.float32)

    for color_val in range(7):
        tensor[color_val] = (state.colors == color_val).astype(np.float32)
    for s_idx in range(5):
        tensor[7 + s_idx] = (state.specials == s_idx).astype(np.float32)
    for j_val in range(3):
        tensor[12 + j_val] = (state.jellies == j_val).astype(np.float32)

    tensor[15] = (state.blockers == 1).astype(np.float32)  # Chocolate
    tensor[16] = (state.blockers == 2).astype(np.float32)  # Licorice
    tensor[17] = (state.blockers == 3).astype(np.float32)  # Frosting
    tensor[18].fill(min(1.0, max(0.0, state.moves_left / 50.0)))
    return tensor


def encode_advanced_spatial_tensor(
    state,
    mechanics_mgr=None
) -> np.ndarray:
    """
    Constructs a 28-channel Spatial Feature Tensor integrating all reverse-engineered
    Candy Crush mechanics (Portals, Cannons, Mixers, Cakes, Toffee, Popcorn, Frogs).
    """
    channels = 28
    h, w = state.height, state.width
    tensor = np.zeros((channels, h, w), dtype=np.float32)

    # Channels 0-6: Candy Colors
    for c_val in range(7):
        tensor[c_val] = (state.colors == c_val).astype(np.float32)

    # Channels 7-11: Special Candies
    for s_val in range(5):
        tensor[7 + s_val] = (state.specials == s_val).astype(np.float32)

    # Channels 12-14: Jellies
    for j_val in range(3):
        tensor[12 + j_val] = (state.jellies == j_val).astype(np.float32)

    # Channels 15-17: Primary Blockers
    tensor[15] = (state.blockers == 1).astype(np.float32)  # Chocolate
    tensor[16] = (state.blockers == 2).astype(np.float32)  # Licorice
    tensor[17] = (state.blockers == 3).astype(np.float32)  # Frosting

    # Channel 18: Moves Left Normalized
    tensor[18].fill(min(1.0, max(0.0, state.moves_left / 50.0)))

    # Channel 19: Portals Map
    if state.config.portals:
        for pair in state.config.portals:
            if isinstance(pair, list) and len(pair) >= 2:
                src, dst = pair[0], pair[1]
                if len(src) >= 2 and len(dst) >= 2:
                    sc, sr = int(src[0]), int(src[1])
                    dc, dr = int(dst[0]), int(dst[1])
                    if 0 <= sr < h and 0 <= sc < w:
                        tensor[19, sr, sc] = -1.0  # Portal Entrance
                    if 0 <= dr < h and 0 <= dc < w:
                        tensor[19, dr, dc] = 1.0   # Portal Exit

    # Channel 20: Cannons / Dispensers
    if state.config.cannons:
        for col_idx, items in enumerate(state.config.cannons):
            if isinstance(items, list) and len(items) > 0 and 0 <= col_idx < w:
                tensor[20, 0, col_idx] = 1.0

    # Channel 21-25: Mechanics Manager
    if mechanics_mgr:
        # 21: Magic Mixers
        for mixer in mechanics_mgr.magic_mixers:
            if 0 <= mixer.row < h and 0 <= mixer.col < w:
                tensor[21, mixer.row, mixer.col] = mixer.countdown / float(max(1, mixer.interval))

        # 22: Cake Bombs
        for cake in mechanics_mgr.cake_bombs:
            for dr in [0, 1]:
                for dc in [0, 1]:
                    cr, cc = cake.top_left_r + dr, cake.top_left_c + dc
                    if 0 <= cr < h and 0 <= cc < w:
                        tensor[22, cr, cc] = cake.slices_remaining / 8.0

        # 23: Toffee Swirls
        for (tr, tc), toffee in mechanics_mgr.toffee_swirls.items():
            if 0 <= tr < h and 0 <= tc < w:
                tensor[23, tr, tc] = toffee.layers / 4.0

        # 24: Popcorn
        for (pr, pc), popcorn in mechanics_mgr.popcorns.items():
            if 0 <= pr < h and 0 <= pc < w:
                tensor[24, pr, pc] = popcorn.hp / 3.0

        # 25: Frog
        if mechanics_mgr.frog:
            fr, fc = mechanics_mgr.frog.row, mechanics_mgr.frog.col
            if 0 <= fr < h and 0 <= fc < w:
                tensor[25, fr, fc] = mechanics_mgr.frog.current_fullness / float(max(1, mechanics_mgr.frog.stomach_capacity))

    # Channel 26: Playable Grid Mask
    tensor[26] = state.active_grid.astype(np.float32)

    # Channel 27: Ingredients
    tensor[27] = (state.ingredients > 0).astype(np.float32)

    return tensor

