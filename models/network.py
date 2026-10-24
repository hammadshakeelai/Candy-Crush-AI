"""
Dual-Head Policy-Value ResNet for Candy Crush.
Similar to AlphaZero / MuZero architectures adapted for stochastic match-3 grids.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


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
