"""Opening-Phase Board Architect Specialist Network.

Focuses on the first 25% of level moves where clearing blockers near the top,
uncapping candy dispensers, and maximizing board openness (expected legal moves)
dominate immediate point scoring.
"""

from typing import Dict, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class DepthwiseSeparableResBlock(nn.Module):
    """Compact residual block with depthwise-separable convolutions and SE gating."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.dw = nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.pw = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, max(channels // 4, 8), kernel_size=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(max(channels // 4, 8), channels, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.silu(self.bn1(self.dw(x)))
        out = self.bn2(self.pw(out))
        out = out * self.se(out)
        return F.silu(x + out)


class OpeningBoardArchitectNet(nn.Module):
    """Opening-phase specialist network for Candy Crush (first 25% of moves).

    Predicts:
        - policy_logits: (B, 144) action preference logits
        - win_prob: (B, 1) estimated level clear probability
        - board_openness_score: (B, 144) expected legal moves unlocked per action
        - dispenser_uncap_bonus: (B, 144) expected top-row/dispenser unblocking utility
    """

    def __init__(
        self,
        in_channels: int = 20,
        hidden_dim: int = 128,
        num_blocks: int = 4,
        num_actions: int = 144,
        opening_weight: float = 0.35,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.hidden_dim = hidden_dim
        self.num_actions = num_actions
        self.opening_weight = opening_weight

        # Row-coordinate positional prior emphasizing top rows (dispensers / gravity flow)
        row_prior = torch.linspace(1.0, -1.0, steps=9).view(1, 1, 9, 1).expand(1, 1, 9, 9)
        col_prior = torch.linspace(-1.0, 1.0, steps=9).view(1, 1, 1, 9).expand(1, 1, 9, 9)
        self.register_buffer("spatial_prior", torch.cat([row_prior, col_prior], dim=1))

        self.stem = nn.Sequential(
            nn.Conv2d(in_channels + 2, hidden_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.SiLU(inplace=True),
        )

        self.backbone = nn.ModuleList(
            [DepthwiseSeparableResBlock(hidden_dim) for _ in range(num_blocks)]
        )

        # Top-gravity flow branch (vertical strip pooling along columns)
        self.column_flow_conv = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim // 2, kernel_size=(9, 1), groups=1, bias=False),
            nn.BatchNorm2d(hidden_dim // 2),
            nn.SiLU(inplace=True),
        )

        flat_dim = (hidden_dim // 4) * 81 + (hidden_dim // 2) * 9
        self.spatial_reduce = nn.Conv2d(hidden_dim, hidden_dim // 4, kernel_size=1)

        self.shared_trunk = nn.Sequential(
            nn.Linear(flat_dim, 256),
            nn.LayerNorm(256),
            nn.SiLU(inplace=True),
        )

        self.base_policy_head = nn.Linear(256, num_actions)
        self.openness_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.SiLU(inplace=True),
            nn.Linear(128, num_actions),
        )
        self.dispenser_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.SiLU(inplace=True),
            nn.Linear(128, num_actions),
        )
        self.value_head = nn.Sequential(
            nn.Linear(256, 64),
            nn.SiLU(inplace=True),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        board_tensor: torch.Tensor,
        legal_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        b = board_tensor.size(0)
        priors = self.spatial_prior.expand(b, -1, -1, -1)
        x = self.stem(torch.cat([board_tensor, priors], dim=1))

        for block in self.backbone:
            x = block(x)

        col_features = self.column_flow_conv(x).flatten(1)
        grid_features = F.silu(self.spatial_reduce(x)).flatten(1)
        latent = self.shared_trunk(torch.cat([grid_features, col_features], dim=1))

        base_logits = self.base_policy_head(latent)
        board_openness_score = F.softplus(self.openness_head(latent))
        dispenser_uncap_bonus = torch.tanh(self.dispenser_head(latent))

        policy_logits = base_logits + self.opening_weight * (
            board_openness_score + dispenser_uncap_bonus
        )
        if legal_mask is not None:
            policy_logits = policy_logits.masked_fill(~legal_mask.bool(), -1e9)

        win_prob = self.value_head(latent)
        return {
            "policy_logits": policy_logits,
            "win_prob": win_prob,
            "board_openness_score": board_openness_score,
            "dispenser_uncap_bonus": dispenser_uncap_bonus,
        }
