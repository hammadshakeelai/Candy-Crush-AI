"""
AlphaCandy Model Zoo — Part 6.6: Distributional Quantile + Aleatoric Variance MCTS Hybrid
(`QuantileMCTSHybridNet`).
=========================================================================================
Designed specifically as a fast neural prior for Monte Carlo Tree / Graph Search (MCTS/MCGS):
  - Outputs PUCT prior logits (B, 144)
  - Outputs 16-Quantile Value Distribution Z(s, a, tau_k) in R^{B x 144 x 16}
  - Outputs an explicit Aleatoric Rollout Variance Head sigma^2(s, a) in R^{B x 144} so search
    allocates more stochastic chance-node rollouts to high-variance cascade moves!
"""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


class QuantileMCTSHybridNet(nn.Module):
    def __init__(
        self,
        in_channels: int = 20,
        d_model: int = 128,
        num_actions: int = 144,
        num_quantiles: int = 16,
    ) -> None:
        super().__init__()
        self.num_actions = num_actions
        self.num_quantiles = num_quantiles

        self.backbone = nn.Sequential(
            nn.Conv2d(in_channels, d_model, kernel_size=3, padding=1),
            nn.BatchNorm2d(d_model),
            nn.GELU(),
            nn.Conv2d(d_model, d_model, kernel_size=3, padding=1),
            nn.BatchNorm2d(d_model),
            nn.GELU(),
            nn.Conv2d(d_model, d_model, kernel_size=3, padding=1),
            nn.GELU(),
        )

        self.puct_prior_head = nn.Sequential(
            nn.Conv2d(d_model, 32, kernel_size=1),
            nn.GELU(),
            nn.Flatten(),
            nn.Linear(32 * 81, num_actions),
        )

        self.quantile_head = nn.Sequential(
            nn.Conv2d(d_model, 32, kernel_size=1),
            nn.GELU(),
            nn.Flatten(),
            nn.Linear(32 * 81, num_actions * num_quantiles),
        )

        self.aleatoric_variance_head = nn.Sequential(
            nn.Conv2d(d_model, 16, kernel_size=1),
            nn.GELU(),
            nn.Flatten(),
            nn.Linear(16 * 81, num_actions),
            nn.Softplus(),
        )

        self.win_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(d_model, 64),
            nn.GELU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, board_tensor: torch.Tensor) -> Dict[str, torch.Tensor]:
        B = board_tensor.size(0)
        feat = self.backbone(board_tensor)
        policy_logits = self.puct_prior_head(feat)  # (B, 144)
        quantiles = self.quantile_head(feat).view(B, self.num_actions, self.num_quantiles)  # (B, 144, 16)
        quantiles = torch.sort(quantiles, dim=-1).values
        aleatoric_var = self.aleatoric_variance_head(feat) + 1e-4  # (B, 144)
        win_prob = self.win_head(feat)  # (B, 1)

        # Variance-Bonus PUCT Score for search allocation
        variance_puct_score = policy_logits + 0.5 * torch.sqrt(aleatoric_var)

        return {
            "policy_logits": policy_logits,
            "win_prob": win_prob,
            "q_quantiles": quantiles,
            "aleatoric_variance": aleatoric_var,
            "variance_puct_score": variance_puct_score,
        }


if __name__ == "__main__":
    model = QuantileMCTSHybridNet()
    x = torch.randn(2, 20, 9, 9)
    out = model(x)
    params = sum(p.numel() for p in model.parameters())
    print(f"QuantileMCTSHybridNet: {params:,} params | quantiles={out['q_quantiles'].shape} | var={out['aleatoric_variance'].shape}")
