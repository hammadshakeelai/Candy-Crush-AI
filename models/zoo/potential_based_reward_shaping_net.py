"""
AlphaCandy Model Zoo — Part 7.4: Policy-Invariant Potential-Based Reward Shaping Network
(`NgPotentialRewardShapingNet`).
========================================================================================
Based on:
  - Ng, Harada, & Russell (ICML 1999), "Policy invariance under reward transformations:
    Theory and application to reward shaping"
  - Wiewiora et al. (ICML 2003), "Principled Methods for Advising Reinforcement Learning Agents"

Learns a Differentiable State Potential Function Phi_theta(s) in R measuring distance-to-victory
(weighted jelly clearance, ingredient drop depth, special candy inventory, blocker openness) such
that the shaped reward:
    R'(s, a, s') = R(s, a, s') + gamma * Phi_theta(s') - Phi_theta(s)
provides dense per-move credit assignment with ZERO distortion to the optimal win policy!
"""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


class NgPotentialRewardShapingNet(nn.Module):
    def __init__(
        self,
        in_channels: int = 20,
        d_model: int = 128,
        num_actions: int = 144,
        gamma: float = 0.99,
    ) -> None:
        super().__init__()
        self.gamma = gamma
        self.num_actions = num_actions

        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, d_model, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(d_model, d_model, kernel_size=3, padding=1),
            nn.GELU(),
        )

        # Learned State Potential Phi_theta(s) decomposed into 4 interpretable potential sub-terms
        self.potential_subheads = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(d_model, 64),
            nn.GELU(),
            nn.Linear(64, 4),  # [Phi_jelly, Phi_ingredient, Phi_special_inventory, Phi_openness]
        )

        # Action Advantage & Policy Head
        self.policy_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(d_model * 81, 256),
            nn.GELU(),
            nn.Linear(256, num_actions),
        )

        self.win_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(d_model, 1),
            nn.Sigmoid(),
        )

    def forward(self, board_tensor: torch.Tensor) -> Dict[str, torch.Tensor]:
        feat = self.encoder(board_tensor)
        sub_potentials = self.potential_subheads(feat)  # (B, 4)
        state_potential = sub_potentials.sum(dim=-1, keepdim=True)  # (B, 1)
        policy_logits = self.policy_head(feat)  # (B, 144)
        win_prob = self.win_head(feat)  # (B, 1)

        return {
            "policy_logits": policy_logits,
            "win_prob": win_prob,
            "state_potential": state_potential,
            "potential_components": sub_potentials,
        }

    def compute_shaped_reward(
        self,
        state: torch.Tensor,
        next_state: torch.Tensor,
        extrinsic_reward: torch.Tensor,
        done: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Computes policy-invariant potential-shaped reward R' = R + gamma * Phi(s') - Phi(s)."""
        phi_s = self.forward(state)["state_potential"].squeeze(-1)
        phi_s_next = self.forward(next_state)["state_potential"].squeeze(-1)
        if done is not None:
            phi_s_next = phi_s_next * (1.0 - done.float())
        return extrinsic_reward + self.gamma * phi_s_next - phi_s


if __name__ == "__main__":
    model = NgPotentialRewardShapingNet()
    s = torch.randn(2, 20, 9, 9)
    s_next = torch.randn(2, 20, 9, 9)
    out = model(s)
    r_shaped = model.compute_shaped_reward(s, s_next, torch.tensor([0.0, 1.0]))
    params = sum(p.numel() for p in model.parameters())
    print(f"NgPotentialRewardShapingNet: {params:,} params | phi={out['state_potential'].shape} | r_shaped={r_shaped.shape}")
