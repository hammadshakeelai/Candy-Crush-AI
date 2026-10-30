"""
AlphaCandy Model Zoo — Part 7.5: Reverse Curriculum & Minimum Moves-to-Go Estimator Network
(`ReverseCurriculumGeneratorNet`).
===========================================================================================
Based on:
  - Florensa et al. (CoRL 2017), "Reverse Curriculum Generation for Reinforcement Learning"
  - McAleer et al. (ICLR 2019), "Solving the Rubik's Cube with Approximate Policy Iteration" ( DeepCubeA )

Estimates the Minimum Moves-to-Go Complexity M_hat(s) of any board state (how many optimal swaps
are required to clear all remaining objectives) and pairs with a Reverse-State Perturbation Head
to train agents starting from 1-move-to-win states backward to 30-move opening states!
"""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


class ReverseCurriculumGeneratorNet(nn.Module):
    def __init__(
        self,
        in_channels: int = 20,
        d_model: int = 128,
        num_actions: int = 144,
        max_moves_horizon: int = 40,
    ) -> None:
        super().__init__()
        self.num_actions = num_actions
        self.max_moves_horizon = max_moves_horizon

        self.backbone = nn.Sequential(
            nn.Conv2d(in_channels, d_model, kernel_size=3, padding=1),
            nn.BatchNorm2d(d_model),
            nn.GELU(),
            nn.Conv2d(d_model, d_model, kernel_size=3, padding=1),
            nn.BatchNorm2d(d_model),
            nn.GELU(),
        )

        # Predicts categorical distribution over Moves-to-Victory (1 .. max_moves_horizon)
        self.moves_to_go_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(d_model, 128),
            nn.GELU(),
            nn.Linear(128, max_moves_horizon),
        )

        # Forward Solving Policy Head
        self.policy_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(d_model * 81, 256),
            nn.GELU(),
            nn.Linear(256, num_actions),
        )

        # Reverse Scrambling Policy Head (generates harder ancestor states from near-win states)
        self.reverse_scramble_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(d_model * 81, 128),
            nn.GELU(),
            nn.Linear(128, num_actions),
        )

        self.win_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(d_model, 1),
            nn.Sigmoid(),
        )

        bins = torch.arange(1, max_moves_horizon + 1, dtype=torch.float32).unsqueeze(0)
        self.register_buffer("move_bins", bins)

    def forward(self, board_tensor: torch.Tensor) -> Dict[str, torch.Tensor]:
        feat = self.backbone(board_tensor)
        policy_logits = self.policy_head(feat)  # (B, 144)
        reverse_logits = self.reverse_scramble_head(feat)  # (B, 144)
        win_prob = self.win_head(feat)  # (B, 1)

        mtg_logits = self.moves_to_go_head(feat)  # (B, 40)
        mtg_probs = F.softmax(mtg_logits, dim=-1)
        expected_moves_to_go = (mtg_probs * self.move_bins).sum(dim=-1, keepdim=True)  # (B, 1)

        return {
            "policy_logits": policy_logits,
            "win_prob": win_prob,
            "expected_moves_to_go": expected_moves_to_go,
            "moves_to_go_logits": mtg_logits,
            "reverse_scramble_logits": reverse_logits,
        }


if __name__ == "__main__":
    model = ReverseCurriculumGeneratorNet()
    x = torch.randn(2, 20, 9, 9)
    out = model(x)
    params = sum(p.numel() for p in model.parameters())
    print(f"ReverseCurriculumGeneratorNet: {params:,} params | moves_to_go={out['expected_moves_to_go'].shape}")
