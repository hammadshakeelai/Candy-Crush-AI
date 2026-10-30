"""Contrastive Trajectory Preference Network (Direct Preference Optimization).

Trains directly on pairs of winning vs. losing attempt trajectories on the same
level seed using the Rafailov et al. (2023) DPO objective, bypassing explicit
reward shaping while extracting an implicit per-action advantage score.
"""

from typing import Dict, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class DirectPreferenceCandyNet(nn.Module):
    """DPO policy network for Candy Crush trajectory preference alignment.

    Outputs:
        - policy_logits: (B, 144) preference-aligned action logits
        - win_prob: (B, 1) estimated probability of winning from current state
        - implicit_reward: (B, 144) centered log-probability advantage r(s, a)
    """

    def __init__(
        self,
        in_channels: int = 20,
        hidden_dim: int = 128,
        num_actions: int = 144,
        dpo_beta: float = 0.1,
    ) -> None:
        super().__init__()
        self.num_actions = num_actions
        self.dpo_beta = dpo_beta

        self.conv_tower = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.GELU(),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.GELU(),
            nn.Conv2d(hidden_dim, hidden_dim // 2, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden_dim // 2),
            nn.GELU(),
        )

        flat_dim = (hidden_dim // 2) * 81
        self.state_proj = nn.Sequential(
            nn.Linear(flat_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
        )

        self.policy_head = nn.Linear(256, num_actions)
        self.ref_baseline_head = nn.Linear(256, num_actions)
        # Freeze reference baseline head by default so implicit_reward measures policy shift
        for param in self.ref_baseline_head.parameters():
            param.requires_grad_(False)

        self.value_head = nn.Sequential(
            nn.Linear(256, 64),
            nn.GELU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        board_tensor: torch.Tensor,
        legal_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        feat = self.conv_tower(board_tensor).flatten(1)
        state_emb = self.state_proj(feat)

        policy_logits = self.policy_head(state_emb)
        ref_logits = self.ref_baseline_head(state_emb)

        if legal_mask is not None:
            policy_logits = policy_logits.masked_fill(~legal_mask.bool(), -1e9)
            ref_logits = ref_logits.masked_fill(~legal_mask.bool(), -1e9)

        log_pi = F.log_softmax(policy_logits, dim=-1)
        log_ref = F.log_softmax(ref_logits, dim=-1)
        implicit_reward = self.dpo_beta * (log_pi - log_ref)

        win_prob = self.value_head(state_emb)
        return {
            "policy_logits": policy_logits,
            "win_prob": win_prob,
            "implicit_reward": implicit_reward,
        }

    @staticmethod
    def compute_dpo_loss(
        win_logits: torch.Tensor,
        lose_logits: torch.Tensor,
        ref_win_logits: torch.Tensor,
        ref_lose_logits: torch.Tensor,
        win_actions: torch.Tensor,
        lose_actions: torch.Tensor,
        beta: float = 0.1,
    ) -> Dict[str, torch.Tensor]:
        """Computes the Direct Preference Optimization loss over winning vs. losing swaps."""
        pi_win_logp = F.log_softmax(win_logits, dim=-1).gather(-1, win_actions.unsqueeze(-1)).squeeze(-1)
        pi_lose_logp = F.log_softmax(lose_logits, dim=-1).gather(-1, lose_actions.unsqueeze(-1)).squeeze(-1)

        with torch.no_grad():
            ref_win_logp = F.log_softmax(ref_win_logits, dim=-1).gather(-1, win_actions.unsqueeze(-1)).squeeze(-1)
            ref_lose_logp = F.log_softmax(ref_lose_logits, dim=-1).gather(-1, lose_actions.unsqueeze(-1)).squeeze(-1)

        win_logratios = pi_win_logp - ref_win_logp
        lose_logratios = pi_lose_logp - ref_lose_logp
        margin = beta * (win_logratios - lose_logratios)

        loss = -F.logsigmoid(margin).mean()
        pref_accuracy = (margin > 0).float().mean()
        return {
            "loss": loss,
            "preference_accuracy": pref_accuracy,
            "reward_margin": margin.detach().mean(),
        }
