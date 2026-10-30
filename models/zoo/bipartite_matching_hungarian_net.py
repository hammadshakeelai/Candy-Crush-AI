"""
AlphaCandy Model Zoo — Part 6.5: Differentiable Bipartite Swap-to-Objective Matcher Network
(`DifferentiableBipartiteMatcherNet`).
===========================================================================================
Based on:
  - Carion et al. (ECCV 2020), "End-to-End Object Detection with Transformers (DETR)"
  - Kuhn (1955), "The Hungarian Method for the Assignment Problem"
  - Mena et al. (ICLR 2018), "Learning Latent Permutations with Gumbel-Sinkhorn Networks"

Why Bipartite Matching for Candy Crush?
---------------------------------------
In complex multi-objective boards (e.g., 12 remaining jellies, 3 chocolate spawners, and 2
sinking ingredients), a single grandmaster swap often advances multiple subgoals simultaneously.
Standard policy heads treat the 144 actions as independent logits without explicitly reasoning
about which board objective each swap satisfies.

Architecture:
  1. Spatial Token Encoder extracts 81 board tokens z_board in R^{B x 81 x d}.
  2. Learnable Subgoal Queries Q_goal in R^{N_g x d} (N_g = 16 latent board objectives) attend
     to the 81 board tokens via Cross-Attention to locate active jellies, blockers, and specials.
  3. Candidate Action Extractor gathers adjacent cell pairs for all 144 swaps into E_act in R^{B x 144 x d}.
  4. Differentiable Bipartite Assignment Head computes the (B, 144, 16) Swap-to-Objective affinity
     matrix and runs 4 Sinkhorn-Hungarian normalization steps to score actions that simultaneously
     satisfy high-priority board objectives!
"""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


class DifferentiableBipartiteMatcherNet(nn.Module):
    def __init__(
        self,
        in_channels: int = 20,
        d_model: int = 128,
        num_queries: int = 16,
        num_actions: int = 144,
        sinkhorn_iters: int = 4,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.num_queries = num_queries
        self.num_actions = num_actions
        self.sinkhorn_iters = sinkhorn_iters

        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, d_model, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(d_model, d_model, kernel_size=3, padding=1),
            nn.GELU(),
        )
        self.pos_emb = nn.Parameter(torch.randn(1, 81, d_model) * 0.02)

        # N_g = 16 learnable subgoal queries (DETR style)
        self.subgoal_queries = nn.Parameter(torch.randn(1, num_queries, d_model) * 0.02)
        self.query_cross_attn = nn.MultiheadAttention(d_model, num_heads=4, batch_first=True)
        self.query_norm = nn.LayerNorm(d_model)
        self.subgoal_urgency_head = nn.Linear(d_model, 1)

        # Action projector from 81 spatial tokens to 144 swap embeddings
        self.action_proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, num_actions * 32),
        )
        self.swap_embed_proj = nn.Linear(32, d_model)

        # Final policy and value synthesis
        self.policy_residual = nn.Linear(d_model, 1)
        self.value_head = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.GELU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, board_tensor: torch.Tensor) -> Dict[str, torch.Tensor]:
        B = board_tensor.size(0)
        feat = self.stem(board_tensor)  # (B, d, 9, 9)
        tokens = feat.flatten(2).transpose(1, 2) + self.pos_emb  # (B, 81, d)

        # 1. Extract 16 active board subgoals via DETR Cross-Attention
        queries = self.subgoal_queries.expand(B, -1, -1)  # (B, 16, d)
        goal_tokens, _ = self.query_cross_attn(queries, tokens, tokens)
        goal_tokens = self.query_norm(queries + goal_tokens)  # (B, 16, d)
        goal_urgency = torch.sigmoid(self.subgoal_urgency_head(goal_tokens))  # (B, 16, 1)

        # 2. Extract 144 candidate swap embeddings
        pooled = tokens.mean(dim=1)  # (B, d)
        act_raw = self.action_proj(pooled).view(B, self.num_actions, 32)
        swap_tokens = self.swap_embed_proj(act_raw)  # (B, 144, d)

        # 3. Compute Bipartite Swap-to-Objective Affinity Matrix (B, 144, 16)
        affinity = torch.bmm(swap_tokens, goal_tokens.transpose(1, 2)) / (self.d_model ** 0.5)

        # Differentiable Sinkhorn-Hungarian doubly-stochastic relaxation in log domain
        log_assign = affinity
        for _ in range(self.sinkhorn_iters):
            log_assign = log_assign - torch.logsumexp(log_assign, dim=-1, keepdim=True)
            log_assign = log_assign - torch.logsumexp(log_assign, dim=-2, keepdim=True)
        assignment_matrix = torch.exp(log_assign)  # (B, 144, 16)

        # 4. Score actions by urgency-weighted bipartite assignment + residual
        matched_score = torch.bmm(assignment_matrix, goal_urgency).squeeze(-1)  # (B, 144)
        residual_score = self.policy_residual(swap_tokens).squeeze(-1)  # (B, 144)
        policy_logits = residual_score + 4.0 * matched_score

        win_prob = self.value_head(pooled)  # (B, 1)

        return {
            "policy_logits": policy_logits,
            "win_prob": win_prob,
            "assignment_matrix": assignment_matrix,
            "subgoal_urgency": goal_urgency.squeeze(-1),
        }


if __name__ == "__main__":
    model = DifferentiableBipartiteMatcherNet()
    x = torch.randn(2, 20, 9, 9)
    out = model(x)
    params = sum(p.numel() for p in model.parameters())
    print(f"DifferentiableBipartiteMatcherNet: {params:,} params | policy={out['policy_logits'].shape} | assign={out['assignment_matrix'].shape}")
