"""
AlphaCandy Model Zoo — Part 2.3: Generative Flow Network (`CandyGFlowNet`).
===========================================================================
Generative Flow Network (GFlowNet; Bengio et al., 2021/2023) with Trajectory Balance (TB)
and Detailed Balance (DB) parameterization for combinatorial multi-attempt puzzle solving.

Why GFlowNets for Candy Crush?
------------------------------
Standard RL (PPO / DQN) is mode-seeking: it collapses onto a single greedy trajectory argmax_a Q(s, a).
On ultra-hard King levels (e.g., Level 276, 147, 890), a single deterministic strategy can get trapped
in a local optimum, whereas multi-attempt retry hunting requires sampling a *diverse portfolio* of
high-reward trajectories strictly proportional to their terminal win reward:
    P_θ(τ) ∝ R(τ)

Architecture & Heads:
---------------------
  1. Forward Edge Flow Head  : log F_θ(s, a) ∈ R^(B, 144) and Forward Policy log P_F(a | s)
  2. State Partition Flow    : log Z_θ(s) (or log F_θ(s)) ∈ R^(B, 1)
  3. Backward Transition Head: log P_B(a | s') ∈ R^(B, 144) for DAG credit assignment across transpositions
  4. `sample_diverse_actions(state_tensor, legal_mask, num_samples=8, temperature=1.0)`:
     Samples `num_samples` diverse, high-flow swaps with spatial coordinate repulsion so retry
     attempts explore orthogonal regions and tactical cascades on the board.
"""

from typing import Dict, List, Optional, Tuple
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class FlowResidualBlock(nn.Module):
    """Residual block with GroupNorm & SiLU for stable log-flow magnitude estimation."""

    def __init__(self, channels: int = 160):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.gn1 = nn.GroupNorm(8, channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.gn2 = nn.GroupNorm(8, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.silu(self.gn1(self.conv1(x)))
        out = self.gn2(self.conv2(out))
        return F.silu(x + out)


class CandyGFlowNet(nn.Module):
    """
    Generative Flow Network (Bengio et al., 2023) parameterized for both
    Trajectory Balance (TB) and Detailed Balance (DB) over the 144 canonical Candy Crush swaps.

    Inputs:
      - board_tensor: (B, 20, 9, 9) canonical 20-channel board state
      - legal_mask: Optional (B, 144) boolean mask of legal moves

    Outputs (Dict):
      - policy_logits: (B, 144) forward flow logits log F_θ(s, a) (masked if legal_mask given)
      - value: (B, 1) tanh-squashed normalized state flow in [-1, 1]
      - win_prob: (B, 1) calibrated win probability in [0, 1]
      - log_F_sa: (B, 144) unnormalized log edge flows log F_θ(s, a)
      - log_Z_s: (B, 1) log state partition flow log F_θ(s) / log Z_θ(s)
      - log_P_F: (B, 144) normalized forward transition log-probabilities log P_F(a | s)
      - log_P_B: (B, 144) normalized backward transition log-probabilities log P_B(a | s')
      - action_embeddings: (B, 144, 32) geometric + semantic action vectors for orthogonal sampling
    """

    def __init__(
        self,
        in_channels: int = 20,
        d_model: int = 160,
        num_blocks: int = 5,
        num_actions: int = 144,
        board_h: int = 9,
        board_w: int = 9
    ):
        super().__init__()
        self.in_channels = in_channels
        self.d_model = d_model
        self.num_actions = num_actions
        self.board_h = board_h
        self.board_w = board_w

        # Global partition function log Z_0 for root initial states (Trajectory Balance)
        self.log_Z0 = nn.Parameter(torch.tensor(2.5, dtype=torch.float32))

        # Perceptual Spatial Stem + Flow Residual Tower
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, d_model, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, d_model),
            nn.SiLU()
        )
        self.tower = nn.Sequential(*[
            FlowResidualBlock(d_model) for _ in range(num_blocks)
        ])

        # 1. Forward Flow Head: predicts log F_θ(s, a) ∈ R^(B, 144)
        self.forward_flow_conv = nn.Sequential(
            nn.Conv2d(d_model, 96, kernel_size=3, padding=1),
            nn.GroupNorm(8, 96),
            nn.SiLU(),
            nn.Conv2d(96, 48, kernel_size=1),
            nn.SiLU()
        )
        self.forward_flow_fc = nn.Sequential(
            nn.Linear(48 * board_h * board_w, 384),
            nn.SiLU(),
            nn.Linear(384, num_actions)
        )

        # 2. State Flow Head: predicts log Z_θ(s) = log F_θ(s) ∈ R^(B, 1)
        self.state_flow_head = nn.Sequential(
            nn.Linear(d_model * 2, 128),
            nn.SiLU(),
            nn.Linear(128, 64),
            nn.SiLU(),
            nn.Linear(64, 1)
        )

        # 3. Backward Flow Head: predicts backward policy logits for P_B(a | s') ∈ R^(B, 144)
        self.backward_flow_conv = nn.Sequential(
            nn.Conv2d(d_model, 64, kernel_size=1),
            nn.SiLU()
        )
        self.backward_flow_fc = nn.Sequential(
            nn.Linear(64 * board_h * board_w, 256),
            nn.SiLU(),
            nn.Linear(256, num_actions)
        )

        # 4. Action Embedding Projection for Determinantal / Orthogonal Diversity Sampling
        self.action_embed_proj = nn.Linear(48 * board_h * board_w, num_actions * 16)
        self.register_buffer("action_coords", self._build_canonical_action_coords(board_h, board_w))

    @staticmethod
    def _build_canonical_action_coords(height: int = 9, width: int = 9) -> torch.Tensor:
        """
        Precomputes normalized (r_center, c_center, is_vertical) coordinates for all 144 canonical actions.
        Shape: (144, 3)
        """
        coords = []
        # 0..71: Horizontal swaps (r, c) <-> (r, c+1)
        for idx in range(72):
            r = idx // (width - 1)
            c = idx % (width - 1)
            coords.append([r / 8.0, (c + 0.5) / 8.0, 0.0])
        # 72..143: Vertical swaps (r, c) <-> (r+1, c)
        for v_idx in range(72):
            r = v_idx // width
            c = v_idx % width
            coords.append([(r + 0.5) / 8.0, c / 8.0, 1.0])
        return torch.tensor(coords, dtype=torch.float32)

    def forward(
        self,
        board_tensor: torch.Tensor,
        legal_mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        B, C, H, W = board_tensor.shape
        feat = self.tower(self.stem(board_tensor))  # (B, d_model, 9, 9)

        # Global pooled representation for State Flow log Z_θ(s)
        g_mean = feat.mean(dim=(2, 3))
        g_max = feat.flatten(2).max(dim=-1).values
        g_pool = torch.cat([g_mean, g_max], dim=-1)  # (B, 2*d_model)
        log_Z_s = self.state_flow_head(g_pool) + self.log_Z0  # (B, 1)

        # Forward Edge Flow log F_θ(s, a)
        f_spatial = self.forward_flow_conv(feat).flatten(1)  # (B, 48*81)
        raw_edge_logits = self.forward_flow_fc(f_spatial)    # (B, 144)

        if legal_mask is not None:
            masked_edge_logits = raw_edge_logits.masked_fill(~legal_mask.bool(), -1e9)
        else:
            masked_edge_logits = raw_edge_logits

        # Normalized Forward Policy log P_F(a | s)
        log_P_F = F.log_softmax(masked_edge_logits, dim=-1)
        # Edge flow satisfies F_θ(s, a) = Z_θ(s) * P_F(a | s)  =>  log F(s,a) = log Z(s) + log P_F(a|s)
        log_F_sa = log_Z_s + log_P_F

        # Backward Flow log P_B(a | s')
        b_spatial = self.backward_flow_conv(feat).flatten(1)
        b_logits = self.backward_flow_fc(b_spatial)
        log_P_B = F.log_softmax(b_logits, dim=-1)

        # Action semantic + geometric embeddings (B, 144, 19) for orthogonal trajectory sampling
        sem_emb = self.action_embed_proj(f_spatial).view(B, self.num_actions, 16)
        sem_emb = F.normalize(sem_emb, p=2, dim=-1)
        geo_emb = self.action_coords.unsqueeze(0).expand(B, -1, -1)
        action_embeddings = torch.cat([sem_emb, geo_emb], dim=-1)  # (B, 144, 19)

        # Calibrated Value [-1, 1] and Win Probability [0, 1] derived from State Flow
        value = torch.tanh(0.25 * log_Z_s)
        win_prob = torch.sigmoid(0.5 * (log_Z_s - self.log_Z0))

        return {
            "policy_logits": masked_edge_logits,
            "value": value,
            "win_prob": win_prob,
            "log_F_sa": log_F_sa,
            "log_Z_s": log_Z_s,
            "log_P_F": log_P_F,
            "log_P_B": log_P_B,
            "action_embeddings": action_embeddings,
        }

    def sample_diverse_actions(
        self,
        state_tensor: torch.Tensor,
        legal_mask: Optional[torch.Tensor] = None,
        num_samples: int = 8,
        temperature: float = 1.0,
        repulsion_strength: float = 1.25
    ) -> Dict[str, torch.Tensor]:
        """
        Samples `num_samples` diverse, high-reward orthogonal actions proportional to flow
        F_θ(s, a)^(1/T) while applying iterative Determinantal / Cosine-Spatial Repulsion
        so multi-attempt retry hunting explores distinct board regions instead of collapsing
        onto redundant neighboring swaps.

        Args:
            state_tensor: (B, 20, 9, 9) or (20, 9, 9) board state tensor
            legal_mask: Optional (B, 144) or (144,) boolean legal move mask
            num_samples: Number of diverse actions to sample per state (default 8)
            temperature: Flow temperature T > 0
            repulsion_strength: Penalty weight against spatial/semantic overlap with already sampled moves

        Returns:
            Dict containing:
              - sampled_actions: (B, num_samples) action indices in [0, 143]
              - sampled_log_flows: (B, num_samples) log F_θ(s, a) of each sampled action
              - sampled_probs: (B, num_samples) forward probabilities P_F(a | s)
              - spatial_diversity_score: (B,) mean pairwise Euclidean distance between sampled swaps
        """
        if state_tensor.dim() == 3:
            state_tensor = state_tensor.unsqueeze(0)
        if legal_mask is not None and legal_mask.dim() == 1:
            legal_mask = legal_mask.unsqueeze(0)

        B = state_tensor.shape[0]
        temp = max(1e-3, float(temperature))

        with torch.no_grad():
            out = self.forward(state_tensor, legal_mask=legal_mask)
            base_logits = out["policy_logits"] / temp          # (B, 144)
            log_F_sa = out["log_F_sa"]                         # (B, 144)
            base_probs = F.softmax(out["policy_logits"], dim=-1)
            act_emb = F.normalize(out["action_embeddings"], p=2, dim=-1)  # (B, 144, 19)

            sampled_indices = []
            sampled_flows = []
            sampled_p = []

            # Working copy of logits updated with orthogonal repulsion after each draw
            running_logits = base_logits.clone()
            if legal_mask is not None:
                num_legal = legal_mask.sum(dim=-1)  # (B,)
            else:
                num_legal = torch.full((B,), self.num_actions, device=state_tensor.device)

            for k in range(num_samples):
                probs_k = F.softmax(running_logits, dim=-1)
                # Guard against numerical underflow if all legal moves were already sampled
                probs_k = torch.nan_to_num(probs_k, nan=1.0 / self.num_actions).clamp(min=1e-8)
                if legal_mask is not None:
                    probs_k = probs_k * legal_mask.float()
                    row_sums = probs_k.sum(dim=-1, keepdim=True).clamp(min=1e-8)
                    probs_k = probs_k / row_sums

                drawn_a = torch.multinomial(probs_k, num_samples=1).squeeze(-1)  # (B,)
                sampled_indices.append(drawn_a)
                sampled_flows.append(log_F_sa.gather(1, drawn_a.unsqueeze(1)).squeeze(1))
                sampled_p.append(base_probs.gather(1, drawn_a.unsqueeze(1)).squeeze(1))

                # Compute cosine + spatial similarity between `drawn_a` and all 144 actions
                chosen_emb = act_emb.gather(
                    1, drawn_a.view(B, 1, 1).expand(B, 1, act_emb.size(-1))
                )  # (B, 1, 19)
                similarity = torch.bmm(act_emb, chosen_emb.transpose(1, 2)).squeeze(-1)  # (B, 144)

                # Repel similar actions and suppress exact repeat if more legal moves exist
                running_logits = running_logits - repulsion_strength * F.relu(similarity)
                can_suppress = (k + 1) < num_legal
                for b_i in range(B):
                    if can_suppress[b_i]:
                        running_logits[b_i, drawn_a[b_i]] -= 20.0

            sampled_actions = torch.stack(sampled_indices, dim=1)     # (B, K)
            sampled_log_flows = torch.stack(sampled_flows, dim=1)     # (B, K)
            sampled_probs = torch.stack(sampled_p, dim=1)             # (B, K)

            # Compute pairwise spatial distance on the 9x9 grid to verify diversity
            coords_2d = self.action_coords[:, :2]  # (144, 2)
            chosen_coords = coords_2d[sampled_actions]  # (B, K, 2)
            dist_matrix = torch.cdist(chosen_coords, chosen_coords, p=2)  # (B, K, K)
            diversity_score = dist_matrix.mean(dim=(1, 2))

            return {
                "sampled_actions": sampled_actions,
                "sampled_log_flows": sampled_log_flows,
                "sampled_probs": sampled_probs,
                "spatial_diversity_score": diversity_score,
            }

    def compute_detailed_balance_loss(
        self,
        s_t: torch.Tensor,
        a_t: torch.Tensor,
        s_next: torch.Tensor,
        log_reward_next: torch.Tensor,
        is_terminal: torch.Tensor
    ) -> torch.Tensor:
        """
        Detailed Balance (DB) Loss for transition tuple (s_t, a_t, s_{t+1}):
            L_DB(s, a, s') = ( log Z_θ(s) + log P_F(a | s) - log Z_θ(s') - log P_B(a | s') )^2
        where for terminal states s', log Z_θ(s') is replaced by log R(s').
        """
        out_s = self.forward(s_t)
        out_next = self.forward(s_next)

        log_Z_s = out_s["log_Z_s"].squeeze(-1)
        log_pf = out_s["log_P_F"].gather(1, a_t.long().unsqueeze(1)).squeeze(-1)
        log_pb = out_next["log_P_B"].gather(1, a_t.long().unsqueeze(1)).squeeze(-1)

        target_log_Z_next = torch.where(
            is_terminal.bool(),
            log_reward_next,
            out_next["log_Z_s"].squeeze(-1)
        )
        discrepancy = (log_Z_s + log_pf) - (target_log_Z_next + log_pb)
        return (discrepancy ** 2).mean()

    def compute_trajectory_balance_loss(
        self,
        sum_log_pf: torch.Tensor,
        sum_log_pb: torch.Tensor,
        log_terminal_reward: torch.Tensor
    ) -> torch.Tensor:
        """
        Trajectory Balance (TB) Loss (Malkin et al., 2022):
            L_TB(τ) = ( log Z_0 + Σ_{t} log P_F(s_{t+1} | s_t) - log R(x) - Σ_{t} log P_B(s_t | s_{t+1}) )^2
        """
        lhs = self.log_Z0 + sum_log_pf
        rhs = log_terminal_reward + sum_log_pb
        return ((lhs - rhs) ** 2).mean()

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


if __name__ == "__main__":
    torch.manual_seed(42)
    gfn = CandyGFlowNet()
    print(f"[CandyGFlowNet] Trainable Parameters: {gfn.count_parameters():,}")
    dummy_board = torch.randn(2, 20, 9, 9)
    dummy_mask = torch.ones(2, 144, dtype=torch.bool)
    out = gfn(dummy_board, legal_mask=dummy_mask)
    samples = gfn.sample_diverse_actions(dummy_board, legal_mask=dummy_mask, num_samples=8, temperature=1.0)
    print("  policy_logits:", out["policy_logits"].shape)
    print("  log_F_sa:", out["log_F_sa"].shape)
    print("  log_Z_s:", out["log_Z_s"].shape)
    print("  sampled_actions (8 diverse):", samples["sampled_actions"].shape, samples["sampled_actions"][0].tolist())
    print("  spatial_diversity_score:", samples["spatial_diversity_score"].tolist())
