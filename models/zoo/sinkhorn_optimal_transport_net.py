"""
Sinkhorn Optimal Transport & Earth Mover's Distance Network (`OptimalTransportCandyNet`).
Based on:
  - Cuturi (NeurIPS 2013), "Sinkhorn Distances: Lightspeed Computation of Optimal Transport"
  - Peyré & Cuturi (Foundations and Trends in Machine Learning 2019), "Computational Optimal Transport"
  - Sarlin et al. (CVPR 2020), "SuperGlue: Learning Feature Matching with Graph Neural Networks"

Formulates Candy Crush clearing as an Entropic Optimal Transport problem between:
  - Source Force Distribution mu in Delta^{81} (Special candies, blast rays, matchable color clusters)
  - Target Sink Distribution  nu in Delta^{81} (Jellies, Double Jellies, Chocolate, Frosting, Blockers)
Computes a state-conditioned ground cost matrix C in R^{B x 81 x 81} and runs 5 iterations of
differentiable log-domain Sinkhorn-Knopp normalization to obtain the Optimal Transport Plan
P* in R^{B x 81 x 81} and Wasserstein distance W_eps(mu, nu), routing destructive energy to
objectives with minimal wasted moves.
"""

import math
from typing import Dict, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class LogDomainSinkhornSolver(nn.Module):
    """
    Differentiable Log-Domain Sinkhorn-Knopp Optimal Transport Solver (Cuturi, 2013).
    Solves:
        P* = argmin_{P in U(mu, nu)} <P, C> - eps * H(P)
    via stabilized log-sum-exp dual updates over `num_iterations` steps (default: 5).
    """
    def __init__(self, num_iterations: int = 5, epsilon: float = 0.15):
        super().__init__()
        self.num_iterations = num_iterations
        self.log_epsilon = nn.Parameter(torch.tensor(math.log(epsilon)))

    @property
    def epsilon(self) -> torch.Tensor:
        return torch.exp(self.log_epsilon).clamp(min=0.03, max=2.0)

    def forward(
        self,
        cost_matrix: torch.Tensor,
        source_mu: torch.Tensor,
        target_nu: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            cost_matrix: Ground transport cost C in R^{B x 81 x 81} (non-negative)
            source_mu: Normalized Source Force marginal mu in Delta^{B x 81}
            target_nu: Normalized Target Sink marginal nu in Delta^{B x 81}
        Returns:
            transport_plan: Optimal coupling matrix P* in R^{B x 81 x 81}
            wasserstein_dist: Expected transport cost <P*, C> in R^{B x 1}
            dual_u: Log-scaling source potential u in R^{B x 81}
            dual_v: Log-scaling target potential v in R^{B x 81}
        """
        eps = self.epsilon
        log_mu = torch.log(source_mu.clamp(min=1e-6))  # (B, 81)
        log_nu = torch.log(target_nu.clamp(min=1e-6))  # (B, 81)

        # Gibbs kernel in log-domain: K_log = -C / eps of shape (B, 81, 81)
        log_k = -cost_matrix / eps

        u = torch.zeros_like(log_mu)
        v = torch.zeros_like(log_nu)

        # 5 iterations of stabilized log-domain Sinkhorn-Knopp updates
        for _ in range(self.num_iterations):
            u = log_mu - torch.logsumexp(log_k + v.unsqueeze(1), dim=2)
            v = log_nu - torch.logsumexp(log_k + u.unsqueeze(2), dim=1)

        # Recover doubly-stochastic transport plan P* = exp(u_i + v_j - C_{ij}/eps)
        log_plan = u.unsqueeze(2) + v.unsqueeze(1) + log_k
        transport_plan = torch.exp(log_plan)  # (B, 81, 81)

        # Earth Mover's (Wasserstein) Distance W_eps(mu, nu) = sum_{i,j} P*_{i,j} * C_{i,j}
        wasserstein_dist = (transport_plan * cost_matrix).sum(dim=(1, 2), keepdim=False).unsqueeze(-1)
        return transport_plan, wasserstein_dist, u, v


class OptimalTransportCandyNet(nn.Module):
    """
    Optimal Transport & Earth Mover's Distance Architecture for Candy Crush Saga.
    Routes destructive force from Special Candies & color clusters (Source Marginal mu)
    to Jellies & Blockers (Target Marginal nu) along minimal-cost blast trajectories.
    """
    def __init__(
        self,
        in_channels: int = 20,
        d_model: int = 256,
        num_res_blocks: int = 4,
        sinkhorn_iterations: int = 5,
        sinkhorn_epsilon: float = 0.15,
        num_actions: int = 144,
        board_h: int = 9,
        board_w: int = 9
    ):
        super().__init__()
        self.in_channels = in_channels
        self.d_model = d_model
        self.sinkhorn_iterations = sinkhorn_iterations
        self.num_actions = num_actions
        self.board_h = board_h
        self.board_w = board_w
        self.num_tokens = board_h * board_w

        # Precompute static 2D Manhattan distance + orthogonal ray geometry on 9x9 grid
        coords = torch.stack(
            torch.meshgrid(
                torch.arange(board_h, dtype=torch.float32),
                torch.arange(board_w, dtype=torch.float32),
                indexing="ij"
            ),
            dim=-1
        ).view(-1, 2)  # (81, 2)

        r_dist = (coords[:, 0].unsqueeze(1) - coords[:, 0].unsqueeze(0)).abs()  # (81, 81)
        c_dist = (coords[:, 1].unsqueeze(1) - coords[:, 1].unsqueeze(0)).abs()  # (81, 81)
        manhattan_dist = (r_dist + c_dist) / float(board_h + board_w)           # normalized [0, 1]
        same_row = (r_dist < 1e-4).float()
        same_col = (c_dist < 1e-4).float()
        blast_3x3 = ((r_dist <= 1.0) & (c_dist <= 1.0)).float()

        self.register_buffer("manhattan_dist", manhattan_dist.unsqueeze(0))
        self.register_buffer("same_row", same_row.unsqueeze(0))
        self.register_buffer("same_col", same_col.unsqueeze(0))
        self.register_buffer("blast_3x3", blast_3x3.unsqueeze(0))

        # Precompute swap endpoint indices (0..80) for all 144 canonical swap actions
        cell1_idx, cell2_idx = [], []
        for r in range(board_h):
            for c in range(board_w - 1):
                cell1_idx.append(r * board_w + c)
                cell2_idx.append(r * board_w + (c + 1))
        for r in range(board_h - 1):
            for c in range(board_w):
                cell1_idx.append(r * board_w + c)
                cell2_idx.append((r + 1) * board_w + c)
        self.register_buffer("swap_cell1", torch.tensor(cell1_idx, dtype=torch.long))
        self.register_buffer("swap_cell2", torch.tensor(cell2_idx, dtype=torch.long))

        # Spatial Feature Backbone
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, d_model, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(d_model),
            nn.GELU()
        )
        self.res_blocks = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(d_model, d_model, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(d_model),
                nn.GELU(),
                nn.Conv2d(d_model, d_model, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(d_model)
            )
            for _ in range(num_res_blocks)
        ])

        # Source Force Marginal Estimator mu(s) in Delta^{81} & Target Sink Marginal Estimator nu(s) in Delta^{81}
        self.source_force_head = nn.Sequential(
            nn.Conv2d(d_model + in_channels, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.Conv2d(64, 1, kernel_size=1)
        )
        self.target_sink_head = nn.Sequential(
            nn.Conv2d(d_model + in_channels, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.Conv2d(64, 1, kernel_size=1)
        )

        # Learned Pairwise Cost Embedding Projections for C(s) in R^{B x 81 x 81}
        self.cost_source_proj = nn.Linear(d_model, 64)
        self.cost_target_proj = nn.Linear(d_model, 64)

        # Differentiable 5-Step Log-Domain Sinkhorn Solver
        self.sinkhorn_solver = LogDomainSinkhornSolver(
            num_iterations=sinkhorn_iterations,
            epsilon=sinkhorn_epsilon
        )

        # Transport-Guided Policy Head combining spatial features, Kantorovich potentials (u, v),
        # and swap-induced transport flux P*_{cell1 <-> cell2}
        self.transport_token_fuse = nn.Sequential(
            nn.Linear(d_model + 4, d_model),
            nn.LayerNorm(d_model),
            nn.GELU()
        )
        self.policy_head = nn.Sequential(
            nn.Conv2d(d_model, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.Conv2d(128, 64, kernel_size=1),
            nn.GELU(),
            nn.Flatten(),
            nn.Linear(64 * self.num_tokens, 256),
            nn.GELU(),
            nn.Linear(256, num_actions)
        )
        self.swap_flux_scorer = nn.Linear(6, 1)

        # Value / Win Probability Head conditioned on Wasserstein distance W_eps(mu, nu)
        self.value_head = nn.Sequential(
            nn.Linear(d_model + 1, 128),
            nn.GELU(),
            nn.Linear(128, 1)
        )

    def compute_marginals_and_cost(
        self,
        board_tensor: torch.Tensor,
        spatial_feat: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Computes:
          - `source_mu` in Delta^{B x 81}: Destructive Force density (specials + clusters)
          - `target_nu` in Delta^{B x 81}: Objective Sink demand (jellies + blockers)
          - `cost_matrix` in R_+^{B x 81 x 81}: Ground transport cost between every cell pair
        """
        B, C, H, W = board_tensor.shape
        joint_in = torch.cat([spatial_feat, board_tensor], dim=1)

        # Domain prior for Source Force (Special candies channels 8..11)
        special_prior = board_tensor[:, 8:12].clamp(0.0, 1.0).sum(dim=1).flatten(1)  # (B, 81)
        source_logits = self.source_force_head(joint_in).flatten(1) + 1.5 * special_prior
        source_mu = F.softmax(source_logits, dim=-1)  # (B, 81)

        # Domain prior for Target Sink (Jellies channels 13..14 + Blockers channels 15..17)
        objective_prior = board_tensor[:, 13:18].clamp(0.0, 1.0).sum(dim=1).flatten(1)  # (B, 81)
        target_logits = self.target_sink_head(joint_in).flatten(1) + 1.5 * objective_prior
        target_nu = F.softmax(target_logits, dim=-1)  # (B, 81)

        # Construct state-dependent 81x81 Cost Matrix C_{i,j}
        tokens = spatial_feat.flatten(2).transpose(1, 2)  # (B, 81, d_model)
        src_emb = F.normalize(self.cost_source_proj(tokens), dim=-1)  # (B, 81, 64)
        tgt_emb = F.normalize(self.cost_target_proj(tokens), dim=-1)  # (B, 81, 64)

        # Learned semantic dissimilarity in [0, 2]
        learned_cost = 1.0 - torch.bmm(src_emb, tgt_emb.transpose(1, 2))  # (B, 81, 81)

        # Blast-corridor cost discounts when source cell i holds a Striped/Wrapped/ColorBomb candy
        striped_h_i = board_tensor[:, 8].clamp(0.0, 1.0).flatten(1).unsqueeze(2)   # (B, 81, 1)
        striped_v_i = board_tensor[:, 9].clamp(0.0, 1.0).flatten(1).unsqueeze(2)   # (B, 81, 1)
        wrapped_i = board_tensor[:, 10].clamp(0.0, 1.0).flatten(1).unsqueeze(2)    # (B, 81, 1)

        blast_discount = (
            0.5 * striped_h_i * self.same_row +
            0.5 * striped_v_i * self.same_col +
            0.5 * wrapped_i * self.blast_3x3
        )
        cost_matrix = (self.manhattan_dist + 0.5 * learned_cost - blast_discount).clamp(min=0.01, max=3.0)
        return source_mu, target_nu, cost_matrix

    def forward(
        self,
        board_tensor: torch.Tensor,
        legal_mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Canonical forward pass accepting (B, 20, 9, 9) board tensor.
        Runs 5 iterations of differentiable log-domain Sinkhorn-Knopp optimal transport
        to align destructive force with board objectives.
        """
        B, C, H, W = board_tensor.shape

        # 1. Extract deep spatial features
        h = self.stem(board_tensor)
        for block in self.res_blocks:
            h = F.gelu(h + block(h))

        # 2. Compute Source Force mu, Target Sink nu, and Ground Cost C in R^{B x 81 x 81}
        source_mu, target_nu, cost_matrix = self.compute_marginals_and_cost(board_tensor, h)

        # 3. Run 5-iteration Log-Domain Sinkhorn-Knopp Optimal Transport
        transport_plan, wasserstein_dist, dual_u, dual_v = self.sinkhorn_solver(
            cost_matrix, source_mu, target_nu
        )

        # 4. Fuse Optimal Transport potentials and marginal flows into cell representations
        # Outgoing destructive flow from cell i to top target sinks, and incoming flow to cell j
        tokens = h.flatten(2).transpose(1, 2)  # (B, 81, d_model)
        ot_cell_features = torch.stack([
            source_mu,
            target_nu,
            torch.tanh(dual_u * 0.1),
            torch.tanh(dual_v * 0.1)
        ], dim=-1)  # (B, 81, 4)

        fused_tokens = self.transport_token_fuse(torch.cat([tokens, ot_cell_features], dim=-1))
        fused_2d = fused_tokens.transpose(1, 2).contiguous().view(B, self.d_model, H, W)

        # 5. Compute Swap Policy Logits combining spatial policy head + direct OT swap flux
        base_logits = self.policy_head(fused_2d)  # (B, 144)

        # Extract pairwise OT flux across each of the 144 swap edges (cell1 <-> cell2)
        c1 = self.swap_cell1
        c2 = self.swap_cell2
        flux_12 = transport_plan[:, c1, c2]
        flux_21 = transport_plan[:, c2, c1]
        mu_1, mu_2 = source_mu[:, c1], source_mu[:, c2]
        nu_1, nu_2 = target_nu[:, c1], target_nu[:, c2]
        edge_ot_feats = torch.stack([flux_12, flux_21, mu_1, mu_2, nu_1, nu_2], dim=-1)  # (B, 144, 6)
        flux_bonus = self.swap_flux_scorer(edge_ot_feats).squeeze(-1)  # (B, 144)

        policy_logits = base_logits + flux_bonus
        if legal_mask is not None:
            policy_logits = policy_logits.masked_fill(~legal_mask.bool(), -1e9)

        # 6. Compute Win Probability & Value conditioned on Earth Mover's Distance W_eps(mu, nu)
        # Lower Wasserstein distance between destructive force and remaining jellies -> higher win prob!
        z_global = fused_tokens.mean(dim=1)  # (B, d_model)
        val_raw = self.value_head(torch.cat([z_global, -wasserstein_dist], dim=-1))  # (B, 1)
        win_prob = torch.sigmoid(val_raw)
        value = torch.tanh(val_raw)

        return {
            "policy_logits": policy_logits,
            "win_prob": win_prob,
            "value": value,
            "transport_plan": transport_plan,
            "cost_matrix": cost_matrix,
            "source_marginal": source_mu,
            "target_marginal": target_nu,
            "wasserstein_distance": wasserstein_dist
        }

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def create_optimal_transport_candy_net(device: str = "cpu") -> OptimalTransportCandyNet:
    """Factory function initializing OptimalTransportCandyNet on the target device."""
    model = OptimalTransportCandyNet(
        in_channels=20,
        d_model=256,
        num_res_blocks=4,
        sinkhorn_iterations=5,
        sinkhorn_epsilon=0.15,
        num_actions=144
    )
    return model.to(device)


if __name__ == "__main__":
    torch.manual_seed(42)
    net = create_optimal_transport_candy_net("cpu")
    params = net.count_parameters()
    print(f"[OptimalTransportCandyNet] Initialized with {params:,} trainable parameters ({params / 1e6:.2f}M)")

    dummy_board = torch.randn(2, 20, 9, 9).abs()
    out = net(dummy_board)
    print(f"  policy_logits:        {out['policy_logits'].shape}")
    print(f"  win_prob:             {out['win_prob'].shape}")
    print(f"  value:                {out['value'].shape}")
    print(f"  transport_plan:       {out['transport_plan'].shape}")
    print(f"  cost_matrix:          {out['cost_matrix'].shape}")
    print(f"  wasserstein_distance: {out['wasserstein_distance'].shape}")
