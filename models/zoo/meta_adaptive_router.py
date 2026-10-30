"""
AlphaCandy Model Zoo — Part 2.5: Meta-Adaptive Mixture-of-Experts Router (`GrandmasterMoERouter`).
==================================================================================================
Level-Topology Mixture-of-Experts (MoE) Router that extracts a 16-dimensional topological feature
vector from any `(B, 20, 9, 9)` board state and dynamically computes gating weights across all
specialist architectures in the AlphaCandy Model Zoo.

16 Extracted Topological Features `φ_topo(s) ∈ R^(B, 16)`:
----------------------------------------------------------
  0. `jelly_ratio`             : Total single + double jelly coverage over active grid
  1. `double_jelly_ratio`      : Thick 2-layer jelly concentration (channel 16)
  2. `blocker_density`         : Overall blocker density (channels 10..14)
  3. `chocolate_presence`      : Expanding chocolate / spawner hazard intensity (channel 10 & 13)
  4. `licorice_frosting_ratio` : Licorice & frosting barrier density (channels 11 & 12)
  5. `special_candy_count`     : Total Striped H/V, Wrapped, and Color Bomb count (channels 6..9)
  6. `color_bomb_presence`     : Color Bomb availability on board (channel 9)
  7. `corner_objective_ratio`  : Fraction of jellies/blockers located in corners & outer rim
  8. `moves_urgency`           : Urgency pressure (1.0 - normalized moves remaining)
  9. `island_fragmentation`    : Non-Euclidean hole/void boundary perimeter ratio (channel 18)
 10. `ingredient_portal_load`  : Ingredient & teleporter channel activity (channels 16, 17)
 11. `top_choke_density`       : Blocker congestion in top 3 dispenser rows (rows 0..2)
 12. `bottom_cascade_potential`: Active playable candy ratio in bottom 3 rows (rows 6..8)
 13. `color_entropy`           : Normalized Shannon entropy over the 6 candy colors (channels 0..5)
 14. `special_combo_adjacency` : Spatially adjacent Special-to-Special pair indicator
 15. `desperation_pressure`    : Remaining objective workload relative to remaining move budget

Specialist Zoo Experts Routed (Default 10 Zoo Specialists):
-----------------------------------------------------------
  0: `AlphaCandyOmegaNet`            (32.8M Unified Graph-Spatial Quantile Super Model)
  1: `EquivariantSymmetricCandyNet`  (D4 Dihedral & C6 Color-Permutation Equivariant Net)
  2: `FactoredMuZeroWorldModel`      (Latent Physics Cascade & Spawner Separation World Model)
  3: `DiscreteDiffusionPlanner`      (Denoising Diffusion Trajectory Lookahead Planner)
  4: `TestTimeTTTAdapter`            (Self-Supervised Test-Time Training LoRA Adapter)
  5: `WormholeGraphNeuralNet`        (4-Relation Heterogeneous Directed Multigraph MPNN)
  6: `HierarchicalSubgoalCandyNet`   (Feudal Manager-Worker 6-Option Critic)
  7: `CandyGFlowNet`                 (Trajectory-Balance Combinatorial Diversity Sampler)
  8: `TranspositionMCGSSolver`       (64-bit Zobrist DAG Search with Dynamic CVaR Tilting)
  9: `FastTacticalHeuristicExpert`   (0.1ms Immediate Combo & Blocker Reflex Head)
"""

from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


ZOO_EXPERT_NAMES: Tuple[str, ...] = (
    "AlphaCandyOmegaNet",
    "EquivariantSymmetricCandyNet",
    "FactoredMuZeroWorldModel",
    "DiscreteDiffusionPlanner",
    "TestTimeTTTAdapter",
    "WormholeGraphNeuralNet",
    "HierarchicalSubgoalCandyNet",
    "CandyGFlowNet",
    "TranspositionMCGSSolver",
    "FastTacticalHeuristicExpert",
)

TOPOLOGICAL_FEATURE_NAMES: Tuple[str, ...] = (
    "jelly_ratio",
    "double_jelly_ratio",
    "blocker_density",
    "chocolate_presence",
    "licorice_frosting_ratio",
    "special_candy_count",
    "color_bomb_presence",
    "corner_objective_ratio",
    "moves_urgency",
    "island_fragmentation",
    "ingredient_portal_load",
    "top_choke_density",
    "bottom_cascade_potential",
    "color_entropy",
    "special_combo_adjacency",
    "desperation_pressure",
)


class TopologicalFeatureExtractor(nn.Module):
    """
    Deterministic, differentiable 16-dimensional board topology feature extractor
    operating directly on the canonical `(B, 20, 9, 9)` tensor.
    """

    def __init__(self, board_h: int = 9, board_w: int = 9):
        super().__init__()
        self.board_h = board_h
        self.board_w = board_w

        # Precompute border + corner mask for feature 7 (corner_objective_ratio)
        border = torch.ones(board_h, board_w, dtype=torch.float32)
        border[1:-1, 1:-1] = 0.0
        # Double weight on the 4 exact corners
        for r in (0, board_h - 1):
            for c in (0, board_w - 1):
                border[r, c] = 2.0
        self.register_buffer("border_mask", border)

    def forward(self, board_tensor: torch.Tensor) -> torch.Tensor:
        """
        Extracts the 16-dimensional topological descriptor vector φ_topo(s) ∈ R^(B, 16).
        """
        B, C, H, W = board_tensor.shape
        eps = 1e-6

        # Active cell count (fallback to 81 if channel 18 is all zeros in synthetic test)
        active = board_tensor[:, 18]
        active_count = active.sum(dim=(1, 2)).clamp(min=1.0)
        is_blank_active = (active.sum(dim=(1, 2)) < 0.5).float()
        norm_cells = active_count * (1.0 - is_blank_active) + 81.0 * is_blank_active

        # 0. jelly_ratio & 1. double_jelly_ratio
        j1 = board_tensor[:, 15].sum(dim=(1, 2))
        j2 = board_tensor[:, 16].sum(dim=(1, 2))
        jelly_ratio = ((j1 + j2) / norm_cells).clamp(0.0, 1.0)
        double_jelly_ratio = (j2 / norm_cells).clamp(0.0, 1.0)

        # 2. blocker_density, 3. chocolate_presence, 4. licorice_frosting_ratio
        blockers_all = board_tensor[:, 10:15].max(dim=1).values.sum(dim=(1, 2))
        blocker_density = (blockers_all / norm_cells).clamp(0.0, 1.0)
        choc = (board_tensor[:, 10] + board_tensor[:, 13]).sum(dim=(1, 2))
        chocolate_presence = (choc / 12.0).clamp(0.0, 1.0)
        lic_frost = (board_tensor[:, 11] + board_tensor[:, 12]).sum(dim=(1, 2))
        licorice_frosting_ratio = (lic_frost / 25.0).clamp(0.0, 1.0)

        # 5. special_candy_count & 6. color_bomb_presence
        specials_map = board_tensor[:, 6:10].sum(dim=1)  # (B, 9, 9)
        special_candy_count = (specials_map.sum(dim=(1, 2)) / 10.0).clamp(0.0, 1.0)
        color_bomb_presence = board_tensor[:, 9].sum(dim=(1, 2)).clamp(0.0, 1.0)

        # 7. corner_objective_ratio
        obj_map = (board_tensor[:, 15] + board_tensor[:, 16] + board_tensor[:, 10:15].max(dim=1).values)
        total_obj = obj_map.sum(dim=(1, 2)).clamp(min=eps)
        corner_obj = (obj_map * self.border_mask.unsqueeze(0)).sum(dim=(1, 2))
        corner_objective_ratio = (corner_obj / (total_obj + 1.0)).clamp(0.0, 1.0)

        # 8. moves_urgency
        ch19 = board_tensor[:, 19].mean(dim=(1, 2)).clamp(0.0, 1.0)
        moves_urgency = (1.0 - ch19).clamp(0.0, 1.0)

        # 9. island_fragmentation (horizontal + vertical transitions in active grid mask ch 18)
        eff_active = active * (1.0 - is_blank_active.view(B, 1, 1)) + is_blank_active.view(B, 1, 1)
        h_edges = torch.abs(eff_active[:, :, 1:] - eff_active[:, :, :-1]).sum(dim=(1, 2))
        v_edges = torch.abs(eff_active[:, 1:, :] - eff_active[:, :-1, :]).sum(dim=(1, 2))
        void_cells = (81.0 - eff_active.sum(dim=(1, 2))).clamp(min=0.0)
        island_fragmentation = ((h_edges + v_edges + void_cells) / 60.0).clamp(0.0, 1.0)

        # 10. ingredient_portal_load
        ing_load = (board_tensor[:, 17].sum(dim=(1, 2)) / 6.0).clamp(0.0, 1.0)

        # 11. top_choke_density (blockers in rows 0..2)
        top_blockers = board_tensor[:, 10:15, :3, :].max(dim=1).values.sum(dim=(1, 2))
        top_choke_density = (top_blockers / 18.0).clamp(0.0, 1.0)

        # 12. bottom_cascade_potential (playable non-blocked candies in rows 6..8)
        bottom_candies = board_tensor[:, 0:6, 6:, :].sum(dim=(1, 2, 3))
        bottom_cascade_potential = (bottom_candies / 27.0).clamp(0.0, 1.0)

        # 13. color_entropy (Normalized Shannon entropy across the 6 candy colors)
        color_counts = board_tensor[:, 0:6].clamp(min=0.0).sum(dim=(2, 3)) + eps  # (B, 6)
        color_probs = color_counts / color_counts.sum(dim=-1, keepdim=True)
        entropy = -(color_probs * torch.log(color_probs + eps)).sum(dim=-1) / 1.7917595  # ln(6)
        color_entropy = entropy.clamp(0.0, 1.0)

        # 14. special_combo_adjacency (adjacent specials horizontally or vertically)
        sp_bin = (specials_map > 0.25).float()
        adj_h = (sp_bin[:, :, :-1] * sp_bin[:, :, 1:]).sum(dim=(1, 2))
        adj_v = (sp_bin[:, :-1, :] * sp_bin[:, 1:, :]).sum(dim=(1, 2))
        special_combo_adjacency = ((adj_h + adj_v) / 3.0).clamp(0.0, 1.0)

        # 15. desperation_pressure (workload vs remaining move fraction)
        desperation_pressure = torch.sigmoid((jelly_ratio + blocker_density) * 3.0 * (0.5 + moves_urgency))

        topo_vec = torch.stack([
            jelly_ratio,
            double_jelly_ratio,
            blocker_density,
            chocolate_presence,
            licorice_frosting_ratio,
            special_candy_count,
            color_bomb_presence,
            corner_objective_ratio,
            moves_urgency,
            island_fragmentation,
            ing_load,
            top_choke_density,
            bottom_cascade_potential,
            color_entropy,
            special_combo_adjacency,
            desperation_pressure,
        ], dim=-1)  # (B, 16)

        return topo_vec


class GrandmasterMoERouter(nn.Module):
    """
    Mixture-of-Experts Level-Topology Router (`GrandmasterMoERouter`).

    Extracts a 16-dimensional topological feature vector `φ_topo(s) ∈ R^(B, 16)`, fuses it with
    a fast spatial convolutional context encoder, and outputs:
      - `expert_weights`: `(B, num_experts)` normalized gating weights across all Model Zoo specialists
      - `top_k_indices` & `top_k_weights`: sparse Top-K expert routing schedule
      - `topological_features`: `(B, 16)` interpretable board topology vector
      - `policy_logits`: `(B, 144)` canonical swap logits synthesized across the MoE specialist heads
      - `value`: `(B, 1)` expected state value in `[-1, 1]`
      - `win_prob`: `(B, 1)` calibrated win probability in `[0, 1]`
      - `load_balancing_loss`: scalar Switch-Transformer auxiliary load-balancing loss
    """

    def __init__(
        self,
        in_channels: int = 20,
        topo_dim: int = 16,
        d_model: int = 144,
        num_experts: int = 10,
        top_k: int = 3,
        num_actions: int = 144,
        board_h: int = 9,
        board_w: int = 9
    ):
        super().__init__()
        self.in_channels = in_channels
        self.topo_dim = topo_dim
        self.d_model = d_model
        self.num_experts = num_experts
        self.top_k = min(top_k, num_experts)
        self.num_actions = num_actions
        self.board_h = board_h
        self.board_w = board_w

        # 1. 16-D Topological Feature Extractor
        self.topology_extractor = TopologicalFeatureExtractor(board_h=board_h, board_w=board_w)

        # 2. Topological MLP + Spatial Context Stem
        self.topo_mlp = nn.Sequential(
            nn.Linear(topo_dim, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Linear(64, 64),
            nn.GELU()
        )
        self.spatial_stem = nn.Sequential(
            nn.Conv2d(in_channels, d_model, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(d_model),
            nn.GELU(),
            nn.Conv2d(d_model, d_model, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(d_model),
            nn.GELU(),
            nn.Conv2d(d_model, d_model, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(d_model),
            nn.GELU()
        )

        # 3. Router Gating Head: maps (64-D topo embedding + 2*d_model global spatial pool) -> num_experts
        self.router_gate = nn.Sequential(
            nn.Linear(64 + d_model * 2, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Linear(128, num_experts)
        )

        # Domain-grounded routing affinity matrix (16 topological features -> 10 Zoo Specialists)
        self.topo_affinity = nn.Linear(topo_dim, num_experts)

        # 4. Internal Lightweight Specialist Policy & Value Adapters
        # Enables standalone forward pass OR dynamic fusion of external Zoo models
        self.spatial_compress = nn.Sequential(
            nn.Conv2d(d_model, 48, kernel_size=1),
            nn.GELU(),
            nn.Flatten()
        )
        self.shared_action_trunk = nn.Sequential(
            nn.Linear(48 * board_h * board_w + 64, 384),
            nn.GELU()
        )
        # Low-rank specialist policy heads for each of the `num_experts` modes
        self.specialist_policy_heads = nn.ModuleList([
            nn.Linear(384, num_actions) for _ in range(num_experts)
        ])
        self.specialist_value_heads = nn.ModuleList([
            nn.Linear(384, 1) for _ in range(num_experts)
        ])

        self.win_prob_calibrator = nn.Sequential(
            nn.Linear(num_experts + 64, 64),
            nn.GELU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def extract_topological_features(self, board_tensor: torch.Tensor) -> torch.Tensor:
        """Public helper returning the (B, 16) topological feature vector."""
        return self.topology_extractor(board_tensor)

    def forward(
        self,
        board_tensor: torch.Tensor,
        legal_mask: Optional[torch.Tensor] = None,
        external_expert_logits: Optional[torch.Tensor] = None,
        external_expert_values: Optional[torch.Tensor] = None,
        use_sparse_top_k: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Computes the 16-D topological vector, expert routing weights across the Model Zoo,
        and routed canonical (B, 144) policy logits and (B, 1) value estimates.

        Args:
            board_tensor: (B, 20, 9, 9) canonical state tensor
            legal_mask: Optional (B, 144) boolean legal action mask
            external_expert_logits: Optional (B, num_experts, 144) precomputed logits from Zoo models
            external_expert_values: Optional (B, num_experts, 1) precomputed values from Zoo models
            use_sparse_top_k: If True, zeroes out non-Top-K experts before combining predictions
        """
        B = board_tensor.shape[0]

        # Step 1: Extract 16-dimensional topological feature vector φ_topo(s)
        topo_features = self.topology_extractor(board_tensor)  # (B, 16)
        topo_emb = self.topo_mlp(topo_features)                # (B, 64)

        # Step 2: Spatial feature extraction
        sp_feat = self.spatial_stem(board_tensor)              # (B, d_model, 9, 9)
        sp_mean = sp_feat.mean(dim=(2, 3))
        sp_max = sp_feat.flatten(2).max(dim=-1).values
        router_in = torch.cat([topo_emb, sp_mean, sp_max], dim=-1)  # (B, 64 + 2*d_model)

        # Step 3: Compute Gating Logits & Probabilities over `num_experts`
        gate_logits = self.router_gate(router_in) + 0.3 * self.topo_affinity(topo_features)  # (B, E)
        dense_weights = F.softmax(gate_logits, dim=-1)  # (B, E)

        # Top-K sparse expert routing
        top_k_vals, top_k_indices = torch.topk(dense_weights, k=self.top_k, dim=-1)  # (B, K)
        top_k_weights = top_k_vals / top_k_vals.sum(dim=-1, keepdim=True).clamp(min=1e-8)

        if use_sparse_top_k:
            sparse_weights = torch.zeros_like(dense_weights)
            sparse_weights.scatter_(1, top_k_indices, top_k_weights)
            active_weights = sparse_weights
        else:
            active_weights = dense_weights

        # Switch-Transformer load-balancing auxiliary loss (encourages uniform expert utilization)
        prob_per_expert = dense_weights.mean(dim=0)
        top1_idx = top_k_indices[:, 0]
        frac_per_expert = F.one_hot(top1_idx, num_classes=self.num_experts).float().mean(dim=0)
        load_balancing_loss = self.num_experts * (prob_per_expert * frac_per_expert).sum()

        # Step 4: Obtain or compute per-expert policy logits (B, E, 144) and values (B, E, 1)
        if external_expert_logits is not None and external_expert_values is not None:
            all_expert_logits = external_expert_logits
            all_expert_values = torch.tanh(external_expert_values)
        else:
            sp_flat = self.spatial_compress(sp_feat)
            trunk_h = self.shared_action_trunk(torch.cat([sp_flat, topo_emb], dim=-1))  # (B, 384)
            all_expert_logits = torch.stack(
                [head(trunk_h) for head in self.specialist_policy_heads], dim=1
            )  # (B, E, 144)
            all_expert_values = torch.stack(
                [torch.tanh(v_head(trunk_h)) for v_head in self.specialist_value_heads], dim=1
            )  # (B, E, 1)

        # Step 5: Mixture-of-Experts weighted synthesis
        policy_logits = torch.einsum("be,bea->ba", active_weights, all_expert_logits)  # (B, 144)
        if legal_mask is not None:
            policy_logits = policy_logits.masked_fill(~legal_mask.bool(), -1e9)

        value = torch.einsum("be,beo->bo", active_weights, all_expert_values)  # (B, 1)
        win_prob = self.win_prob_calibrator(
            torch.cat([all_expert_values.squeeze(-1), topo_emb], dim=-1)
        )  # (B, 1)

        return {
            "policy_logits": policy_logits,
            "value": value,
            "win_prob": win_prob,
            "expert_weights": active_weights,
            "dense_expert_weights": dense_weights,
            "top_k_indices": top_k_indices,
            "top_k_weights": top_k_weights,
            "topological_features": topo_features,
            "load_balancing_loss": load_balancing_loss,
        }

    def explain_routing(self, board_tensor: torch.Tensor) -> Dict[str, object]:
        """
        Human-readable diagnostic summary of the 16 topological features and selected Top-3 specialists.
        """
        if board_tensor.dim() == 3:
            board_tensor = board_tensor.unsqueeze(0)
        with torch.no_grad():
            out = self.forward(board_tensor)
            topo = out["topological_features"][0].cpu().tolist()
            weights = out["dense_expert_weights"][0].cpu().tolist()
            top_indices = out["top_k_indices"][0].cpu().tolist()

            topo_dict = {
                TOPOLOGICAL_FEATURE_NAMES[i]: round(topo[i], 4)
                for i in range(min(len(topo), len(TOPOLOGICAL_FEATURE_NAMES)))
            }
            ranked_experts = [
                (
                    ZOO_EXPERT_NAMES[idx] if idx < len(ZOO_EXPERT_NAMES) else f"Expert_{idx}",
                    round(weights[idx], 4)
                )
                for idx in top_indices
            ]
            return {
                "top_specialists": ranked_experts,
                "topological_profile": topo_dict,
                "win_prob": round(float(out["win_prob"][0, 0].item()), 4),
            }

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


if __name__ == "__main__":
    torch.manual_seed(42)
    router = GrandmasterMoERouter()
    print(f"[GrandmasterMoERouter] Trainable Parameters: {router.count_parameters():,}")
    dummy = torch.rand(2, 20, 9, 9)
    out = router(dummy)
    print("  policy_logits:", out["policy_logits"].shape)
    print("  value:", out["value"].shape)
    print("  expert_weights:", out["expert_weights"].shape)
    print("  topological_features:", out["topological_features"].shape)
    print("  Routing explanation sample:", router.explain_routing(dummy[0])["top_specialists"])
