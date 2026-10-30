"""
AlphaCandy Model Zoo — Part 4.3: Blocker Containment Specialist Network (`BlockerContainmentSpecialistNet`).
============================================================================================================
Ultra-specialized architecture for dense blocker fortresses (Spreading Chocolate, Multi-Layer Frosting,
Licorice Locks, and Magic Mixers).

Architectural Innovations:
  1. 3-Step Blocker Contagion Forecaster Head (`ThreeStepContagionForecaster`):
     - Simulates 3 steps (t+1, t+2, t+3) of morphological + neural Chocolate & Mixer contagion
       expansion across playable board cells if left unhit, outputting a (B, 3, 9, 9) spatial
       infection probability trajectory.
  2. Frosting Cluster & 3x3 Wrapped Blast-Yield Analyzer (`BlockerClusterBlastAnalyzer`):
     - Computes the exact 4-neighbor Adjacent Blocker Strike Perimeter (where standard matches
       stun Chocolate growth and chip Frosting layers) and the 3x3 Wrapped Candy Detonation
       Yield Map inside high-density frosting fortresses.
  3. Action Logit Containment Booster:
     - Explicitly boosts swap logits for moves that strike adjacent to active Chocolate/Mixers
       or detonate Wrapped candies (channel 8) inside high-density frosting clusters.
"""

import math
from typing import Dict, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class ThreeStepContagionForecaster(nn.Module):
    """
    Predicts where Chocolate (channel 10) and Spawning Mixers (channel 13) will spread over
    the next 3 moves (t+1, t+2, t+3) if left unhit on the current turn.
    Combines exact 4-neighbor morphological infection dilation with learned neural contagion
    gating conditioned on board topology and blocker barriers.
    """

    def __init__(self, d_model: int = 192):
        super().__init__()
        # 4-neighbor orthogonal spread kernel (including center retention)
        spread_kernel = torch.tensor(
            [[[[0.0, 1.0, 0.0],
               [1.0, 1.0, 1.0],
               [0.0, 1.0, 0.0]]]],
            dtype=torch.float32
        )
        # Strict 4-neighbor perimeter kernel (excluding center)
        perimeter_kernel = torch.tensor(
            [[[[0.0, 1.0, 0.0],
               [1.0, 0.0, 1.0],
               [0.0, 1.0, 0.0]]]],
            dtype=torch.float32
        )
        # 3x3 Wrapped Candy double-blast kernel (center + 8 neighbors weighted for 2-stage blast)
        wrapped_blast_kernel = torch.tensor(
            [[[[1.0, 1.25, 1.0],
               [1.25, 1.5, 1.25],
               [1.0, 1.25, 1.0]]]],
            dtype=torch.float32
        )

        self.register_buffer("spread_kernel", spread_kernel)
        self.register_buffer("perimeter_kernel", perimeter_kernel)
        self.register_buffer("wrapped_blast_kernel", wrapped_blast_kernel)

        # Neural step refinement for each of the 3 contagion horizon steps
        self.step_refiners = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(d_model + 3, 64, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(64),
                nn.GELU(),
                nn.Conv2d(64, 1, kernel_size=1)
            )
            for _ in range(3)
        ])

    def compute_static_blocker_maps(
        self, board_tensor: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Extracts initial contagion seed map, adjacent strike perimeter map, and 3x3 Wrapped
        detonation yield map inside frosting/blocker clusters.
        """
        B, C, H, W = board_tensor.shape

        active_raw = board_tensor[:, 18:19]
        has_active = (active_raw.sum(dim=(2, 3), keepdim=True) > 0).float()
        active_grid = active_raw * has_active + (1.0 - has_active)

        choc = board_tensor[:, 10:11]          # Spreading Chocolate
        licorice = board_tensor[:, 11:12]      # Licorice locks
        frosting = board_tensor[:, 12:13]      # Frosting layers
        thick_blocker = board_tensor[:, 13:14] # Multi-layer frosting / Magic Mixers
        any_blocker = board_tensor[:, 14:15]

        # Contagion seed (Chocolate + Spawning Mixers)
        contagion_seed = (choc + 0.75 * thick_blocker).clamp(0.0, 1.0)

        # Infectible cells: active cells not currently walled off by hard frosting or void
        infectible_mask = (active_grid * (1.0 - 0.8 * frosting)).clamp(0.0, 1.0)

        # Adjacent Blocker Strike Perimeter: 4-neighbors of Chocolate (highest priority) + Frosting
        choc_perim = F.conv2d(contagion_seed, self.perimeter_kernel, padding=1)
        frost_perim = F.conv2d(frosting + thick_blocker, self.perimeter_kernel, padding=1)
        adjacent_strike_map = (
            (1.5 * choc_perim + 0.85 * frost_perim + 0.4 * F.conv2d(any_blocker, self.perimeter_kernel, padding=1))
            * active_grid
            * (1.0 - any_blocker.clamp(0.0, 1.0) * 0.5)
        )

        # 3x3 Wrapped Candy Detonation Yield inside Frosting & Blocker Clusters
        cluster_density = 1.4 * frosting + 2.0 * thick_blocker + 1.5 * choc + 0.9 * licorice
        wrapped_blast_yield = F.conv2d(cluster_density, self.wrapped_blast_kernel, padding=1) * active_grid

        return (
            contagion_seed.squeeze(1),
            infectible_mask.squeeze(1),
            adjacent_strike_map.squeeze(1),
            wrapped_blast_yield.squeeze(1),
        )

    def forward(
        self,
        feat_2d: torch.Tensor,
        contagion_seed: torch.Tensor,
        infectible_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Rolls out the 3-step Blocker Contagion Forecast over steps t+1, t+2, t+3.
        Returns:
            contagion_forecast_3step: (B, 3, 9, 9) probability map in [0, 1]
        """
        curr_infection = contagion_seed.unsqueeze(1)  # (B, 1, 9, 9)
        inf_mask = infectible_mask.unsqueeze(1)       # (B, 1, 9, 9)
        seed_plane = contagion_seed.unsqueeze(1)

        forecasts = []
        for step_idx, refiner in enumerate(self.step_refiners):
            # Morphological 4-neighbor expansion gated by infectible board cells
            dilated = (F.conv2d(curr_infection, self.spread_kernel, padding=1) / 2.5).clamp(0.0, 1.0)
            morph_next = torch.maximum(curr_infection, dilated * inf_mask)

            # Neural contagion residual correction
            step_in = torch.cat([feat_2d, curr_infection, morph_next, seed_plane], dim=1)
            delta_logit = refiner(step_in)
            curr_infection = torch.sigmoid(torch.logit(morph_next.clamp(1e-4, 1.0 - 1e-4)) + 0.35 * delta_logit)
            curr_infection = curr_infection * inf_mask
            forecasts.append(curr_infection.squeeze(1))

        return torch.stack(forecasts, dim=1)  # (B, 3, 9, 9)


class FortressContainmentAttentionBlock(nn.Module):
    """
    Spatial Residual + Multi-Head Containment Attention Block.
    Focuses attention on the boundary between spreading blockers, frosting clusters,
    and available special candies (especially Wrapped candies on channel 8).
    """

    def __init__(self, d_model: int = 192, num_heads: int = 6):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)

        # Local Multi-Dilated Depthwise-Separable Conv for 3x3 and 5x5 Fortress Clusters
        self.local_conv = nn.Sequential(
            nn.Conv2d(d_model, d_model, kernel_size=3, padding=1, groups=d_model, bias=False),
            nn.BatchNorm2d(d_model),
            nn.GELU(),
            nn.Conv2d(d_model, d_model, kernel_size=1, bias=False),
            nn.BatchNorm2d(d_model),
            nn.GELU()
        )

        self.norm1 = nn.LayerNorm(d_model)
        self.qkv = nn.Linear(d_model, d_model * 3, bias=False)
        self.out_proj = nn.Linear(d_model, d_model)

        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 3),
            nn.GELU(),
            nn.Linear(d_model * 3, d_model)
        )

    def forward(self, x_2d: torch.Tensor, hazard_priority_bias: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x_2d: (B, D, 9, 9)
            hazard_priority_bias: (B, 1, 1, 81) key-side attention boost for high-hazard cells
        """
        B, D, H, W = x_2d.shape
        N = H * W

        x_2d = x_2d + self.local_conv(x_2d)
        tokens = x_2d.flatten(2).transpose(1, 2)  # (B, 81, D)

        t_norm = self.norm1(tokens)
        q, k, v = self.qkv(t_norm).chunk(3, dim=-1)
        q = q.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale + hazard_priority_bias
        attn = F.softmax(scores, dim=-1)

        out = torch.matmul(attn, v).transpose(1, 2).contiguous().view(B, N, D)
        tokens = tokens + self.out_proj(out)
        tokens = tokens + self.ffn(self.norm2(tokens))

        return tokens.transpose(1, 2).view(B, D, H, W)


class BlockerContainmentSpecialistNet(nn.Module):
    """
    BlockerContainmentSpecialistNet: Specialized Deep Architecture for Dense Blocker Fortresses.
    Features a 3-Step Blocker Contagion Forecaster Head and an explicit Action Logit Containment
    Booster for adjacent chocolate/frosting strikes and 3x3 Wrapped candy cluster detonations.
    """

    def __init__(
        self,
        in_channels: int = 20,
        board_h: int = 9,
        board_w: int = 9,
        d_model: int = 192,
        num_layers: int = 6,
        num_heads: int = 6,
        num_actions: int = 144
    ):
        super().__init__()
        self.in_channels = in_channels
        self.board_h = board_h
        self.board_w = board_w
        self.num_nodes = board_h * board_w
        self.d_model = d_model
        self.num_actions = num_actions

        # 1. 3-Step Blocker Contagion Forecaster & Blast-Yield Analyzer
        self.contagion_forecaster = ThreeStepContagionForecaster(d_model=d_model)

        # 2. Stem fusing 20 canonical channels + 4 static blocker/blast maps
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels + 4, d_model, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(d_model),
            nn.GELU(),
            nn.Conv2d(d_model, d_model, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(d_model),
            nn.GELU()
        )
        self.pos_embed = nn.Parameter(torch.randn(1, d_model, board_h, board_w) * 0.02)

        # 3. 6-Layer Fortress Containment Backbone
        self.blocks = nn.ModuleList([
            FortressContainmentAttentionBlock(d_model=d_model, num_heads=num_heads)
            for _ in range(num_layers)
        ])

        # 4. Containment & Wrapped-Blast Action Booster Head
        # Takes latent features + 3-step contagion forecast (3 channels) + strike/blast maps (2 channels)
        self.containment_fusion = nn.Sequential(
            nn.Conv2d(d_model + 5, d_model, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(d_model),
            nn.GELU()
        )
        self.horiz_strike_scorer = nn.Sequential(
            nn.Linear(d_model * 2 + 6, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1)
        )
        self.vert_strike_scorer = nn.Sequential(
            nn.Linear(d_model * 2 + 6, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1)
        )
        self.global_policy_head = nn.Sequential(
            nn.Conv2d(d_model, 96, kernel_size=3, padding=1),
            nn.BatchNorm2d(96),
            nn.GELU(),
            nn.Flatten(),
            nn.Linear(96 * self.num_nodes, 256),
            nn.GELU(),
            nn.Linear(256, num_actions)
        )

        # 5. Blocker Containment Urgency, Value & Win Probability Heads
        self.urgency_head = nn.Sequential(
            nn.Linear(d_model * 2 + 3, 64),
            nn.GELU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        self.value_head = nn.Sequential(
            nn.Linear(d_model * 2 + 3, 128),
            nn.GELU(),
            nn.Linear(128, 1),
            nn.Tanh()
        )
        self.win_head = nn.Sequential(
            nn.Linear(d_model * 2 + 3, 128),
            nn.GELU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def _score_containment_swaps(
        self,
        z_2d: torch.Tensor,
        adj_strike_map: torch.Tensor,
        wrapped_blast_map: torch.Tensor,
        contagion_step3: torch.Tensor
    ) -> torch.Tensor:
        """
        Scores all 72 horizontal and 72 vertical swaps with explicit boosts for:
          - Adjacent Blocker Strikes (stunning chocolate/mixer expansion & clearing frosting)
          - Detonating Wrapped Candies inside high-density frosting clusters
          - Pre-empting 3-step contagion spread hotspots
        """
        B, D, H, W = z_2d.shape

        # Horizontal swaps (9 x 8 = 72)
        h_l = z_2d[:, :, :, :-1].permute(0, 2, 3, 1)
        h_r = z_2d[:, :, :, 1:].permute(0, 2, 3, 1)
        h_phys = torch.stack([
            adj_strike_map[:, :, :-1],
            adj_strike_map[:, :, 1:],
            wrapped_blast_map[:, :, :-1],
            wrapped_blast_map[:, :, 1:],
            contagion_step3[:, :, :-1],
            contagion_step3[:, :, 1:]
        ], dim=-1)  # (B, 9, 8, 6)
        h_in = torch.cat([h_l, h_r, h_phys], dim=-1).reshape(B, 72, 2 * D + 6)
        h_logits = self.horiz_strike_scorer(h_in).squeeze(-1)

        # Vertical swaps (8 x 9 = 72)
        v_t = z_2d[:, :, :-1, :].permute(0, 2, 3, 1)
        v_b = z_2d[:, :, 1:, :].permute(0, 2, 3, 1)
        v_phys = torch.stack([
            adj_strike_map[:, :-1, :],
            adj_strike_map[:, 1:, :],
            wrapped_blast_map[:, :-1, :],
            wrapped_blast_map[:, 1:, :],
            contagion_step3[:, :-1, :],
            contagion_step3[:, 1:, :]
        ], dim=-1)  # (B, 8, 9, 6)
        v_in = torch.cat([v_t, v_b, v_phys], dim=-1).reshape(B, 72, 2 * D + 6)
        v_logits = self.vert_strike_scorer(v_in).squeeze(-1)

        return torch.cat([h_logits, v_logits], dim=-1)  # (B, 144)

    def forward(
        self,
        board_tensor: torch.Tensor,
        legal_mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        B, C, H, W = board_tensor.shape

        # Step 1: Extract static blocker perimeter & 3x3 Wrapped frosting blast maps
        contagion_seed, infectible_mask, adj_strike_map, wrapped_blast_yield = (
            self.contagion_forecaster.compute_static_blocker_maps(board_tensor)
        )

        # Highlight cells where an actual Wrapped candy (channel 8) sits inside a frosting cluster
        wrapped_present = board_tensor[:, 8]
        active_wrapped_blast = wrapped_blast_yield * (0.35 + 1.65 * wrapped_present)

        static_planes = torch.stack([
            contagion_seed, infectible_mask, adj_strike_map, active_wrapped_blast
        ], dim=1)  # (B, 4, 9, 9)

        # Step 2: Stem + 6-layer Fortress Containment Attention Backbone
        x = self.stem(torch.cat([board_tensor, static_planes], dim=1)) + self.pos_embed
        hazard_bias = (0.25 * (adj_strike_map + active_wrapped_blast)).view(B, 1, 1, H * W)
        for block in self.blocks:
            x = block(x, hazard_bias)

        # Step 3: 3-Step Blocker Contagion Forecaster Head (t+1, t+2, t+3)
        contagion_forecast_3step = self.contagion_forecaster(x, contagion_seed, infectible_mask)  # (B, 3, 9, 9)

        # Step 4: Fuse 3-step contagion forecast into policy scoring
        aux_planes = torch.cat([
            contagion_forecast_3step,
            adj_strike_map.unsqueeze(1),
            active_wrapped_blast.unsqueeze(1)
        ], dim=1)  # (B, 5, 9, 9)
        z_fused = self.containment_fusion(torch.cat([x, aux_planes], dim=1))

        strike_logits = self._score_containment_swaps(
            z_fused, adj_strike_map, active_wrapped_blast, contagion_forecast_3step[:, 2]
        )
        policy_logits = strike_logits + self.global_policy_head(z_fused)

        if legal_mask is not None:
            policy_logits = policy_logits.masked_fill(~legal_mask.bool(), -1e9)

        # Step 5: Readout for Blocker Containment Urgency, Value & Win Probability
        z_flat = z_fused.flatten(2)
        c_rates = contagion_forecast_3step.flatten(2).mean(dim=-1)  # (B, 3) mean infection at t+1, t+2, t+3
        readout = torch.cat([z_flat.mean(dim=-1), z_flat.max(dim=-1).values, c_rates], dim=-1)

        blocker_containment_urgency = self.urgency_head(readout)
        value = self.value_head(readout)
        win_prob = self.win_head(readout)

        return {
            "policy_logits": policy_logits,
            "value": value,
            "win_prob": win_prob,
            "contagion_forecast_3step": contagion_forecast_3step,
            "adjacent_blocker_strike_map": adj_strike_map,
            "wrapped_frosting_blast_yield": active_wrapped_blast,
            "blocker_containment_urgency": blocker_containment_urgency,
        }

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def create_blocker_containment_specialist(device: str = "cpu") -> BlockerContainmentSpecialistNet:
    model = BlockerContainmentSpecialistNet()
    return model.to(device)


if __name__ == "__main__":
    torch.manual_seed(42)
    model = create_blocker_containment_specialist("cpu")
    params = model.count_parameters()
    print(f"[BlockerContainmentSpecialistNet] Trainable Parameters: {params:,} ({params / 1e6:.2f}M)")

    dummy_board = torch.zeros(2, 20, 9, 9)
    dummy_board[:, 18] = 1.0  # Active grid
    dummy_board[0, 10, 4, 4] = 1.0  # Spreading Chocolate at (4, 4)
    dummy_board[0, 12, 2:5, 2:5] = 1.0  # 3x3 Frosting cluster
    dummy_board[0, 8, 3, 3] = 1.0   # Wrapped candy inside the frosting cluster!

    out = model(dummy_board)
    for k, v in out.items():
        print(f"  {k}: {tuple(v.shape)}")
    print("  Mean Contagion Spread (t+1, t+2, t+3):",
          [round(x, 4) for x in out["contagion_forecast_3step"][0].mean(dim=(1, 2)).tolist()])
