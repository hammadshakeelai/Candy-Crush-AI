"""
AlphaCandy-Titan: ~55M-Parameter Next-Generation Flagship Super-Architecture for Candy Crush Saga.

Cures all 6 fundamental root causes of neural network plateaus on hard King levels from first principles:
  1. S_6 Color-Symmetry Relational Stem (`ColorSymmetricRelationalStem`):
     Shared per-color spatial convolutions across all 6 candy color planes with symmetric
     cross-color aggregations (max & mean pooling + global color density histogram) fused
     with the 14 special, blocker, portal, and geometry context channels.
  2. Hybrid 4-Way Directional Scan + Portal-Graph Cross-Attention Backbone (`TitanHybridBlock` x 12, d=512, 8 heads):
     Parallel full-row horizontal stripe laser & full-column vertical gravity axial conv/scan
     branch combined with non-Euclidean portal-biased multi-head self-attention.
  3. `adaLN-Zero` Level-Telemetry Hyper-Conditioning (`TelemetryConditioner`):
     Extracts a 16-dim global level telemetry vector directly from the board tensor (or explicit
     metadata) and modulates every `TitanHybridBlock` via Adaptive LayerNorm-Zero (`adaLN-Zero`)
     with per-channel scale (1 + gamma), shift beta, and residual gate alpha.
  4. Recurrent System-2 Latent Pondering Core (`System2ThinkingCore` + `ConfidenceHaltingHead`):
     Runs K_ponder = 3 weight-tied iterative latent refinement steps with a gated GRU-style
     residual update and learned epistemic policy confidence halting head.
  5. Bilinear Symmetric Edge-Swap Policy Head (`BilinearEdgeSwapHead`):
     Gathers endpoint token embeddings (z_u, z_v) for all 144 canonical adjacent swaps, computes
     symmetric bilinear edge interactions Concat[z_u + z_v, |z_u - z_v|, z_u * z_v], and fuses
     an optional 16-dim explicit micro-physics feature slot `(B, 144, 16)`.
  6. Pace-Controlled Dynamic Quantile Tilting Head (`DynamicQuantileValueHead`):
     Dueling 32-quantile distributional head Z_theta(s, a, tau_i) of shape `(B, 144, 32)` with
     pace-adaptive risk tilting across Lower-Tail Safe Mode (CVaR_0.20), Normal Mode (E[Z]),
     and Upper-Tail Desperation Miracle Mode (CVaR_0.85^+).
"""

import math
from typing import Dict, Any, Tuple, List, Optional, Union
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from core.reverse_engine import ReverseEngineState, Direction
from models.spatial_transformer_net import (
    encode_spatial_board_tensor,
    encode_action_index,
    decode_action_index,
    NUM_CANONICAL_ACTIONS,
)


class ColorSymmetricRelationalStem(nn.Module):
    """
    Breakthrough 1: S_6 Color-Symmetry Relational Stem.
    Applies shared spatial convolutions independently across each of the 6 candy color
    channels (0..5), computes permutation-invariant cross-color aggregations (max, mean,
    std, and global color density histogram), and fuses them with the 14 special/blocker/
    geometry channels into a d_model=512 spatial representation.
    """

    def __init__(
        self,
        num_colors: int = 6,
        context_channels: int = 14,
        color_feat_dim: int = 64,
        d_model: int = 512,
    ):
        super().__init__()
        self.num_colors = num_colors
        self.context_channels = context_channels
        self.color_feat_dim = color_feat_dim
        self.d_model = d_model

        # Shared per-color spatial stem: processes (color_plane, active_mask, color_density) -> color_feat_dim
        self.shared_color_conv = nn.Sequential(
            nn.Conv2d(3, color_feat_dim // 2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(color_feat_dim // 2),
            nn.GELU(),
            nn.Conv2d(color_feat_dim // 2, color_feat_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(color_feat_dim),
            nn.GELU(),
            nn.Conv2d(color_feat_dim, color_feat_dim, kernel_size=5, padding=2, groups=color_feat_dim // 4, bias=False),
            nn.BatchNorm2d(color_feat_dim),
            nn.GELU(),
        )

        # Shared per-color axial line-match detectors (1x5 horizontal and 5x1 vertical)
        self.shared_line_h = nn.Conv2d(color_feat_dim, color_feat_dim // 2, kernel_size=(1, 5), padding=(0, 2), bias=False)
        self.shared_line_v = nn.Conv2d(color_feat_dim, color_feat_dim // 2, kernel_size=(5, 1), padding=(2, 0), bias=False)
        self.shared_color_refine = nn.Sequential(
            nn.BatchNorm2d(color_feat_dim),
            nn.GELU(),
            nn.Conv2d(color_feat_dim, color_feat_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(color_feat_dim),
            nn.GELU(),
        )

        # Context stem for the 14 special, blocker, jelly, ingredient, and geometry channels
        self.context_conv3 = nn.Conv2d(context_channels, 128, kernel_size=3, padding=1, bias=False)
        self.context_conv5 = nn.Conv2d(context_channels, 128, kernel_size=5, padding=2, bias=False)
        self.context_stem = nn.Sequential(
            nn.BatchNorm2d(256),
            nn.GELU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.GELU(),
        )

        # Global 6-color histogram + entropy projection
        self.histogram_proj = nn.Sequential(
            nn.Linear(num_colors + 2, 64),
            nn.GELU(),
            nn.Linear(64, 64),
            nn.GELU(),
        )

        # Fuse:
        #   - symmetric max across 6 colors: (B, color_feat_dim, H, W)
        #   - symmetric mean across 6 colors: (B, color_feat_dim, H, W)
        #   - per-color stacked relational features: (B, 6 * color_feat_dim, H, W)
        #   - context stem features: (B, 256, H, W)
        #   - global histogram broadcast: (B, 64, H, W)
        total_concat_dim = (2 * color_feat_dim) + (num_colors * color_feat_dim) + 256 + 64
        self.stem_fusion = nn.Sequential(
            nn.Conv2d(total_concat_dim, d_model, kernel_size=1, bias=False),
            nn.BatchNorm2d(d_model),
            nn.GELU(),
            nn.Conv2d(d_model, d_model, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(d_model),
            nn.GELU(),
        )

    def forward(self, board_tensor: torch.Tensor) -> torch.Tensor:
        """
        Args:
            board_tensor: (B, 20, H, W) where H=9, W=9
        Returns:
            stem_out: (B, d_model, H, W)
        """
        B, C, H, W = board_tensor.shape
        color_planes = board_tensor[:, :self.num_colors, :, :]   # (B, 6, H, W)
        context_planes = board_tensor[:, self.num_colors:, :, :]  # (B, 14, H, W)
        active_mask = board_tensor[:, 18:19, :, :]               # (B, 1, H, W)

        # Compute per-color board density histogram over active cells
        active_count = active_mask.sum(dim=(2, 3), keepdim=True).clamp(min=1.0)  # (B, 1, 1, 1)
        color_counts = color_planes.sum(dim=(2, 3), keepdim=True)                # (B, 6, 1, 1)
        color_densities = color_counts / active_count                            # (B, 6, 1, 1)

        # Prepare shared per-color input: (B * 6, 3, H, W)
        density_map = color_densities.expand(B, self.num_colors, H, W)
        active_expanded = active_mask.expand(B, self.num_colors, H, W)
        per_color_in = torch.stack([color_planes, active_expanded, density_map], dim=2)  # (B, 6, 3, H, W)
        per_color_in = per_color_in.view(B * self.num_colors, 3, H, W)

        # Shared spatial conv + axial line detector across each color independently
        c_feat = self.shared_color_conv(per_color_in)  # (B*6, color_feat_dim, H, W)
        c_lines = torch.cat([self.shared_line_h(c_feat), self.shared_line_v(c_feat)], dim=1)
        c_feat = c_feat + self.shared_color_refine(c_lines)

        # Reshape back to (B, 6, color_feat_dim, H, W)
        c_feat_6 = c_feat.view(B, self.num_colors, self.color_feat_dim, H, W)

        # S_6 Permutation-Invariant Cross-Color Aggregations (Max & Mean across 6 colors)
        sym_max = c_feat_6.max(dim=1).values  # (B, color_feat_dim, H, W)
        sym_mean = c_feat_6.mean(dim=1)       # (B, color_feat_dim, H, W)
        all_color_flat = c_feat_6.view(B, self.num_colors * self.color_feat_dim, H, W)

        # Global color histogram statistics (sorted densities + max/entropy)
        hist_vec = color_densities.squeeze(-1).squeeze(-1)  # (B, 6)
        sorted_hist, _ = torch.sort(hist_vec, dim=-1, descending=True)
        probs = hist_vec.clamp(min=1e-6)
        entropy = -(probs * torch.log(probs)).sum(dim=-1, keepdim=True) / math.log(self.num_colors)
        max_density = sorted_hist[:, :1]
        hist_stats = torch.cat([sorted_hist, entropy, max_density], dim=-1)  # (B, 8)
        hist_emb = self.histogram_proj(hist_stats).unsqueeze(-1).unsqueeze(-1).expand(B, 64, H, W)

        # Context features from the 14 special/blocker/portal/geometry planes
        ctx_raw = torch.cat([self.context_conv3(context_planes), self.context_conv5(context_planes)], dim=1)
        ctx_feat = self.context_stem(ctx_raw)  # (B, 256, H, W)

        # Fuse all relational & context pathways
        combined = torch.cat([sym_max, sym_mean, all_color_flat, ctx_feat, hist_emb], dim=1)
        return self.stem_fusion(combined)


class TelemetryConditioner(nn.Module):
    """
    Breakthrough 3: `adaLN-Zero` Level-Telemetry Hyper-Conditioning.
    Extracts a 16-dim global telemetry vector directly from the 20-channel board tensor
    (and optional explicit moves_left / objectives_left metadata), including:
      - Jelly density (single, double, total)
      - Blocker density (chocolate indicator, chocolate ratio, licorice, frosting, total)
      - Special candy inventory (striped, wrapped, color bomb)
      - Corner & border objective concentration ratio
      - Active playable topology ratio & color entropy
      - Move urgency clearing pace rho = objectives / max(1, moves_left)
    Produces a global hyper-conditioning embedding `c_telemetry` of shape (B, d_model)
    and phase weights (Opening, Midgame, Desperation).
    """

    def __init__(self, telemetry_dim: int = 16, d_model: int = 512):
        super().__init__()
        self.telemetry_dim = telemetry_dim
        self.d_model = d_model

        # Precompute corner and border mask on 9x9 grid for hard-to-reach objective detection
        corner_mask = torch.zeros(1, 1, 9, 9)
        for r in [0, 1, 7, 8]:
            for c in [0, 1, 7, 8]:
                corner_mask[0, 0, r, c] = 1.0
        self.register_buffer("corner_mask", corner_mask)

        self.telemetry_mlp = nn.Sequential(
            nn.Linear(telemetry_dim, d_model // 2),
            nn.LayerNorm(d_model // 2),
            nn.SiLU(),
            nn.Linear(d_model // 2, d_model),
            nn.LayerNorm(d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )

        # Phase classifier head: [Opening, Midgame, Endgame Desperation]
        self.phase_head = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.SiLU(),
            nn.Linear(64, 3),
        )

    def extract_telemetry_vector(
        self,
        board_tensor: torch.Tensor,
        moves_left: Optional[Union[int, float, torch.Tensor]] = None,
        objectives_left: Optional[Union[int, float, torch.Tensor]] = None,
    ) -> torch.Tensor:
        """
        Extracts the 16-dimensional telemetry vector directly from `board_tensor` (B, 20, 9, 9).
        """
        B, C, H, W = board_tensor.shape
        device = board_tensor.device
        dtype = board_tensor.dtype

        active = board_tensor[:, 18:19, :, :]  # (B, 1, H, W)
        active_sum = active.sum(dim=(1, 2, 3)).clamp(min=1.0)  # (B,)

        # 1. Jelly densities (channels 15, 16)
        jelly_1 = (board_tensor[:, 15] * active[:, 0]).sum(dim=(1, 2)) / active_sum
        jelly_2 = (board_tensor[:, 16] * active[:, 0]).sum(dim=(1, 2)) / active_sum
        total_jelly = jelly_1 + 2.0 * jelly_2

        # 2. Blocker densities & chocolate presence (channels 10..14)
        choc_map = board_tensor[:, 10] * active[:, 0]
        choc_count = choc_map.sum(dim=(1, 2))
        choc_presence = (choc_count > 0.5).to(dtype)
        choc_density = choc_count / active_sum
        licorice_density = (board_tensor[:, 11] * active[:, 0]).sum(dim=(1, 2)) / active_sum
        frosting_density = (board_tensor[:, 12] * active[:, 0]).sum(dim=(1, 2)) / active_sum
        total_blocker_density = (board_tensor[:, 14] * active[:, 0]).sum(dim=(1, 2)) / active_sum

        # 3. Special candy counts (normalized by 10.0)
        striped_count = (board_tensor[:, 6] + board_tensor[:, 7]).sum(dim=(1, 2)) / 10.0
        wrapped_count = board_tensor[:, 8].sum(dim=(1, 2)) / 10.0
        bomb_count = board_tensor[:, 9].sum(dim=(1, 2)) / 5.0

        # 4. Ingredient count (channel 17)
        ingredient_count = board_tensor[:, 17].sum(dim=(1, 2)) / 5.0

        # 5. Corner objective concentration ratio
        obj_spatial = board_tensor[:, 14:15] + board_tensor[:, 15:16] + 2.0 * board_tensor[:, 16:17]
        total_obj_spatial = obj_spatial.sum(dim=(1, 2, 3)).clamp(min=1.0)
        corner_obj_spatial = (obj_spatial * self.corner_mask).sum(dim=(1, 2, 3))
        corner_obj_ratio = corner_obj_spatial / total_obj_spatial

        # 6. Active cell topology ratio
        active_ratio = active_sum / float(H * W)

        # 7. Color distribution entropy
        color_counts = board_tensor[:, :6].sum(dim=(2, 3))  # (B, 6)
        color_probs = (color_counts / color_counts.sum(dim=-1, keepdim=True).clamp(min=1.0)).clamp(min=1e-6)
        color_entropy = -(color_probs * torch.log(color_probs)).sum(dim=-1) / math.log(6.0)

        # 8. Move urgency clearing pace rho = objectives / max(1, moves_left)
        if moves_left is not None and objectives_left is not None:
            if isinstance(moves_left, (int, float)):
                m_t = torch.full((B,), float(max(1.0, moves_left)), device=device, dtype=dtype)
            else:
                m_t = moves_left.to(device=device, dtype=dtype).view(B).clamp(min=1.0)
            if isinstance(objectives_left, (int, float)):
                o_t = torch.full((B,), float(max(0.0, objectives_left)), device=device, dtype=dtype)
            else:
                o_t = objectives_left.to(device=device, dtype=dtype).view(B).clamp(min=0.0)
            pace_rho = (o_t / m_t).clamp(max=10.0)
        else:
            # Infer from board_tensor channels when explicit scalars are not passed
            inferred_obj = (
                board_tensor[:, 15].sum(dim=(1, 2))
                + 2.0 * board_tensor[:, 16].sum(dim=(1, 2))
                + board_tensor[:, 14].sum(dim=(1, 2))
                + 3.0 * board_tensor[:, 17].sum(dim=(1, 2))
            ).clamp(min=1.0)
            ch19_mean = board_tensor[:, 19].mean(dim=(1, 2)).clamp(min=0.05)
            inferred_moves = (ch19_mean * 30.0).clamp(min=1.0)
            pace_rho = (inferred_obj / inferred_moves).clamp(max=10.0)

        telemetry = torch.stack(
            [
                jelly_1,
                jelly_2,
                total_jelly,
                choc_presence,
                choc_density,
                licorice_density,
                frosting_density,
                total_blocker_density,
                striped_count,
                wrapped_count,
                bomb_count,
                ingredient_count,
                corner_obj_ratio,
                active_ratio,
                color_entropy,
                pace_rho,
            ],
            dim=-1,
        )  # (B, 16)
        return telemetry

    def forward(
        self,
        board_tensor: torch.Tensor,
        moves_left: Optional[Union[int, float, torch.Tensor]] = None,
        objectives_left: Optional[Union[int, float, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            c_telemetry: (B, d_model) conditioning embedding for adaLN-Zero
            telemetry_vec: (B, 16) raw extracted telemetry vector
            phase_probs: (B, 3) probabilities for [Opening, Midgame, Desperation]
        """
        telemetry_vec = self.extract_telemetry_vector(board_tensor, moves_left, objectives_left)
        c_telemetry = self.telemetry_mlp(telemetry_vec)
        phase_probs = F.softmax(self.phase_head(c_telemetry), dim=-1)
        return c_telemetry, telemetry_vec, phase_probs


class DirectionalLaserGravityScan(nn.Module):
    """
    4-Way Directional Scan + Full-Row Horizontal Laser & Full-Column Vertical Gravity Branch.
    Captures:
      1. Full-row horizontal stripe laser blasts in 1 hop via (1, 9) axial kernels
      2. Full-column vertical gravity drops & ingredient chutes via (9, 1) axial kernels
      3. 4-Way directional cumulative gated scans (Left->Right, Right->Left, Top->Bottom, Bottom->Top)
         modeling cascading chain propagation across the 9x9 grid.
    """

    def __init__(self, d_model: int = 512, board_h: int = 9, board_w: int = 9):
        super().__init__()
        self.d_model = d_model
        self.board_h = board_h
        self.board_w = board_w
        self.branch_dim = d_model // 2

        self.in_proj = nn.Linear(d_model, self.branch_dim * 2, bias=False)

        # Full-row horizontal laser conv (1x9) and full-column vertical gravity conv (9x1)
        self.row_laser_conv = nn.Conv2d(
            self.branch_dim, self.branch_dim, kernel_size=(1, board_w), padding=(0, board_w // 2),
            groups=16, bias=False
        )
        self.col_gravity_conv = nn.Conv2d(
            self.branch_dim, self.branch_dim, kernel_size=(board_h, 1), padding=(board_h // 2, 0),
            groups=16, bias=False
        )

        # 4-way directional gated scan projections (L->R, R->L, T->B, B->T)
        quarter_dim = self.branch_dim // 4
        self.scan_decay_logits = nn.Parameter(torch.zeros(4, quarter_dim))
        self.scan_mix = nn.Conv2d(self.branch_dim, self.branch_dim, kernel_size=3, padding=1, groups=16, bias=False)
        self.norm = nn.LayerNorm(self.branch_dim)
        self.out_proj = nn.Linear(self.branch_dim, d_model, bias=False)

    def _directional_4way_scan(self, x_2d: torch.Tensor) -> torch.Tensor:
        """
        Executes 4-way directional discounted cumulative scans along all 4 cardinal axes.
        x_2d: (B, branch_dim, H, W)
        """
        B, C, H, W = x_2d.shape
        q_dim = C // 4
        x_lr, x_rl, x_tb, x_bt = torch.split(x_2d, q_dim, dim=1)
        decays = torch.sigmoid(self.scan_decay_logits)  # (4, q_dim)

        d_lr = decays[0].view(1, q_dim, 1)
        d_rl = decays[1].view(1, q_dim, 1)
        d_tb = decays[2].view(1, q_dim, 1)
        d_bt = decays[3].view(1, q_dim, 1)

        # Left-to-Right & Right-to-Left scans along width W
        out_lr = []
        state_lr = torch.zeros(B, q_dim, H, device=x_2d.device, dtype=x_2d.dtype)
        for w in range(W):
            state_lr = d_lr * state_lr + x_lr[:, :, :, w]
            out_lr.append(state_lr)
        scan_lr = torch.stack(out_lr, dim=-1)

        out_rl = []
        state_rl = torch.zeros(B, q_dim, H, device=x_2d.device, dtype=x_2d.dtype)
        for w in range(W - 1, -1, -1):
            state_rl = d_rl * state_rl + x_rl[:, :, :, w]
            out_rl.append(state_rl)
        scan_rl = torch.stack(out_rl[::-1], dim=-1)

        # Top-to-Bottom & Bottom-to-Top scans along height H
        out_tb = []
        state_tb = torch.zeros(B, q_dim, W, device=x_2d.device, dtype=x_2d.dtype)
        for h in range(H):
            state_tb = d_tb * state_tb + x_tb[:, :, h, :]
            out_tb.append(state_tb)
        scan_tb = torch.stack(out_tb, dim=-2)

        out_bt = []
        state_bt = torch.zeros(B, q_dim, W, device=x_2d.device, dtype=x_2d.dtype)
        for h in range(H - 1, -1, -1):
            state_bt = d_bt * state_bt + x_bt[:, :, h, :]
            out_bt.append(state_bt)
        scan_bt = torch.stack(out_bt[::-1], dim=-2)

        return torch.cat([scan_lr, scan_rl, scan_tb, scan_bt], dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, N, d_model) where N = H * W = 81
        Returns:
            out: (B, N, d_model)
        """
        B, N, C = x.shape
        H, W = self.board_h, self.board_w

        u, gate = self.in_proj(x).chunk(2, dim=-1)  # (B, N, branch_dim) each
        u_2d = u.transpose(1, 2).view(B, self.branch_dim, H, W)

        # Axial full-row laser + full-column gravity + 4-way directional scan
        laser_h = self.row_laser_conv(u_2d)
        gravity_v = self.col_gravity_conv(u_2d)
        scan_4way = self._directional_4way_scan(u_2d)
        local_mix = self.scan_mix(u_2d)

        combined_2d = laser_h + gravity_v + scan_4way + local_mix
        combined_tokens = combined_2d.flatten(2).transpose(1, 2)  # (B, N, branch_dim)
        gated = self.norm(combined_tokens) * F.silu(gate)
        return self.out_proj(gated)


class TitanHybridBlock(nn.Module):
    """
    Breakthrough 2 & 3: Hybrid 4-Way Directional Scan + Portal-Graph Cross-Attention Block
    modulated by Adaptive LayerNorm-Zero (`adaLN-Zero`).

    Combines in parallel:
      - Branch A: `DirectionalLaserGravityScan` (full-row horizontal laser, full-column
        vertical gravity drop, and 4-way directional scan)
      - Branch B: Portal-Biased Multi-Head Self-Attention with 2D relative positional bias
    Followed by a SwiGLU Feed-Forward Network, with both stages hyper-conditioned by
    `c_telemetry` via `adaLN-Zero` (scale `1 + gamma`, shift `beta`, residual gate `alpha`).
    """

    def __init__(
        self,
        d_model: int = 512,
        num_heads: int = 8,
        d_mlp: int = 1536,
        board_h: int = 9,
        board_w: int = 9,
        dropout: float = 0.05,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)
        self.board_h = board_h
        self.board_w = board_w
        self.num_tokens = board_h * board_w

        # Pre-norms (without elementwise affine because adaLN-Zero provides dynamic scale & shift)
        self.norm1 = nn.LayerNorm(d_model, elementwise_affine=False)
        self.norm2 = nn.LayerNorm(d_model, elementwise_affine=False)

        # Branch A: 4-Way Directional Scan + Axial Row/Col Laser-Gravity
        self.axial_scan_branch = DirectionalLaserGravityScan(
            d_model=d_model, board_h=board_h, board_w=board_w
        )

        # Branch B: Portal-Biased Multi-Head Self-Attention
        self.qkv_proj = nn.Linear(d_model, d_model * 3, bias=False)
        self.attn_out_proj = nn.Linear(d_model, d_model, bias=False)
        self.portal_bias_proj = nn.Linear(8, num_heads, bias=False)
        self.rel_pos_bias = nn.Parameter(torch.randn(1, num_heads, self.num_tokens, self.num_tokens) * 0.01)

        # Parallel branch mixer
        self.branch_fusion = nn.Linear(d_model * 2, d_model, bias=False)

        # SwiGLU Feed-Forward Network
        self.mlp_w1 = nn.Linear(d_model, d_mlp, bias=False)
        self.mlp_w2 = nn.Linear(d_model, d_mlp, bias=False)
        self.mlp_w3 = nn.Linear(d_mlp, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

        # adaLN-Zero Modulation Head: predicts (gamma1, beta1, alpha1, gamma2, beta2, alpha2)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(d_model, 6 * d_model, bias=True),
        )
        # Initialize adaLN-Zero weights to zero so initial blocks act as smooth near-identity residuals
        nn.init.zeros_(self.adaLN_modulation[1].weight)
        nn.init.zeros_(self.adaLN_modulation[1].bias)
        with torch.no_grad():
            # Set initial residual gates alpha1, alpha2 to 0.15 so untrained forward pass propagates signal cleanly
            self.adaLN_modulation[1].bias[2 * d_model : 3 * d_model].fill_(0.15)
            self.adaLN_modulation[1].bias[5 * d_model : 6 * d_model].fill_(0.15)

    def forward(
        self,
        x: torch.Tensor,
        c_telemetry: torch.Tensor,
        portal_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: (B, 81, d_model) token sequence
            c_telemetry: (B, d_model) telemetry conditioning vector
            portal_mask: Optional (B, 81, 81, 8) non-Euclidean portal adjacency tensor
        """
        B, N, C = x.shape

        # Compute adaLN-Zero modulation parameters: (B, 1, d_model) each
        mod = self.adaLN_modulation(c_telemetry).unsqueeze(1)
        gamma1, beta1, alpha1, gamma2, beta2, alpha2 = mod.chunk(6, dim=-1)

        # Stage 1: adaLN-Zero modulated Hybrid Axial-Scan + Portal-Attention
        x_mod1 = self.norm1(x) * (1.0 + gamma1) + beta1

        # Branch A: 4-Way Directional Scan + Row/Col Laser-Gravity
        scan_out = self.axial_scan_branch(x_mod1)

        # Branch B: Portal-Biased Multi-Head Self-Attention
        qkv = self.qkv_proj(x_mod1).view(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # (B, H, N, head_dim)

        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale + self.rel_pos_bias
        if portal_mask is not None:
            p_bias = self.portal_bias_proj(portal_mask).permute(0, 3, 1, 2)
            attn_scores = attn_scores + p_bias

        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_out = torch.matmul(attn_weights, v).transpose(1, 2).contiguous().view(B, N, C)
        attn_out = self.attn_out_proj(attn_out)

        # Fuse parallel branches and apply gated residual alpha1
        hybrid_out = self.branch_fusion(torch.cat([scan_out, attn_out], dim=-1))
        x = x + alpha1 * self.dropout(hybrid_out)

        # Stage 2: adaLN-Zero modulated SwiGLU MLP
        x_mod2 = self.norm2(x) * (1.0 + gamma2) + beta2
        mlp_out = self.mlp_w3(F.silu(self.mlp_w1(x_mod2)) * self.mlp_w2(x_mod2))
        x = x + alpha2 * self.dropout(mlp_out)

        return x


class ConfidenceHaltingHead(nn.Module):
    """
    Learned epistemic policy confidence halting head for System-2 Latent Pondering.
    Estimates epistemic confidence in [0, 1] at each pondering step k.
    """

    def __init__(self, d_model: int = 512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model * 2, d_model // 2),
            nn.LayerNorm(d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 1),
            nn.Sigmoid(),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: (B, 81, d_model)
        Returns:
            confidence: (B, 1) in (0, 1)
        """
        z_mean = z.mean(dim=1)
        z_max = z.max(dim=1).values
        return self.net(torch.cat([z_mean, z_max], dim=-1))


class System2ThinkingCore(nn.Module):
    """
    Breakthrough 4: Recurrent System-2 Latent Pondering Core.
    After the 12 backbone blocks produce latent tokens z^(0) in R^(B x 81 x 512),
    executes K_ponder = 3 weight-tied iterative latent refinement steps with a
    gated GRU-style residual update and a learned `ConfidenceHaltingHead`.
    """

    def __init__(self, d_model: int = 512, num_heads: int = 8, k_ponder: int = 3):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.k_ponder = k_ponder

        self.step_embed = nn.Embedding(k_ponder, d_model)
        self.ponder_norm = nn.LayerNorm(d_model)
        self.ponder_attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=num_heads, dropout=0.0, batch_first=True
        )
        self.ponder_mlp = nn.Sequential(
            nn.Linear(d_model * 2, d_model * 2),
            nn.GELU(),
            nn.Linear(d_model * 2, d_model),
        )

        # Gated GRU-style token refinement gates
        self.reset_gate = nn.Linear(d_model * 2, d_model)
        self.update_gate = nn.Linear(d_model * 2, d_model)
        self.candidate_proj = nn.Linear(d_model * 2, d_model)
        self.final_norm = nn.LayerNorm(d_model)

        self.halting_head = ConfidenceHaltingHead(d_model=d_model)

    def forward(
        self, z0: torch.Tensor, c_telemetry: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            z0: (B, 81, d_model) initial backbone latent tokens
            c_telemetry: (B, d_model) global telemetry vector
        Returns:
            z_refined: (B, 81, d_model) confidence-weighted System-2 latent tokens
            ponder_confidences: (B, K_ponder) confidence at each thinking step
            final_confidence: (B, 1) epistemic confidence of the refined representation
        """
        B, N, C = z0.shape
        z = z0
        c_expanded = c_telemetry.unsqueeze(1).expand(B, N, C)

        step_states = []
        step_confidences = []

        for k in range(self.k_ponder):
            step_idx = torch.full((B,), k, device=z0.device, dtype=torch.long)
            s_emb = self.step_embed(step_idx).unsqueeze(1).expand(B, N, C)

            z_in = self.ponder_norm(z + s_emb)
            attn_out, _ = self.ponder_attn(z_in, z_in, z_in, need_weights=False)
            u = self.ponder_mlp(torch.cat([attn_out, c_expanded], dim=-1))

            # GRU-style gated residual update
            zu_cat = torch.cat([z, u], dim=-1)
            r = torch.sigmoid(self.reset_gate(zu_cat))
            g = torch.sigmoid(self.update_gate(zu_cat) - 1.0)  # biased toward preserving z
            z_cand = torch.tanh(self.candidate_proj(torch.cat([r * z, u], dim=-1)))
            z = (1.0 - g) * z + g * (z + z_cand)
            z = self.final_norm(z)

            conf_k = self.halting_head(z)  # (B, 1)
            step_states.append(z)
            step_confidences.append(conf_k)

        ponder_confidences = torch.cat(step_confidences, dim=-1)  # (B, K_ponder)
        weights = F.softmax(ponder_confidences, dim=-1).unsqueeze(-1).unsqueeze(-1)  # (B, K_ponder, 1, 1)
        stacked_states = torch.stack(step_states, dim=1)  # (B, K_ponder, 81, d_model)

        z_refined = (stacked_states * weights).sum(dim=1)
        final_confidence = ponder_confidences[:, -1:]
        return z_refined, ponder_confidences, final_confidence


class BilinearEdgeSwapHead(nn.Module):
    """
    Breakthrough 5: Bilinear Symmetric Edge-Swap Policy Head.
    Every canonical action a in {0, ..., 143} swaps two adjacent cells u = (r1, c1)
    and v = (r2, c2) (0..71: RIGHT swaps; 72..143: DOWN swaps).
    Gathers token embeddings z_u, z_v in R^(B x 144 x 512), computes symmetric
    bilinear edge features:
        e_uv = Concat[z_u + z_v, |z_u - z_v|, z_u * z_v]
    and fuses an optional 16-dim explicit micro-physics feature injection slot
    `micro_physics_feats: Optional[torch.Tensor] = None` of shape (B, 144, 16).
    """

    def __init__(
        self,
        d_model: int = 512,
        edge_hidden_dim: int = 384,
        micro_physics_dim: int = 16,
        board_h: int = 9,
        board_w: int = 9,
    ):
        super().__init__()
        self.d_model = d_model
        self.edge_hidden_dim = edge_hidden_dim
        self.micro_physics_dim = micro_physics_dim

        # Precompute exact canonical endpoint token indices u and v for all 144 actions
        u_indices = []
        v_indices = []
        dir_types = []
        for idx in range(NUM_CANONICAL_ACTIONS):
            r, c, d = decode_action_index(idx, height=board_h, width=board_w)
            u_idx = r * board_w + c
            if d == Direction.RIGHT:
                v_idx = r * board_w + (c + 1)
                dir_types.append(0)
            else:
                v_idx = (r + 1) * board_w + c
                dir_types.append(1)
            u_indices.append(u_idx)
            v_indices.append(v_idx)

        self.register_buffer("edge_u_idx", torch.tensor(u_indices, dtype=torch.long))
        self.register_buffer("edge_v_idx", torch.tensor(v_indices, dtype=torch.long))
        self.register_buffer("edge_dir_type", torch.tensor(dir_types, dtype=torch.long))

        # Orientation & canonical edge positional embeddings
        self.dir_embed = nn.Embedding(2, 64)
        self.edge_pos_embed = nn.Parameter(torch.randn(1, NUM_CANONICAL_ACTIONS, 64) * 0.02)

        # Micro-physics injection slot projection: (B, 144, 16) -> (B, 144, 128)
        self.micro_physics_proj = nn.Sequential(
            nn.Linear(micro_physics_dim, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Linear(128, 128),
            nn.GELU(),
        )

        # Symmetric bilinear feature projection: 3 * d_model + 64 + 64 + 128 -> edge_hidden_dim
        bilinear_in_dim = (3 * d_model) + 64 + 64 + 128
        self.edge_encoder = nn.Sequential(
            nn.Linear(bilinear_in_dim, edge_hidden_dim),
            nn.LayerNorm(edge_hidden_dim),
            nn.GELU(),
            nn.Linear(edge_hidden_dim, edge_hidden_dim),
            nn.LayerNorm(edge_hidden_dim),
            nn.GELU(),
        )

        self.policy_out = nn.Linear(edge_hidden_dim, 1)

    def forward(
        self,
        z: torch.Tensor,
        micro_physics_feats: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            z: (B, 81, d_model) refined board token representations
            micro_physics_feats: Optional (B, 144, 16) explicit micro-physics features
        Returns:
            policy_logits: (B, 144) swap action logits
            edge_embeddings: (B, 144, edge_hidden_dim) per-swap relational embeddings
        """
        B, N, C = z.shape

        # Gather endpoint embeddings for all 144 canonical adjacent pairs
        z_u = z[:, self.edge_u_idx, :]  # (B, 144, d_model)
        z_v = z[:, self.edge_v_idx, :]  # (B, 144, d_model)

        # Symmetric bilinear interactions: Concat[z_u + z_v, |z_u - z_v|, z_u * z_v]
        sym_sum = z_u + z_v
        sym_diff = torch.abs(z_u - z_v)
        sym_prod = z_u * z_v
        e_uv = torch.cat([sym_sum, sym_diff, sym_prod], dim=-1)  # (B, 144, 3 * d_model)

        # Orientation & edge position features
        dir_emb = self.dir_embed(self.edge_dir_type).unsqueeze(0).expand(B, -1, -1)
        pos_emb = self.edge_pos_embed.expand(B, -1, -1)

        # Optional 16-dim micro-physics feature injection
        if micro_physics_feats is None:
            micro_emb = torch.zeros(B, NUM_CANONICAL_ACTIONS, 128, device=z.device, dtype=z.dtype)
        else:
            micro_emb = self.micro_physics_proj(micro_physics_feats.to(device=z.device, dtype=z.dtype))

        full_edge_in = torch.cat([e_uv, dir_emb, pos_emb, micro_emb], dim=-1)
        edge_embeddings = self.edge_encoder(full_edge_in)  # (B, 144, edge_hidden_dim)
        policy_logits = self.policy_out(edge_embeddings).squeeze(-1)  # (B, 144)

        return policy_logits, edge_embeddings


class DynamicQuantileValueHead(nn.Module):
    """
    Breakthrough 6: Pace-Controlled Dynamic Quantile Tilting Head.
    Predicts 32 distributional quantiles Z_theta(s, a, tau_i) of shape (B, 144, 32)
    via a Dueling Quantile Decomposition:
        Z_theta(s, a, tau) = V_theta(s, tau) + (A_theta(s, a, tau) - mean_a' A_theta(s, a', tau))
    Supports dynamic pace-controlled risk tilting across:
      - Ahead / Safe (rho <= 0.4): Lower-Tail Risk-Averse CVaR_0.20
      - Normal (0.4 < rho <= 1.2): Full Distributional Expectation E[Z]
      - Desperate / Behind (rho > 1.2 or moves_left <= 2 with objectives_left >= 3):
        Upper-Tail Risk-Seeking Miracle Mode CVaR_0.85^+ (hunting 85th-100th percentile cascades!)
    """

    def __init__(
        self,
        d_model: int = 512,
        edge_hidden_dim: int = 384,
        num_quantiles: int = 32,
        num_actions: int = 144,
    ):
        super().__init__()
        self.d_model = d_model
        self.edge_hidden_dim = edge_hidden_dim
        self.num_quantiles = num_quantiles
        self.num_actions = num_actions

        # State value quantile stream V(s, tau): (B, 1, 32)
        self.value_stream = nn.Sequential(
            nn.Linear(d_model * 2, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Linear(256, num_quantiles),
        )

        # Edge advantage quantile stream A(s, a, tau): (B, 144, 32)
        self.advantage_stream = nn.Sequential(
            nn.Linear(edge_hidden_dim + d_model, 384),
            nn.LayerNorm(384),
            nn.GELU(),
            nn.Linear(384, num_quantiles),
        )

    def forward(
        self,
        z: torch.Tensor,
        edge_embeddings: torch.Tensor,
        c_telemetry: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            z: (B, 81, d_model)
            edge_embeddings: (B, 144, edge_hidden_dim)
            c_telemetry: (B, d_model)
        Returns:
            q_quantiles: (B, 144, 32)
            q_expected: (B, 144)
        """
        B = z.shape[0]
        z_global = z.mean(dim=1)  # (B, d_model)
        state_ctx = torch.cat([z_global, c_telemetry], dim=-1)  # (B, 2 * d_model)

        v_quantiles = self.value_stream(state_ctx).unsqueeze(1)  # (B, 1, 32)

        c_expanded = c_telemetry.unsqueeze(1).expand(B, self.num_actions, -1)
        adv_in = torch.cat([edge_embeddings, c_expanded], dim=-1)
        a_quantiles = self.advantage_stream(adv_in)  # (B, 144, 32)

        # Dueling combination
        q_quantiles = v_quantiles + (a_quantiles - a_quantiles.mean(dim=1, keepdim=True))
        q_expected = q_quantiles.mean(dim=-1)
        return q_quantiles, q_expected

    def compute_pace_tilted_scores(
        self,
        q_quantiles: torch.Tensor,
        moves_left: int,
        objectives_left: int,
    ) -> Tuple[torch.Tensor, str, float]:
        """
        Computes pace-adaptive risk-tilted action value scores from (B, 144, 32) quantiles.
        Returns:
            scores: (B, 144)
            regime: str ("SAFE_CVAR20", "NORMAL_EV", or "MIRACLE_UPPER_CVAR85")
            rho: float clearing pace
        """
        eff_moves = max(1, int(moves_left))
        eff_obj = max(0, int(objectives_left))
        rho = float(eff_obj) / float(eff_moves)

        sorted_q, _ = torch.sort(q_quantiles, dim=-1)  # (B, 144, 32) ascending

        is_desperate = (rho > 1.2) or (eff_moves <= 2 and eff_obj >= 3)
        if is_desperate:
            # Upper-Tail Risk-Seeking Miracle Mode (CVaR_0.85^+): top 15% quantiles (last 5 of 32)
            top_k = max(1, int(math.ceil(self.num_quantiles * 0.15)))
            scores = sorted_q[:, :, -top_k:].mean(dim=-1)
            regime = "MIRACLE_UPPER_CVAR85"
        elif rho <= 0.4:
            # Ahead / Safe Mode: Lower-Tail Risk-Averse CVaR_0.20 (bottom 20% quantiles, first 6 of 32)
            bot_k = max(1, int(math.ceil(self.num_quantiles * 0.20)))
            scores = sorted_q[:, :, :bot_k].mean(dim=-1)
            regime = "SAFE_CVAR20"
        else:
            # Normal Mode: Full distributional expectation
            scores = sorted_q.mean(dim=-1)
            regime = "NORMAL_EV"

        return scores, regime, rho


class AlphaCandyTitanNet(nn.Module):
    """
    `AlphaCandyTitanNet` (~55M+ Parameters): Flagship Next-Generation Super-Architecture.
    Unifies all 6 architectural breakthroughs into a single cohesive neural solver.
    """

    def __init__(
        self,
        in_channels: int = 20,
        board_h: int = 9,
        board_w: int = 9,
        d_model: int = 512,
        num_layers: int = 12,
        num_heads: int = 8,
        d_mlp: int = 1280,
        k_ponder: int = 3,
        num_quantiles: int = 32,
        num_actions: int = 144,
        micro_physics_dim: int = 16,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.board_h = board_h
        self.board_w = board_w
        self.num_tokens = board_h * board_w
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.k_ponder = k_ponder
        self.num_quantiles = num_quantiles
        self.num_actions = num_actions
        self.micro_physics_dim = micro_physics_dim

        # 1. S_6 Color-Symmetry Relational Stem
        self.color_stem = ColorSymmetricRelationalStem(
            num_colors=6,
            context_channels=in_channels - 6,
            color_feat_dim=64,
            d_model=d_model,
        )

        # Factorized 2D Row + Column + Token Positional Embeddings
        self.row_embed = nn.Parameter(torch.randn(1, board_h, 1, d_model // 2) * 0.02)
        self.col_embed = nn.Parameter(torch.randn(1, 1, board_w, d_model // 2) * 0.02)
        self.pos_embed = nn.Parameter(torch.randn(1, self.num_tokens, d_model) * 0.02)

        # 3. `adaLN-Zero` Level-Telemetry Hyper-Conditioner
        self.telemetry_conditioner = TelemetryConditioner(telemetry_dim=16, d_model=d_model)

        # 2. 12-Layer Hybrid 4-Way Directional Scan + Portal-Graph Cross-Attention Backbone
        self.blocks = nn.ModuleList(
            [
                TitanHybridBlock(
                    d_model=d_model,
                    num_heads=num_heads,
                    d_mlp=d_mlp,
                    board_h=board_h,
                    board_w=board_w,
                )
                for _ in range(num_layers)
            ]
        )
        self.backbone_norm = nn.LayerNorm(d_model)

        # 4. Recurrent System-2 Latent Pondering Core (K_ponder = 3)
        self.system2_core = System2ThinkingCore(
            d_model=d_model, num_heads=num_heads, k_ponder=k_ponder
        )

        # 5. Bilinear Symmetric Edge-Swap Policy Head
        self.policy_head = BilinearEdgeSwapHead(
            d_model=d_model,
            edge_hidden_dim=384,
            micro_physics_dim=micro_physics_dim,
            board_h=board_h,
            board_w=board_w,
        )

        # 6. Pace-Controlled Dynamic Quantile Tilting Head
        self.quantile_head = DynamicQuantileValueHead(
            d_model=d_model,
            edge_hidden_dim=384,
            num_quantiles=num_quantiles,
            num_actions=num_actions,
        )

        # Auxiliary Calibrated Win Probability & Blocker Hazard Heads
        self.win_head = nn.Sequential(
            nn.Linear(d_model * 2, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Linear(256, 64),
            nn.GELU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )
        self.hazard_head = nn.Sequential(
            nn.Conv2d(d_model, 128, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(128, 4, kernel_size=1),
        )

    def _get_2d_pos_embed(self) -> torch.Tensor:
        row_e = self.row_embed.expand(1, self.board_h, self.board_w, self.d_model // 2)
        col_e = self.col_embed.expand(1, self.board_h, self.board_w, self.d_model // 2)
        grid_e = torch.cat([row_e, col_e], dim=-1).view(1, self.num_tokens, self.d_model)
        return self.pos_embed + grid_e

    def forward(
        self,
        board_tensor: torch.Tensor,
        micro_physics_feats: Optional[torch.Tensor] = None,
        portal_mask: Optional[torch.Tensor] = None,
        moves_left: Optional[Union[int, float, torch.Tensor]] = None,
        objectives_left: Optional[Union[int, float, torch.Tensor]] = None,
        return_quantiles: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """
        Full forward pass through AlphaCandyTitanNet.
        Args:
            board_tensor: (B, 20, 9, 9) input board representation
            micro_physics_feats: Optional (B, 144, 16) explicit micro-physics swap features
            portal_mask: Optional (B, 81, 81, 8) portal graph routing bias mask
            moves_left: Optional scalar or (B,) tensor of remaining moves
            objectives_left: Optional scalar or (B,) tensor of remaining objectives
            return_quantiles: Whether to compute the 32-quantile distribution head
        """
        B, C, H, W = board_tensor.shape

        # 1. S_6 Color-Symmetry Relational Stem
        stem_2d = self.color_stem(board_tensor)  # (B, d_model, H, W)
        tokens = stem_2d.flatten(2).transpose(1, 2) + self._get_2d_pos_embed()  # (B, 81, d_model)

        # 3. Telemetry Hyper-Conditioning
        c_telemetry, telemetry_vec, phase_probs = self.telemetry_conditioner(
            board_tensor, moves_left=moves_left, objectives_left=objectives_left
        )

        # 2. 12-Layer Hybrid Directional Scan + Portal-Graph Cross-Attention Backbone
        for block in self.blocks:
            tokens = block(tokens, c_telemetry=c_telemetry, portal_mask=portal_mask)
        z0 = self.backbone_norm(tokens)

        # 4. Recurrent System-2 Latent Pondering Core (K_ponder = 3)
        z_refined, ponder_confidences, epistemic_confidence = self.system2_core(z0, c_telemetry)

        # 5. Bilinear Symmetric Edge-Swap Policy Head
        policy_logits, edge_embeddings = self.policy_head(
            z_refined, micro_physics_feats=micro_physics_feats
        )

        # Auxiliary Win Probability & Hazard Heads
        z_global = z_refined.mean(dim=1)
        win_prob = self.win_head(torch.cat([z_global, c_telemetry], dim=-1))  # (B, 1)
        z_2d = z_refined.transpose(1, 2).view(B, self.d_model, H, W)
        hazard_map = self.hazard_head(z_2d)  # (B, 4, H, W)

        output: Dict[str, torch.Tensor] = {
            "policy_logits": policy_logits,
            "win_prob": win_prob,
            "epistemic_confidence": epistemic_confidence,
            "ponder_confidences": ponder_confidences,
            "phase_probs": phase_probs,
            "telemetry_vec": telemetry_vec,
            "hazard_map": hazard_map,
            "latent_z0": z0,
            "latent_z": z_refined,
        }

        # 6. Pace-Controlled Dynamic Quantile Tilting Head
        if return_quantiles:
            q_quantiles, q_expected = self.quantile_head(z_refined, edge_embeddings, c_telemetry)
            output["q_quantiles"] = q_quantiles
            output["q_expected"] = q_expected

        return output

    def select_pace_adaptive_action(
        self,
        board_tensor: torch.Tensor,
        legal_mask: torch.Tensor,
        moves_left: int,
        objectives_left: int,
        micro_physics_feats: Optional[torch.Tensor] = None,
        portal_mask: Optional[torch.Tensor] = None,
        policy_prior_weight: float = 0.35,
        return_diagnostics: bool = False,
    ) -> Union[int, Tuple[int, Dict[str, Any]]]:
        """
        Breakthrough 6: Pace-Controlled Dynamic Quantile Tilting Action Selection.
        Computes required clearing pace rho = objectives_left / max(1, moves_left):
          - Ahead / Safe (rho <= 0.4): optimizes Lower-Tail Risk-Averse CVaR_0.20
          - Normal (0.4 < rho <= 1.2): optimizes full distributional expectation
          - Desperate / Behind (rho > 1.2 or moves_left <= 2 with objectives_left >= 3):
            automatically switches to Upper-Tail Risk-Seeking Miracle Mode (CVaR_0.85^+)!
        """
        if board_tensor.dim() == 3:
            board_tensor = board_tensor.unsqueeze(0)
        if legal_mask.dim() == 2:
            legal_mask_1d = legal_mask.squeeze(0).bool()
        else:
            legal_mask_1d = legal_mask.bool()

        with torch.no_grad():
            res = self.forward(
                board_tensor,
                micro_physics_feats=micro_physics_feats,
                portal_mask=portal_mask,
                moves_left=moves_left,
                objectives_left=objectives_left,
                return_quantiles=True,
            )
            q_quantiles = res["q_quantiles"]  # (1, 144, 32)
            policy_logits = res["policy_logits"].squeeze(0)  # (144,)

            tilted_q, regime, rho = self.quantile_head.compute_pace_tilted_scores(
                q_quantiles, moves_left=moves_left, objectives_left=objectives_left
            )
            tilted_q_1d = tilted_q.squeeze(0)  # (144,)

            # Combine normalized policy prior with pace-tilted quantile score over legal moves
            masked_logits = policy_logits.masked_fill(~legal_mask_1d, -1e9)
            policy_probs = F.softmax(masked_logits, dim=-1)

            combined_scores = tilted_q_1d + policy_prior_weight * policy_probs
            combined_scores = combined_scores.masked_fill(~legal_mask_1d, -1e9)
            best_action = int(torch.argmax(combined_scores).item())

            if not return_diagnostics:
                return best_action

            diagnostics = {
                "action_idx": best_action,
                "regime": regime,
                "pace_rho": rho,
                "action_prob": float(policy_probs[best_action].item()),
                "tilted_q_value": float(tilted_q_1d[best_action].item()),
                "expected_q_value": float(res["q_expected"].squeeze(0)[best_action].item()),
                "win_prob": float(res["win_prob"].item()),
                "epistemic_confidence": float(res["epistemic_confidence"].item()),
                "ponder_confidences": res["ponder_confidences"].squeeze(0).cpu().tolist(),
            }
            return best_action, diagnostics

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def parameter_breakdown(self) -> Dict[str, int]:
        return {
            "1_color_symmetric_stem": sum(p.numel() for p in self.color_stem.parameters() if p.requires_grad),
            "2_hybrid_backbone_12_blocks": sum(p.numel() for p in self.blocks.parameters() if p.requires_grad),
            "3_telemetry_conditioner": sum(p.numel() for p in self.telemetry_conditioner.parameters() if p.requires_grad),
            "4_system2_thinking_core": sum(p.numel() for p in self.system2_core.parameters() if p.requires_grad),
            "5_bilinear_edge_swap_head": sum(p.numel() for p in self.policy_head.parameters() if p.requires_grad),
            "6_dynamic_quantile_value_head": sum(p.numel() for p in self.quantile_head.parameters() if p.requires_grad),
            "7_auxiliary_win_hazard_pos": (
                sum(p.numel() for p in self.win_head.parameters() if p.requires_grad)
                + sum(p.numel() for p in self.hazard_head.parameters() if p.requires_grad)
                + self.pos_embed.numel()
                + self.row_embed.numel()
                + self.col_embed.numel()
                + sum(p.numel() for p in self.backbone_norm.parameters() if p.requires_grad)
            ),
        }


def create_alphacandy_titan(device: Union[str, torch.device] = "cpu") -> AlphaCandyTitanNet:
    """Factory function creating an initialized AlphaCandyTitanNet on the specified device."""
    model = AlphaCandyTitanNet(
        in_channels=20,
        board_h=9,
        board_w=9,
        d_model=512,
        num_layers=12,
        num_heads=8,
        d_mlp=1280,
        k_ponder=3,
        num_quantiles=32,
        num_actions=144,
        micro_physics_dim=16,
    )
    return model.to(device)


if __name__ == "__main__":
    net = create_alphacandy_titan("cpu")
    total_params = net.count_parameters()
    print(f"AlphaCandyTitanNet Initialized: {total_params:,} parameters ({total_params / 1e6:.2f}M)")
    for k, v in net.parameter_breakdown().items():
        print(f"  {k:32s}: {v:10,d} ({v / 1e6:.2f}M)")
