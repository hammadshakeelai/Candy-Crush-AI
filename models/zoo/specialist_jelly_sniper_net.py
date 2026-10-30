"""
AlphaCandy Model Zoo — Part 4.2: Jelly Corner Sniper Network (`JellyCornerSniperNet`).
======================================================================================
Ultra-specialized architecture for JELLY levels with isolated corners, cutout boundaries,
and detached island chambers (such as stubborn Levels 201, 235, 279, and 300).

Architectural Innovations:
  1. Static + Dynamic Degree-of-Freedom (DoF) Friction Mask (`DegreeOfFreedomFrictionAnalyzer`):
     - Explicitly weights board corners (0,0), (0,8), (8,0), (8,8) and cutout/void boundaries
       3.0x higher than interior center cells.
     - Dynamically computes the local match-3 degree of freedom d(r, c) in {0..4} from the
       active board mask (channel 18) and blocker barriers (channels 10..14), identifying
       isolated jellies (channels 15 & 16) that cannot be cleared by local matches.
  2. Cross-Hair Laser Intersection Attention (`CrossHairLaserIntersectionBlock`):
     - Axial row-and-column cross-hair attention that aligns horizontal Striped candies
       (channel 6) and vertical Striped candies (channel 7) — plus 1-hop gravity-shifted
       striped candidates — with remote high-friction corner and island jellies.
  3. Remote Laser Trigger & Direct Corner Sniper Action Head:
     - Scores all 144 canonical swaps by combining local high-friction jelly clears with
       long-range cross-hair laser firing scores across rows and columns.
"""

import math
from typing import Dict, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class DegreeOfFreedomFrictionAnalyzer(nn.Module):
    """
    Computes the Static + Dynamic Degree-of-Freedom (DoF) Friction Mask M_friction in R^{B x 9 x 9}
    and the High-Friction Jelly Sniper Target Map M_sniper in R^{B x 9 x 9}.
      - Corners (0,0), (0,8), (8,0), (8,8) are weighted 3.0x higher than center cells (1.0x).
      - Cutout boundaries (cells with <= 2 active orthogonal neighbors) receive up to 3.0x multiplier.
      - Double-layer jellies (channel 16) receive 2.0x thickness weight over single-layer (channel 15).
    """

    def __init__(self, height: int = 9, width: int = 9, d_out: int = 48):
        super().__init__()
        self.height = height
        self.width = width

        # Construct static geometric friction prior:
        # Center cells = 1.0, Edges = 1.85, Corners (0,0),(0,8),(8,0),(8,8) = 3.0
        static_friction = torch.ones(1, 1, height, width, dtype=torch.float32)
        static_friction[:, :, 0, :] = 1.85
        static_friction[:, :, height - 1, :] = 1.85
        static_friction[:, :, :, 0] = 1.85
        static_friction[:, :, :, width - 1] = 1.85

        for cr in (0, height - 1):
            for cc in (0, width - 1):
                static_friction[:, :, cr, cc] = 3.0

        # 4-neighbor orthogonal degree kernel
        dof_kernel = torch.tensor(
            [[[[0.0, 1.0, 0.0],
               [1.0, 0.0, 1.0],
               [0.0, 1.0, 0.0]]]],
            dtype=torch.float32
        )

        self.register_buffer("static_friction", static_friction)
        self.register_buffer("dof_kernel", dof_kernel)

        self.proj = nn.Sequential(
            nn.Conv2d(8, d_out, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(d_out),
            nn.GELU(),
            nn.Conv2d(d_out, d_out, kernel_size=1),
            nn.GELU()
        )

    def forward(
        self, board_tensor: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            board_tensor: (B, 20, 9, 9)
        Returns:
            friction_feat: (B, d_out, 9, 9)
            dof_friction_mask: (B, 9, 9) static + dynamic friction map (3x at corners & cutouts)
            jelly_priority_map: (B, 9, 9) friction-weighted jelly target map
            crosshair_demand_map: (B, 9, 9) row + col remote sniper demand induced by high-friction jellies
        """
        B, C, H, W = board_tensor.shape

        # Active cell mask (channel 18); fallback to ones if all-zero dummy tensor
        active_raw = board_tensor[:, 18:19]  # (B, 1, 9, 9)
        has_active = (active_raw.sum(dim=(2, 3), keepdim=True) > 0).float()
        active_grid = active_raw * has_active + (1.0 - has_active)

        # Unblocked playable mask (active and not blocked by immobile chocolate/frosting/licorice)
        blocker_present = board_tensor[:, 14:15].clamp(0.0, 1.0)
        free_cell = (active_grid * (1.0 - 0.65 * blocker_present)).clamp(0.0, 1.0)

        # Count local 4-neighbor degrees of freedom in [0, 4]
        active_neighbors = F.conv2d(active_grid, self.dof_kernel, padding=1)  # (B, 1, 9, 9)
        free_neighbors = F.conv2d(free_cell, self.dof_kernel, padding=1)      # (B, 1, 9, 9)

        # Cutout boundary penalty: cells with <=2 active neighbors get scaled up to 3.0x
        cutout_deficit = (4.0 - active_neighbors).clamp(0.0, 4.0) / 2.0  # 1.0 when 2 neighbors missing
        blocker_deficit = (4.0 - free_neighbors).clamp(0.0, 4.0) / 4.0

        # Combine static (3.0x corner prior) + dynamic cutout/blocker restriction -> [1.0, 3.0+]
        dof_friction = torch.maximum(
            self.static_friction.expand(B, 1, H, W),
            1.0 + 2.0 * cutout_deficit.clamp(0.0, 1.0)
        ) + 0.75 * blocker_deficit
        dof_friction = dof_friction * active_grid  # Zero out void holes

        # Jelly thickness map: single-layer (ch 15) + 2x double-layer (ch 16)
        jelly_single = board_tensor[:, 15:16]
        jelly_double = board_tensor[:, 16:17]
        jelly_thickness = jelly_single + 2.0 * jelly_double  # (B, 1, 9, 9)

        # High-friction Jelly Sniper Priority Map
        jelly_priority = jelly_thickness * dof_friction  # (B, 1, 9, 9)

        # Cross-hair sniper demand: broadcast row-max and col-max jelly priority across lines of fire
        row_demand = jelly_priority.max(dim=3, keepdim=True).values.expand(B, 1, H, W)
        col_demand = jelly_priority.max(dim=2, keepdim=True).values.expand(B, 1, H, W)
        crosshair_demand = 0.5 * (row_demand + col_demand)

        # Striped H (ch 6), Striped V (ch 7), Wrapped (ch 8)
        striped_h = board_tensor[:, 6:7]
        striped_v = board_tensor[:, 7:8]

        feat_stack = torch.cat([
            dof_friction,
            jelly_thickness,
            jelly_priority,
            row_demand,
            col_demand,
            striped_h * row_demand,
            striped_v * col_demand,
            free_neighbors / 4.0
        ], dim=1)  # (B, 8, 9, 9)

        friction_feat = self.proj(feat_stack)
        return (
            friction_feat,
            dof_friction.squeeze(1),
            jelly_priority.squeeze(1),
            crosshair_demand.squeeze(1)
        )


class CrossHairLaserIntersectionBlock(nn.Module):
    """
    Cross-Hair Laser Intersection Attention Block.
    Computes axial Row-Laser Attention + Column-Laser Attention + 1-to-2 row Gravity Drop
    Alignment so horizontal/vertical Striped candies (and candidate striped formations)
    directly exchange sniper targeting signals with remote high-friction corner/island jellies.
    """

    def __init__(self, d_model: int = 192, num_heads: int = 6, height: int = 9, width: int = 9):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)
        self.height = height
        self.width = width

        # Construct static cross-hair mask on 81x81 grid:
        # crosshair_mask[u, v] = 1 if u and v share the same row OR the same column
        N = height * width
        row_share = torch.zeros(N, N, dtype=torch.float32)
        col_share = torch.zeros(N, N, dtype=torch.float32)
        drop1_row_share = torch.zeros(N, N, dtype=torch.float32)
        for r1 in range(height):
            for c1 in range(width):
                u = r1 * width + c1
                for r2 in range(height):
                    for c2 in range(width):
                        v = r2 * width + c2
                        if r1 == r2:
                            row_share[u, v] = 1.0
                        if c1 == c2:
                            col_share[u, v] = 1.0
                        # If a horizontal striped candy at (r1, c1) drops 1 or 2 rows to r2, it snipes (r2, c2)
                        if r2 in (r1 + 1, r1 + 2):
                            drop1_row_share[u, v] = 1.0

        self.register_buffer("row_share", row_share)
        self.register_buffer("col_share", col_share)
        self.register_buffer("drop1_row_share", drop1_row_share)

        self.norm1 = nn.LayerNorm(d_model)
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model)

        # Learnable head-specific weights for [row_laser, col_laser, drop_row_laser, target_jelly_priority]
        self.crosshair_bias_proj = nn.Linear(4, num_heads, bias=False)

        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 3),
            nn.GELU(),
            nn.Linear(d_model * 3, d_model)
        )

    def forward(
        self,
        tokens: torch.Tensor,
        jelly_priority_map: torch.Tensor,
        striped_h_map: torch.Tensor,
        striped_v_map: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            tokens: (B, 81, d_model)
            jelly_priority_map: (B, 9, 9) high-friction jelly target weights
            striped_h_map: (B, 9, 9) horizontal striped candy map
            striped_v_map: (B, 9, 9) vertical striped candy map
        Returns:
            tokens_out: (B, 81, d_model)
            laser_hit_intensity: (B, 9, 9) cross-hair laser attention received by each cell
        """
        B, N, D = tokens.shape
        H_heads = self.num_heads
        Dh = self.head_dim

        x_norm = self.norm1(tokens)
        q = self.q_proj(x_norm).view(B, N, H_heads, Dh).transpose(1, 2)  # (B, H, 81, Dh)
        k = self.k_proj(x_norm).view(B, N, H_heads, Dh).transpose(1, 2)
        v = self.v_proj(x_norm).view(B, N, H_heads, Dh).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # (B, H, 81, 81)

        # Construct dynamic cross-hair laser bias tensor (B, 81, 81, 4)
        jp_flat = jelly_priority_map.view(B, 1, N).expand(B, N, N)         # Target v's jelly friction priority
        sh_flat = striped_h_map.view(B, N, 1).expand(B, N, N)              # Source u has Striped H
        sv_flat = striped_v_map.view(B, N, 1).expand(B, N, N)              # Source u has Striped V

        row_laser = self.row_share.unsqueeze(0) * (0.5 + sh_flat) * (0.2 + jp_flat)
        col_laser = self.col_share.unsqueeze(0) * (0.5 + sv_flat) * (0.2 + jp_flat)
        drop_laser = self.drop1_row_share.unsqueeze(0) * sh_flat * jp_flat

        cross_feats = torch.stack([row_laser, col_laser, drop_laser, jp_flat], dim=-1)  # (B, 81, 81, 4)
        cross_bias = self.crosshair_bias_proj(cross_feats).permute(0, 3, 1, 2)  # (B, H, 81, 81)

        attn = F.softmax(scores + cross_bias, dim=-1)
        out = torch.matmul(attn, v).transpose(1, 2).contiguous().view(B, N, D)
        tokens = tokens + self.out_proj(out)
        tokens = tokens + self.ffn(self.norm2(tokens))

        # Measure cross-hair laser attention intersecting each cell
        cross_mask = (self.row_share + self.col_share).clamp(0.0, 1.0).view(1, 1, N, N)
        laser_hit_intensity = (attn * cross_mask).mean(dim=(1, 2)).view(B, self.height, self.width)
        return tokens, laser_hit_intensity


class JellyCornerSniperNet(nn.Module):
    """
    JellyCornerSniperNet: Specialized Deep Architecture for JELLY Levels with Isolated Corners & Cutouts.
    Combines a 3x Corner/Cutout Degree-of-Freedom Friction Mask with Cross-Hair Laser Intersection
    Attention to snipe hard-to-reach jellies via remote Striped/Wrapped combos and direct corner clears.
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

        # 1. Static + Dynamic Degree-of-Freedom (DoF) Friction Analyzer
        dof_dim = 48
        self.friction_analyzer = DegreeOfFreedomFrictionAnalyzer(
            height=board_h, width=board_w, d_out=dof_dim
        )

        # 2. Convolutional Stem + Coordinate Embedding
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels + dof_dim, d_model, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(d_model),
            nn.GELU(),
            nn.Conv2d(d_model, d_model, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(d_model),
            nn.GELU()
        )
        self.pos_embed = nn.Parameter(torch.randn(1, self.num_nodes, d_model) * 0.02)

        # 3. 6-Layer Cross-Hair Laser Intersection Attention Backbone
        self.layers = nn.ModuleList([
            CrossHairLaserIntersectionBlock(
                d_model=d_model, num_heads=num_heads, height=board_h, width=board_w
            )
            for _ in range(num_layers)
        ])
        self.final_norm = nn.LayerNorm(d_model)

        # 4. Sniper Swap Scorer (72 horizontal + 72 vertical swaps weighted by DoF friction & laser alignment)
        self.horiz_sniper_scorer = nn.Sequential(
            nn.Linear(d_model * 2 + 4, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1)
        )
        self.vert_sniper_scorer = nn.Sequential(
            nn.Linear(d_model * 2 + 4, d_model),
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

        # 5. Corner Isolation Severity, Value & Win Probability Heads
        self.isolation_head = nn.Sequential(
            nn.Linear(d_model * 2 + 2, 64),
            nn.GELU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        self.value_head = nn.Sequential(
            nn.Linear(d_model * 2 + 2, 128),
            nn.GELU(),
            nn.Linear(128, 1),
            nn.Tanh()
        )
        self.win_head = nn.Sequential(
            nn.Linear(d_model * 2 + 2, 128),
            nn.GELU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def _score_sniper_swaps(
        self,
        z_2d: torch.Tensor,
        jelly_priority_map: torch.Tensor,
        crosshair_demand_map: torch.Tensor
    ) -> torch.Tensor:
        B, D, H, W = z_2d.shape

        # Horizontal swaps (9 x 8 = 72)
        h_l = z_2d[:, :, :, :-1].permute(0, 2, 3, 1)
        h_r = z_2d[:, :, :, 1:].permute(0, 2, 3, 1)
        jp_l = jelly_priority_map[:, :, :-1].unsqueeze(-1)
        jp_r = jelly_priority_map[:, :, 1:].unsqueeze(-1)
        cd_l = crosshair_demand_map[:, :, :-1].unsqueeze(-1)
        cd_r = crosshair_demand_map[:, :, 1:].unsqueeze(-1)
        h_in = torch.cat([h_l, h_r, jp_l, jp_r, cd_l, cd_r], dim=-1).reshape(B, 72, 2 * D + 4)
        h_logits = self.horiz_sniper_scorer(h_in).squeeze(-1)

        # Vertical swaps (8 x 9 = 72)
        v_t = z_2d[:, :, :-1, :].permute(0, 2, 3, 1)
        v_b = z_2d[:, :, 1:, :].permute(0, 2, 3, 1)
        jp_t = jelly_priority_map[:, :-1, :].unsqueeze(-1)
        jp_b = jelly_priority_map[:, 1:, :].unsqueeze(-1)
        cd_t = crosshair_demand_map[:, :-1, :].unsqueeze(-1)
        cd_b = crosshair_demand_map[:, 1:, :].unsqueeze(-1)
        v_in = torch.cat([v_t, v_b, jp_t, jp_b, cd_t, cd_b], dim=-1).reshape(B, 72, 2 * D + 4)
        v_logits = self.vert_sniper_scorer(v_in).squeeze(-1)

        return torch.cat([h_logits, v_logits], dim=-1)  # (B, 144)

    def forward(
        self,
        board_tensor: torch.Tensor,
        legal_mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        B, C, H, W = board_tensor.shape

        # Step 1: Compute Static + Dynamic Degree-of-Freedom Friction Mask & Jelly Sniper Priority Map
        friction_feat, dof_friction_mask, jelly_priority_map, crosshair_demand_map = (
            self.friction_analyzer(board_tensor)
        )

        # Step 2: Stem encoding
        stem_2d = self.stem(torch.cat([board_tensor, friction_feat], dim=1))
        tokens = stem_2d.flatten(2).transpose(1, 2) + self.pos_embed  # (B, 81, d_model)

        # Step 3: Cross-Hair Laser Intersection Attention Layers
        striped_h_map = board_tensor[:, 6]
        striped_v_map = board_tensor[:, 7]
        crosshair_laser_map = None
        for layer in self.layers:
            tokens, crosshair_laser_map = layer(
                tokens, jelly_priority_map, striped_h_map, striped_v_map
            )

        tokens = self.final_norm(tokens)
        z_2d = tokens.transpose(1, 2).view(B, self.d_model, H, W)

        # Step 4: Policy logits (B, 144) combining Sniper Swap Scorer + Global Conv Head
        sniper_swap_logits = self._score_sniper_swaps(z_2d, jelly_priority_map, crosshair_demand_map)
        policy_logits = sniper_swap_logits + self.global_policy_head(z_2d)

        if legal_mask is not None:
            policy_logits = policy_logits.masked_fill(~legal_mask.bool(), -1e9)

        # Step 5: Global Readout weighted by high-friction jelly priority
        mean_pool = tokens.mean(dim=1)
        max_pool = tokens.max(dim=1).values
        max_jelly_friction = jelly_priority_map.flatten(1).max(dim=-1, keepdim=True).values
        total_jelly_friction = jelly_priority_map.flatten(1).mean(dim=-1, keepdim=True)
        readout = torch.cat([mean_pool, max_pool, max_jelly_friction, total_jelly_friction], dim=-1)

        corner_isolation_severity = self.isolation_head(readout)
        value = self.value_head(readout)
        win_prob = self.win_head(readout)

        return {
            "policy_logits": policy_logits,
            "value": value,
            "win_prob": win_prob,
            "dof_friction_mask": dof_friction_mask,
            "jelly_sniper_priority_map": jelly_priority_map,
            "crosshair_laser_map": crosshair_laser_map,
            "corner_isolation_severity": corner_isolation_severity,
        }

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def create_jelly_corner_sniper(device: str = "cpu") -> JellyCornerSniperNet:
    model = JellyCornerSniperNet()
    return model.to(device)


if __name__ == "__main__":
    torch.manual_seed(42)
    model = create_jelly_corner_sniper("cpu")
    params = model.count_parameters()
    print(f"[JellyCornerSniperNet] Trainable Parameters: {params:,} ({params / 1e6:.2f}M)")

    dummy_board = torch.zeros(2, 20, 9, 9)
    dummy_board[:, 18] = 1.0  # Active grid
    dummy_board[0, 16, 0, 0] = 1.0  # Double jelly at top-left corner (0, 0)
    dummy_board[0, 15, 8, 8] = 1.0  # Single jelly at bottom-right corner (8, 8)
    dummy_board[0, 6, 0, 5] = 1.0   # Horizontal Striped candy aligned with row 0 corner jelly!

    out = model(dummy_board)
    for k, v in out.items():
        print(f"  {k}: {tuple(v.shape)}")
    print(f"  Corner (0,0) vs Center (4,4) friction ratio: "
          f"{out['dof_friction_mask'][0, 0, 0].item():.2f} vs {out['dof_friction_mask'][0, 4, 4].item():.2f}")
