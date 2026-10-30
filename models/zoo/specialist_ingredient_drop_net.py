"""
AlphaCandy Model Zoo — Part 4.1: Ingredient Drop Specialist Network (`IngredientDropSpecialistNet`).
====================================================================================================
Ultra-specialized architecture for INGREDIENTS (Drop) levels (such as stubborn Levels 148 and 272).

Architectural Innovations:
  1. Explicit Below-Ingredient Column Obstruction Analyzer:
     - Tracks ingredient coordinates from channel 15 (with seamless support for channel 17)
       and computes the downward cumulative blocker & cutout obstruction density strictly below
       each ingredient toward the bottom exit row (r = 8).
  2. Column-Flow Vertical Attention (`ColumnFlowVerticalAttentionBlock`):
     - Factorized Intra-Column Gravity Attention (along the 9 vertical cells of each drop chute)
       coupled with Cross-Column Lane-Transfer Attention, biased by downward gravity flow and
       ingredient-to-exit path visibility.
  3. Lateral Lane-Shift & Vertical Sink Scorer (`LateralLaneShiftAndSinkScorer`):
     - Explicitly scores all 72 horizontal swaps (r, c) <-> (r, c+1) for shifting trapped
       ingredients out of blocked columns into unobstructed drop lanes, and scores all 72
       vertical swaps (r, c) <-> (r+1, c) for sinking ingredients directly toward the exit row
       or firing vertical column-clearing cascades beneath ingredients.
"""

import math
from typing import Dict, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class ColumnObstructionAnalyzer(nn.Module):
    """
    Computes explicit geometric and physical drop-path features for Ingredients levels:
      - Ingredient presence map (from channel 15 and channel 17)
      - Cumulative blocker obstruction strictly below each cell (r' > r, c)
      - Column obstruction differential Delta_Obs(c, c+1) for lateral lane shifting
      - Distance-to-exit harmonic potential toward row H-1
    """

    def __init__(self, height: int = 9, width: int = 9, d_out: int = 48):
        super().__init__()
        self.height = height
        self.width = width

        # Strict lower-column summation matrix: M_below[r, r'] = 1 iff r' > r
        below_mask = torch.triu(torch.ones(height, height, dtype=torch.float32), diagonal=1)
        # Include self and below: M_self_below[r, r'] = 1 iff r' >= r
        self_below_mask = torch.triu(torch.ones(height, height, dtype=torch.float32), diagonal=0)

        # Normalized vertical depth coordinate (0.0 at top row 0 -> 1.0 at bottom exit row 8)
        row_depth = torch.linspace(0.0, 1.0, steps=height).view(1, 1, height, 1).expand(1, 1, height, width)

        self.register_buffer("below_mask", below_mask)
        self.register_buffer("self_below_mask", self_below_mask)
        self.register_buffer("row_depth", row_depth)

        # Project 8 physics-derived drop feature planes into d_out channels
        self.physics_proj = nn.Sequential(
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
            board_tensor: (B, 20, 9, 9) canonical board state tensor
        Returns:
            physics_feat: (B, d_out, 9, 9) encoded drop-physics feature planes
            ing_map: (B, 9, 9) active ingredient coordinates
            col_obstruction: (B, 9) total obstruction density below ingredients per column
            below_hazard_map: (B, 9, 9) cumulative obstruction density below every cell (r, c)
        """
        B, C, H, W = board_tensor.shape

        # Track ingredient coordinates explicitly from channel 15 (and channel 17 fallback)
        ing_ch15 = board_tensor[:, 15]
        ing_ch17 = board_tensor[:, 17] if C > 17 else torch.zeros_like(ing_ch15)
        # If channel 17 has sparse ingredient markers or channel 15 is used directly:
        ing_map = torch.clamp(ing_ch15 + ing_ch17, 0.0, 1.0)  # (B, 9, 9)

        # Active cell mask (channel 18); treat all ones if uninitialized dummy tensor
        active_raw = board_tensor[:, 18]
        has_active = (active_raw.sum(dim=(1, 2), keepdim=True) > 0).float()
        active_grid = active_raw * has_active + (1.0 - has_active)
        void_cutouts = (1.0 - active_grid).clamp(0.0, 1.0)

        # Blocker density per cell (channels 10..14: chocolate, licorice, frosting, multi-layer)
        blocker_density = (
            board_tensor[:, 10] * 1.5
            + board_tensor[:, 11] * 1.0
            + board_tensor[:, 12] * 1.2
            + board_tensor[:, 13] * 2.0
            + board_tensor[:, 14] * 1.0
            + void_cutouts * 2.5
        ) / 5.0  # Normalized cell obstruction in [0, ~1.8]

        # Cumulative obstruction strictly below each cell (r, c):
        # below_mask is (H, H), blocker_density is (B, H, W) -> matmul sums over r' > r
        below_hazard_map = torch.einsum("ij,bjw->biw", self.below_mask, blocker_density)  # (B, 9, 9)

        # Cumulative ingredient load above or at each cell (indicates cells sitting in an active drop chute):
        # self_below_mask^T sums over r_ing <= r
        chute_active_map = torch.einsum("ji,bjw->biw", self.self_below_mask, ing_map).clamp(0.0, 1.0)  # (B, 9, 9)

        # Column-level obstruction specifically beneath ingredients (or full column if no ingredient in col)
        col_ing_weight = chute_active_map + 0.15
        col_obstruction = (blocker_density * col_ing_weight).sum(dim=1)  # (B, 9)

        # Lateral obstruction gradient: compares column c obstruction with adjacent columns c-1 and c+1
        pad_col_obs = F.pad(below_hazard_map, (1, 1, 0, 0), mode="replicate")
        left_obs_diff = below_hazard_map - pad_col_obs[:, :, :-2]   # >0 means shifting left reduces obstruction
        right_obs_diff = below_hazard_map - pad_col_obs[:, :, 2:]   # >0 means shifting right reduces obstruction

        # Vertical striped laser availability in column (channel 7)
        vert_laser_col = board_tensor[:, 7].sum(dim=1, keepdim=True).expand(B, H, W)

        depth_plane = self.row_depth.expand(B, 1, H, W).squeeze(1)

        physics_stack = torch.stack([
            ing_map,
            blocker_density,
            below_hazard_map,
            chute_active_map,
            left_obs_diff,
            right_obs_diff,
            vert_laser_col,
            depth_plane
        ], dim=1)  # (B, 8, 9, 9)

        physics_feat = self.physics_proj(physics_stack)
        return physics_feat, ing_map, col_obstruction, below_hazard_map


class ColumnFlowVerticalAttentionBlock(nn.Module):
    """
    Column-Flow Vertical Attention + Lateral Lane-Exchange Block.
    1. Intra-Column Vertical Self-Attention:
       Treats each of the 9 columns as a length-9 vertical sequence (B*9, 9, D), applying
       multi-head attention with a learnable downward gravity bias and obstruction gating.
    2. Cross-Column Horizontal Attention:
       Allows adjacent and remote columns to compare drop-lane clearance and coordinate
       lateral lane shifts.
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

        # --- Intra-Column Vertical Attention ---
        self.norm_vert = nn.LayerNorm(d_model)
        self.v_qkv = nn.Linear(d_model, d_model * 3, bias=False)
        self.v_out = nn.Linear(d_model, d_model)
        # Learnable relative vertical displacement bias (r_query - r_key) across 9x9 column pairs
        self.vert_rel_bias = nn.Parameter(torch.zeros(num_heads, height, height))
        self.hazard_attn_proj = nn.Linear(2, num_heads, bias=False)

        # --- Cross-Column Horizontal Lane Attention ---
        self.norm_horiz = nn.LayerNorm(d_model)
        self.h_qkv = nn.Linear(d_model, d_model * 3, bias=False)
        self.h_out = nn.Linear(d_model, d_model)
        self.horiz_rel_bias = nn.Parameter(torch.zeros(num_heads, width, width))

        # --- Feed-Forward Network ---
        self.norm_ffn = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 3),
            nn.GELU(),
            nn.Linear(d_model * 3, d_model)
        )

        self._init_gravity_biases()

    def _init_gravity_biases(self) -> None:
        with torch.no_grad():
            for r1 in range(self.height):
                for r2 in range(self.height):
                    if r2 >= r1:
                        # Prefer attending downward along the ingredient drop trajectory
                        self.vert_rel_bias[:, r1, r2] = 0.25 / (1.0 + float(r2 - r1))
                    else:
                        self.vert_rel_bias[:, r1, r2] = -0.1 * float(r1 - r2)
            for c1 in range(self.width):
                for c2 in range(self.width):
                    dist = abs(c1 - c2)
                    self.horiz_rel_bias[:, c1, c2] = 0.3 if dist == 1 else -0.15 * float(dist)

    def forward(
        self,
        x_2d: torch.Tensor,
        ing_map: torch.Tensor,
        below_hazard_map: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x_2d: (B, D, 9, 9) spatial feature map
            ing_map: (B, 9, 9) ingredient coordinates
            below_hazard_map: (B, 9, 9) obstruction density below each cell
        Returns:
            x_out: (B, D, 9, 9) updated spatial feature map
            col_flow_attn: (B, 9, 9, 9) column-wise vertical attention weights (B, W, H, H)
        """
        B, D, H, W = x_2d.shape
        nh = self.num_heads
        dh = self.head_dim

        # 1. Intra-Column Vertical Attention: reshape to (B * W, H, D)
        x_cols = x_2d.permute(0, 3, 2, 1).contiguous().view(B * W, H, D)
        v_norm = self.norm_vert(x_cols)
        q_v, k_v, val_v = self.v_qkv(v_norm).chunk(3, dim=-1)

        q_v = q_v.view(B * W, H, nh, dh).transpose(1, 2)  # (B*W, nh, H, dh)
        k_v = k_v.view(B * W, H, nh, dh).transpose(1, 2)
        val_v = val_v.view(B * W, H, nh, dh).transpose(1, 2)

        v_scores = torch.matmul(q_v, k_v.transpose(-2, -1)) * self.scale  # (B*W, nh, H, H)

        # Pairwise (ingredient_i, hazard_j) bias inside each column
        ing_cols = ing_map.permute(0, 2, 1).contiguous().view(B * W, H, 1)
        haz_cols = below_hazard_map.permute(0, 2, 1).contiguous().view(B * W, 1, H)
        pair_phys = torch.stack([
            ing_cols.expand(B * W, H, H),
            haz_cols.expand(B * W, H, H)
        ], dim=-1)  # (B*W, H, H, 2)
        phys_bias = self.hazard_attn_proj(pair_phys).permute(0, 3, 1, 2)  # (B*W, nh, H, H)

        v_scores = v_scores + self.vert_rel_bias.unsqueeze(0) + phys_bias
        v_attn = F.softmax(v_scores, dim=-1)

        v_out = torch.matmul(v_attn, val_v).transpose(1, 2).contiguous().view(B * W, H, D)
        x_cols = x_cols + self.v_out(v_out)

        # Reshape back to 2D and prepare for Cross-Column Horizontal Attention: (B * H, W, D)
        x_2d = x_cols.view(B, W, H, D).permute(0, 2, 1, 3).contiguous()  # (B, H, W, D)
        x_rows = x_2d.view(B * H, W, D)
        h_norm = self.norm_horiz(x_rows)
        q_h, k_h, val_h = self.h_qkv(h_norm).chunk(3, dim=-1)

        q_h = q_h.view(B * H, W, nh, dh).transpose(1, 2)
        k_h = k_h.view(B * H, W, nh, dh).transpose(1, 2)
        val_h = val_h.view(B * H, W, nh, dh).transpose(1, 2)

        h_scores = torch.matmul(q_h, k_h.transpose(-2, -1)) * self.scale + self.horiz_rel_bias.unsqueeze(0)
        h_attn = F.softmax(h_scores, dim=-1)
        h_out = torch.matmul(h_attn, val_h).transpose(1, 2).contiguous().view(B * H, W, D)
        x_rows = x_rows + self.h_out(h_out)

        # 3. FFN Update
        x_rows = x_rows + self.ffn(self.norm_ffn(x_rows))
        x_out = x_rows.view(B, H, W, D).permute(0, 3, 1, 2).contiguous()  # (B, D, H, W)

        col_flow_attn = v_attn.mean(dim=1).view(B, W, H, H)
        return x_out, col_flow_attn


class LateralLaneShiftAndSinkScorer(nn.Module):
    """
    Explicitly scores the 144 canonical swap actions by decomposing them into:
      1. Lateral Lane-Shift Scorer (actions 0..71, horizontal swaps (r, c) <-> (r, c+1)):
         Evaluates whether swapping horizontally moves an ingredient from a high-obstruction
         column into a lower-obstruction adjacent drop column, or clears a blocker beneath
         an ingredient column.
      2. Vertical Sink Scorer (actions 72..143, vertical swaps (r, c) <-> (r+1, c)):
         Evaluates whether swapping vertically directly drops an ingredient 1 row closer
         to the bottom exit chute or triggers a vertical column cascade.
    """

    def __init__(self, d_model: int = 192):
        super().__init__()
        # Horizontal swap scorer: takes [left_feat, right_feat, physics_delta_4]
        self.horiz_lane_scorer = nn.Sequential(
            nn.Linear(d_model * 2 + 4, d_model),
            nn.GELU(),
            nn.Linear(d_model, 64),
            nn.GELU(),
            nn.Linear(64, 1)
        )

        # Vertical swap scorer: takes [top_feat, bot_feat, vertical_sink_physics_4]
        self.vert_sink_scorer = nn.Sequential(
            nn.Linear(d_model * 2 + 4, d_model),
            nn.GELU(),
            nn.Linear(d_model, 64),
            nn.GELU(),
            nn.Linear(64, 1)
        )

    def forward(
        self,
        feat_2d: torch.Tensor,
        ing_map: torch.Tensor,
        below_hazard_map: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            lane_shift_logits: (B, 72) horizontal swap scores
            vertical_sink_logits: (B, 72) vertical swap scores
        """
        B, D, H, W = feat_2d.shape

        # --- 1. Horizontal Swaps (9 x 8 = 72 actions) ---
        f_left = feat_2d[:, :, :, :-1].permute(0, 2, 3, 1)   # (B, 9, 8, D)
        f_right = feat_2d[:, :, :, 1:].permute(0, 2, 3, 1)   # (B, 9, 8, D)

        ing_l = ing_map[:, :, :-1].unsqueeze(-1)             # (B, 9, 8, 1)
        ing_r = ing_map[:, :, 1:].unsqueeze(-1)              # (B, 9, 8, 1)
        haz_l = below_hazard_map[:, :, :-1].unsqueeze(-1)    # (B, 9, 8, 1)
        haz_r = below_hazard_map[:, :, 1:].unsqueeze(-1)     # (B, 9, 8, 1)

        # Net obstruction reduction when shifting an ingredient from left->right or right->left
        shift_right_gain = ing_l * (haz_l - haz_r)
        shift_left_gain = ing_r * (haz_r - haz_l)
        horiz_phys = torch.cat([ing_l + ing_r, haz_l - haz_r, shift_right_gain, shift_left_gain], dim=-1)

        horiz_in = torch.cat([f_left, f_right, horiz_phys], dim=-1).reshape(B, 72, 2 * D + 4)
        lane_shift_logits = self.horiz_lane_scorer(horiz_in).squeeze(-1)  # (B, 72)

        # --- 2. Vertical Swaps (8 x 9 = 72 actions) ---
        f_top = feat_2d[:, :, :-1, :].permute(0, 2, 3, 1)    # (B, 8, 9, D)
        f_bot = feat_2d[:, :, 1:, :].permute(0, 2, 3, 1)     # (B, 8, 9, D)

        ing_top = ing_map[:, :-1, :].unsqueeze(-1)           # (B, 8, 9, 1)
        ing_bot = ing_map[:, 1:, :].unsqueeze(-1)            # (B, 8, 9, 1)
        haz_top = below_hazard_map[:, :-1, :].unsqueeze(-1)
        haz_bot = below_hazard_map[:, 1:, :].unsqueeze(-1)

        # Direct downward sink indicator: ingredient on top swapping downward with non-ingredient
        direct_sink = ing_top * (1.0 - ing_bot)
        vert_phys = torch.cat([direct_sink, ing_bot, haz_top, haz_bot], dim=-1)

        vert_in = torch.cat([f_top, f_bot, vert_phys], dim=-1).reshape(B, 72, 2 * D + 4)
        vertical_sink_logits = self.vert_sink_scorer(vert_in).squeeze(-1)  # (B, 72)

        return lane_shift_logits, vertical_sink_logits


class IngredientDropSpecialistNet(nn.Module):
    """
    IngredientDropSpecialistNet: Specialized Deep Architecture for INGREDIENTS (Drop) Levels.
    Combines explicit below-ingredient column obstruction physics, 6 layers of Column-Flow
    Vertical & Lateral Attention, and a dedicated Lateral Lane-Shift vs. Vertical Sink Scorer.
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
        self.d_model = d_model
        self.num_actions = num_actions

        # 1. Explicit Below-Ingredient Column Obstruction Analyzer
        phys_dim = 48
        self.obstruction_analyzer = ColumnObstructionAnalyzer(
            height=board_h, width=board_w, d_out=phys_dim
        )

        # 2. Multi-scale Spatial Stem fused with Drop-Physics Tensor
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels + phys_dim, d_model, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(d_model),
            nn.GELU(),
            nn.Conv2d(d_model, d_model, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(d_model),
            nn.GELU()
        )
        self.coord_embed = nn.Parameter(torch.randn(1, d_model, board_h, board_w) * 0.02)

        # 3. Stack of Column-Flow Vertical Attention Blocks
        self.blocks = nn.ModuleList([
            ColumnFlowVerticalAttentionBlock(
                d_model=d_model, num_heads=num_heads, height=board_h, width=board_w
            )
            for _ in range(num_layers)
        ])

        # 4. Lateral Lane-Shift & Vertical Sink Scorer + Global Context Policy Refiner
        self.lane_sink_scorer = LateralLaneShiftAndSinkScorer(d_model=d_model)
        self.global_policy_conv = nn.Sequential(
            nn.Conv2d(d_model, 96, kernel_size=3, padding=1),
            nn.BatchNorm2d(96),
            nn.GELU(),
            nn.Flatten(),
            nn.Linear(96 * board_h * board_w, 256),
            nn.GELU(),
            nn.Linear(256, num_actions)
        )

        # 5. Exit Chute Clearance Telemetry Head + Value & Win Probability Heads
        self.chute_clearance_head = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.GELU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        self.value_head = nn.Sequential(
            nn.Linear(d_model * 2 + board_w, 128),
            nn.GELU(),
            nn.Linear(128, 1),
            nn.Tanh()
        )
        self.win_head = nn.Sequential(
            nn.Linear(d_model * 2 + board_w, 128),
            nn.GELU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(
        self,
        board_tensor: torch.Tensor,
        legal_mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for IngredientDropSpecialistNet.
        Args:
            board_tensor: (B, 20, 9, 9) canonical 20-channel board state
            legal_mask: Optional (B, 144) boolean mask of legal moves
        """
        B, C, H, W = board_tensor.shape

        # Step 1: Analyze ingredient positions and downward column obstructions
        phys_feat, ing_map, col_obstruction, below_hazard_map = self.obstruction_analyzer(board_tensor)

        # Step 2: Fuse board tensor with drop-physics planes
        x = self.stem(torch.cat([board_tensor, phys_feat], dim=1)) + self.coord_embed

        # Step 3: Pass through Column-Flow Vertical Attention layers
        last_col_attn = None
        for block in self.blocks:
            x, last_col_attn = block(x, ing_map, below_hazard_map)

        # Step 4: Score Lateral Lane-Shifts (0..71) vs Vertical Sinks (72..143)
        lane_shift_logits, vertical_sink_logits = self.lane_sink_scorer(x, ing_map, below_hazard_map)
        specialist_swap_logits = torch.cat([lane_shift_logits, vertical_sink_logits], dim=-1)  # (B, 144)
        policy_logits = specialist_swap_logits + self.global_policy_conv(x)

        if legal_mask is not None:
            policy_logits = policy_logits.masked_fill(~legal_mask.bool(), -1e9)

        # Step 5: Per-column exit chute clearance probability & Value/Win readout
        col_pooled = x.mean(dim=2).transpose(1, 2)  # (B, 9, d_model)
        exit_chute_clearance_prob = self.chute_clearance_head(col_pooled).squeeze(-1)  # (B, 9)

        x_flat = x.flatten(2)  # (B, d_model, 81)
        readout = torch.cat([x_flat.mean(dim=-1), x_flat.max(dim=-1).values, exit_chute_clearance_prob], dim=-1)
        value = self.value_head(readout)
        win_prob = self.win_head(readout)

        return {
            "policy_logits": policy_logits,
            "value": value,
            "win_prob": win_prob,
            "column_obstruction_density": col_obstruction,
            "below_ingredient_hazard_map": below_hazard_map,
            "lane_shift_logits": lane_shift_logits,
            "vertical_sink_logits": vertical_sink_logits,
            "exit_chute_clearance_prob": exit_chute_clearance_prob,
            "column_vertical_attention": last_col_attn,
        }

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def create_ingredient_drop_specialist(device: str = "cpu") -> IngredientDropSpecialistNet:
    model = IngredientDropSpecialistNet()
    return model.to(device)


if __name__ == "__main__":
    torch.manual_seed(42)
    model = create_ingredient_drop_specialist("cpu")
    params = model.count_parameters()
    print(f"[IngredientDropSpecialistNet] Trainable Parameters: {params:,} ({params / 1e6:.2f}M)")

    dummy_board = torch.zeros(2, 20, 9, 9)
    dummy_board[:, 18] = 1.0  # Active grid
    dummy_board[0, 15, 2, 4] = 1.0  # Ingredient at (2, 4) on channel 15
    dummy_board[0, 12, 5, 4] = 1.0  # Frosting blocker below ingredient at (5, 4)
    dummy_board[1, 15, 1, 2] = 1.0  # Ingredient at (1, 2)

    out = model(dummy_board)
    for k, v in out.items():
        print(f"  {k}: {tuple(v.shape)}")
