"""
AlphaCandy Model Zoo — Part 4.4: Special Combo Alchemist Network (`SpecialComboAlchemistNet`).
==============================================================================================
Ultra-specialized architecture for manufacturing and pairing Special Candies:
  - Color Bomb + Striped, Color Bomb + Wrapped, Double Color Bomb, Wrapped + Striped.

Architectural Innovations:
  1. Bank of Geometric Template Filters (`GeometricSpecialTemplateBank`):
     - Applies exact geometric pattern-matching filters across all 6 candy color channels (0..5):
       * 5x1 and 1x5 Color Bomb line-5 templates
       * 3x3 T-shape (4 orientations) and L-shape (4 orientations) Wrapped candy templates
       * 4x1 and 1x4 Striped candy line-4 templates
     - Identifies exact 1-swap away recipes for synthesizing Color Bombs, Wrapped, and Striped candies.
  2. 2-Hop Special Proximity Router (`TwoHopSpecialProximityRouter`):
     - Detects when two Special Candies (channels 6..9) are either in 1-hop adjacent contact
       (ready for immediate super-combo detonation) or separated by 1-2 tiles (2-hop proximity),
       prioritizing moves that drop separated specials into adjacent contact.
"""

import math
from typing import Dict, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class GeometricSpecialTemplateBank(nn.Module):
    """
    Bank of exact geometric convolutional template filters applied per-color across channels 0..5:
      1. Color Bomb Templates : 1x5 horizontal and 5x1 vertical (detects >=4 matching cells in a 5-span)
      2. Wrapped Templates    : 8 distinct 3x3 T-shape (4 orientations) and L-shape (4 orientations) masks
      3. Striped Templates    : 1x4 horizontal and 4x1 vertical (detects >=3 matching cells in a 4-span)
    """

    def __init__(self, d_out: int = 64):
        super().__init__()
        # 1. Color Bomb 1x5 and 5x1 kernels
        cb_h = torch.ones(1, 1, 1, 5, dtype=torch.float32)
        cb_v = torch.ones(1, 1, 5, 1, dtype=torch.float32)

        # 2. Wrapped Candy 3x3 T-shapes (4) and L-shapes (4) -> (8, 1, 3, 3)
        t_up = [[1, 1, 1], [0, 1, 0], [0, 1, 0]]
        t_down = [[0, 1, 0], [0, 1, 0], [1, 1, 1]]
        t_left = [[1, 0, 0], [1, 1, 1], [1, 0, 0]]
        t_right = [[0, 0, 1], [1, 1, 1], [0, 0, 1]]

        l_tl = [[1, 1, 1], [1, 0, 0], [1, 0, 0]]
        l_tr = [[1, 1, 1], [0, 0, 1], [0, 0, 1]]
        l_bl = [[1, 0, 0], [1, 0, 0], [1, 1, 1]]
        l_br = [[0, 0, 1], [0, 0, 1], [1, 1, 1]]

        wrapped_templates = torch.tensor(
            [t_up, t_down, t_left, t_right, l_tl, l_tr, l_bl, l_br],
            dtype=torch.float32
        ).unsqueeze(1)  # (8, 1, 3, 3)

        # 3. Striped Candy 1x4 and 4x1 kernels (padded to 5 for symmetric centering or conv with pad)
        st_h = torch.ones(1, 1, 1, 4, dtype=torch.float32)
        st_v = torch.ones(1, 1, 4, 1, dtype=torch.float32)

        self.register_buffer("cb_h", cb_h)
        self.register_buffer("cb_v", cb_v)
        self.register_buffer("wrapped_templates", wrapped_templates)
        self.register_buffer("st_h", st_h)
        self.register_buffer("st_v", st_v)

        # Learnable multi-template synthesis head
        self.template_fuse = nn.Sequential(
            nn.Conv2d(14, d_out, kernel_size=3, padding=1, bias=False),
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
            template_feat: (B, d_out, 9, 9)
            color_bomb_recipe_map: (B, 9, 9)
            wrapped_recipe_map: (B, 9, 9)
            striped_recipe_map: (B, 9, 9)
        """
        B, C, H, W = board_tensor.shape
        # Extract the 6 candy colors (channels 0..5) -> reshape to (B*6, 1, 9, 9)
        colors = board_tensor[:, :6].contiguous().view(B * 6, 1, H, W)

        # --- 1. 5x1 and 1x5 Color Bomb Template Responses ---
        # A 4-of-5 match in a 5-span means 1 adjacent swap can complete a 5-in-a-row Color Bomb!
        cb_h_resp = F.conv2d(colors, self.cb_h, padding=(0, 2)).view(B, 6, H, W)
        cb_v_resp = F.conv2d(colors, self.cb_v, padding=(2, 0)).view(B, 6, H, W)
        cb_h_max = F.relu(cb_h_resp - 2.5).max(dim=1).values  # Peaks when >=3..4 of 5 aligned
        cb_v_max = F.relu(cb_v_resp - 2.5).max(dim=1).values
        color_bomb_recipe_map = torch.maximum(cb_h_max, cb_v_max) / 2.5  # (B, 9, 9)

        # --- 2. 3x3 T-Shape & L-Shape Wrapped Candy Template Responses ---
        # Convolve 8 T/L templates (each has 5 ones); response >= 3.5 indicates 1-swap-away T/L!
        w_resp = F.conv2d(colors, self.wrapped_templates, padding=1).view(B, 6, 8, H, W)
        w_orient_max = F.relu(w_resp - 2.5).amax(dim=1)  # (B, 8, 9, 9) max over 6 colors per T/L orientation
        wrapped_recipe_map = w_orient_max.amax(dim=1) / 2.5  # (B, 9, 9)

        # --- 3. 4x1 and 1x4 Striped Candy Template Responses ---
        # Pad asymmetrically (1, 2) to preserve 9x9 spatial dimensions
        colors_pad_h = F.pad(colors, (1, 2, 0, 0))
        colors_pad_v = F.pad(colors, (0, 0, 1, 2))
        st_h_resp = F.conv2d(colors_pad_h, self.st_h).view(B, 6, H, W)
        st_v_resp = F.conv2d(colors_pad_v, self.st_v).view(B, 6, H, W)
        st_h_max = F.relu(st_h_resp - 2.0).max(dim=1).values
        st_v_max = F.relu(st_v_resp - 2.0).max(dim=1).values
        striped_recipe_map = torch.maximum(st_h_max, st_v_max) / 2.0  # (B, 9, 9)

        # Stack all 14 geometric response maps:
        # [cb_h, cb_v, 8 T/L orientations, st_h, st_v, cb_recipe, wrapped_recipe] -> 14 channels
        all_templates = torch.cat([
            cb_h_max.unsqueeze(1),
            cb_v_max.unsqueeze(1),
            w_orient_max,
            st_h_max.unsqueeze(1),
            st_v_max.unsqueeze(1),
            color_bomb_recipe_map.unsqueeze(1),
            wrapped_recipe_map.unsqueeze(1)
        ], dim=1)  # (B, 14, 9, 9)

        template_feat = self.template_fuse(all_templates)
        return template_feat, color_bomb_recipe_map, wrapped_recipe_map, striped_recipe_map


class TwoHopSpecialProximityRouter(nn.Module):
    """
    2-Hop Special Proximity Router.
    Detects existing Special Candies (channels 6..9: Striped H, Striped V, Wrapped, Color Bomb)
    and constructs:
      1. 1-Hop Direct Contact Graph (distance == 1): Specials ready for immediate combo swap!
      2. 2-Hop Proximity Separation Graph (distance == 2, especially vertical/diagonal separation
         where 1 column drop brings the two specials into adjacent 1-hop contact!).
      3. Routes multi-head attention across 1-hop and 2-hop special neighborhoods.
    """

    def __init__(self, d_model: int = 192, num_heads: int = 6, height: int = 9, width: int = 9):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)
        self.height = height
        self.width = width
        N = height * width

        # Precompute 1-hop, 2-hop, and gravity-drop proximity matrices on 81x81 grid
        hop1_mask = torch.zeros(N, N, dtype=torch.float32)
        hop2_mask = torch.zeros(N, N, dtype=torch.float32)
        drop_align_mask = torch.zeros(N, N, dtype=torch.float32)

        for r1 in range(height):
            for c1 in range(width):
                u = r1 * width + c1
                for r2 in range(height):
                    for c2 in range(width):
                        v = r2 * width + c2
                        manhattan = abs(r1 - r2) + abs(c1 - c2)
                        if manhattan == 1:
                            hop1_mask[u, v] = 1.0
                        elif manhattan in (2, 3):
                            hop2_mask[u, v] = 1.0 / float(manhattan - 1)
                        # If two specials are in the same or adjacent columns separated vertically by 1-2 rows,
                        # clearing the cells beneath the upper special drops it directly next to the lower special!
                        if abs(c1 - c2) <= 1 and 1 <= abs(r1 - r2) <= 2:
                            drop_align_mask[u, v] = 1.0

        # 3x3 and 5x5 spatial proximity kernels
        hop1_kernel = torch.tensor(
            [[[[0.0, 1.0, 0.0],
               [1.0, 0.0, 1.0],
               [0.0, 1.0, 0.0]]]],
            dtype=torch.float32
        )
        hop2_kernel = torch.tensor(
            [[[[0.0, 0.5, 1.0, 0.5, 0.0],
               [0.5, 1.0, 0.5, 1.0, 0.5],
               [1.0, 0.5, 0.0, 0.5, 1.0],
               [0.5, 1.0, 0.5, 1.0, 0.5],
               [0.0, 0.5, 1.0, 0.5, 0.0]]]],
            dtype=torch.float32
        )

        self.register_buffer("hop1_mask", hop1_mask)
        self.register_buffer("hop2_mask", hop2_mask)
        self.register_buffer("drop_align_mask", drop_align_mask)
        self.register_buffer("hop1_kernel", hop1_kernel)
        self.register_buffer("hop2_kernel", hop2_kernel)

        self.norm1 = nn.LayerNorm(d_model)
        self.qkv = nn.Linear(d_model, d_model * 3, bias=False)
        self.prox_bias_proj = nn.Linear(3, num_heads, bias=False)
        self.out_proj = nn.Linear(d_model, d_model)

        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 3),
            nn.GELU(),
            nn.Linear(d_model * 3, d_model)
        )

    def compute_proximity_map(self, board_tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Computes:
          - special_value_map: weighted presence of Specials (Color Bomb=2.5, Wrapped=1.6, Striped=1.3)
          - two_hop_proximity_map: (B, 9, 9) highlighting 1-hop ready combos and 2-hop bridge cells
            that drop separated specials into contact!
        """
        striped_h = board_tensor[:, 6:7]
        striped_v = board_tensor[:, 7:8]
        wrapped = board_tensor[:, 8:9]
        color_bomb = board_tensor[:, 9:10]

        any_special = (striped_h + striped_v + wrapped + color_bomb).clamp(0.0, 1.0)
        special_weight = 1.3 * (striped_h + striped_v) + 1.6 * wrapped + 2.5 * color_bomb

        # 1-hop neighbor field and 2-hop neighbor field around existing specials
        hop1_field = F.conv2d(special_weight, self.hop1_kernel, padding=1)
        hop2_field = F.conv2d(special_weight, self.hop2_kernel, padding=2)

        # Cells where a special sits adjacent to another special (1-hop ready super-combo)
        ready_combo = any_special * hop1_field
        # Bridge cells sitting between two specials separated by 1-2 tiles (dropping bridge merges them!)
        bridge_zone = (hop1_field >= 2.0).float() * hop1_field + 0.5 * any_special * hop2_field

        two_hop_map = (ready_combo * 1.5 + bridge_zone + 0.35 * hop2_field).squeeze(1)
        return special_weight.squeeze(1), two_hop_map

    def forward(
        self,
        tokens: torch.Tensor,
        special_weight_map: torch.Tensor
    ) -> torch.Tensor:
        B, N, D = tokens.shape
        H = self.num_heads
        Dh = self.head_dim

        t_norm = self.norm1(tokens)
        q, k, v = self.qkv(t_norm).chunk(3, dim=-1)
        q = q.view(B, N, H, Dh).transpose(1, 2)
        k = k.view(B, N, H, Dh).transpose(1, 2)
        v = v.view(B, N, H, Dh).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        # Pairwise special proximity features (u, v)
        sp_u = special_weight_map.view(B, N, 1)
        sp_v = special_weight_map.view(B, 1, N)
        sp_pair = (sp_u * sp_v) + 0.3 * (sp_u + sp_v)  # (B, 81, 81)

        prox_feats = torch.stack([
            self.hop1_mask.unsqueeze(0) * sp_pair,
            self.hop2_mask.unsqueeze(0) * sp_pair,
            self.drop_align_mask.unsqueeze(0) * sp_pair
        ], dim=-1)  # (B, 81, 81, 3)

        prox_bias = self.prox_bias_proj(prox_feats).permute(0, 3, 1, 2)
        attn = F.softmax(scores + prox_bias, dim=-1)

        out = torch.matmul(attn, v).transpose(1, 2).contiguous().view(B, N, D)
        tokens = tokens + self.out_proj(out)
        tokens = tokens + self.ffn(self.norm2(tokens))
        return tokens


class SpecialComboAlchemistNet(nn.Module):
    """
    SpecialComboAlchemistNet: Specialized Deep Architecture for Manufacturing & Pairing Special Candies.
    Combines a bank of 5x1/1x5 Color Bomb, 3x3 T/L Wrapped, and 4x1/1x4 Striped geometric template
    filters with a 6-layer 2-Hop Special Proximity Router.
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

        # 1. Geometric Template Filter Bank (5x1/1x5 CB, 3x3 T/L Wrapped, 4x1/1x4 Striped)
        tmpl_dim = 64
        self.template_bank = GeometricSpecialTemplateBank(d_out=tmpl_dim)

        # 2. Stem fusing 20 board channels + 64 geometric template channels + 2 proximity planes
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels + tmpl_dim + 2, d_model, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(d_model),
            nn.GELU(),
            nn.Conv2d(d_model, d_model, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(d_model),
            nn.GELU()
        )
        self.pos_embed = nn.Parameter(torch.randn(1, self.num_nodes, d_model) * 0.02)

        # 3. 6-Layer 2-Hop Special Proximity Router Backbone
        self.routers = nn.ModuleList([
            TwoHopSpecialProximityRouter(
                d_model=d_model, num_heads=num_heads, height=board_h, width=board_w
            )
            for _ in range(num_layers)
        ])
        self.final_norm = nn.LayerNorm(d_model)

        # 4. Alchemist Pairwise Swap Scorer (Manufacture + 1-Hop Combo + 2-Hop Drop Alignment)
        self.horiz_alchemist_scorer = nn.Sequential(
            nn.Linear(d_model * 2 + 8, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1)
        )
        self.vert_alchemist_scorer = nn.Sequential(
            nn.Linear(d_model * 2 + 8, d_model),
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

        # 5. Super-Combo Synergy Telemetry Head ([CB+Striped, CB+Wrapped, Double_CB, Wrapped+Striped])
        self.combo_synergy_head = nn.Sequential(
            nn.Linear(d_model * 2 + 4, 64),
            nn.GELU(),
            nn.Linear(64, 4),
            nn.Sigmoid()
        )
        self.value_head = nn.Sequential(
            nn.Linear(d_model * 2 + 4, 128),
            nn.GELU(),
            nn.Linear(128, 1),
            nn.Tanh()
        )
        self.win_head = nn.Sequential(
            nn.Linear(d_model * 2 + 4, 128),
            nn.GELU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def _score_alchemist_swaps(
        self,
        z_2d: torch.Tensor,
        cb_map: torch.Tensor,
        wrapped_map: torch.Tensor,
        striped_map: torch.Tensor,
        prox_map: torch.Tensor
    ) -> torch.Tensor:
        B, D, H, W = z_2d.shape
        stack4 = torch.stack([cb_map, wrapped_map, striped_map, prox_map], dim=-1)  # (B, 9, 9, 4)

        # Horizontal swaps (9 x 8 = 72)
        h_l = z_2d[:, :, :, :-1].permute(0, 2, 3, 1)
        h_r = z_2d[:, :, :, 1:].permute(0, 2, 3, 1)
        h_recipes = torch.cat([stack4[:, :, :-1, :], stack4[:, :, 1:, :]], dim=-1)  # (B, 9, 8, 8)
        h_in = torch.cat([h_l, h_r, h_recipes], dim=-1).reshape(B, 72, 2 * D + 8)
        h_logits = self.horiz_alchemist_scorer(h_in).squeeze(-1)

        # Vertical swaps (8 x 9 = 72)
        v_t = z_2d[:, :, :-1, :].permute(0, 2, 3, 1)
        v_b = z_2d[:, :, 1:, :].permute(0, 2, 3, 1)
        v_recipes = torch.cat([stack4[:, :-1, :, :], stack4[:, 1:, :, :]], dim=-1)  # (B, 8, 9, 8)
        v_in = torch.cat([v_t, v_b, v_recipes], dim=-1).reshape(B, 72, 2 * D + 8)
        v_logits = self.vert_alchemist_scorer(v_in).squeeze(-1)

        return torch.cat([h_logits, v_logits], dim=-1)  # (B, 144)

    def forward(
        self,
        board_tensor: torch.Tensor,
        legal_mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        B, C, H, W = board_tensor.shape

        # Step 1: Run Geometric Template Bank (5x1/1x5 CB, 3x3 T/L Wrapped, 4x1/1x4 Striped)
        tmpl_feat, cb_recipe_map, wrapped_recipe_map, striped_recipe_map = (
            self.template_bank(board_tensor)
        )

        # Step 2: Compute 1-Hop and 2-Hop Special Proximity Maps
        special_weight_map, two_hop_proximity_map = self.routers[0].compute_proximity_map(board_tensor)

        stem_in = torch.cat([
            board_tensor,
            tmpl_feat,
            special_weight_map.unsqueeze(1),
            two_hop_proximity_map.unsqueeze(1)
        ], dim=1)
        stem_2d = self.stem(stem_in)
        tokens = stem_2d.flatten(2).transpose(1, 2) + self.pos_embed

        # Step 3: Route through 6 layers of 2-Hop Special Proximity Attention
        for router in self.routers:
            tokens = router(tokens, special_weight_map)

        tokens = self.final_norm(tokens)
        z_2d = tokens.transpose(1, 2).view(B, self.d_model, H, W)

        # Step 4: Score all 144 canonical actions for special crafting & 1/2-hop combo pairing
        alchemist_logits = self._score_alchemist_swaps(
            z_2d, cb_recipe_map, wrapped_recipe_map, striped_recipe_map, two_hop_proximity_map
        )
        policy_logits = alchemist_logits + self.global_policy_head(z_2d)

        if legal_mask is not None:
            policy_logits = policy_logits.masked_fill(~legal_mask.bool(), -1e9)

        # Step 5: Global Readout & Combo Synergy Telemetry
        summary_stats = torch.stack([
            cb_recipe_map.flatten(1).max(dim=-1).values,
            wrapped_recipe_map.flatten(1).max(dim=-1).values,
            striped_recipe_map.flatten(1).max(dim=-1).values,
            two_hop_proximity_map.flatten(1).max(dim=-1).values
        ], dim=-1)  # (B, 4)

        readout = torch.cat([tokens.mean(dim=1), tokens.max(dim=1).values, summary_stats], dim=-1)
        combo_synergy_probs = self.combo_synergy_head(readout)  # (B, 4)
        value = self.value_head(readout)
        win_prob = self.win_head(readout)

        return {
            "policy_logits": policy_logits,
            "value": value,
            "win_prob": win_prob,
            "color_bomb_recipe_map": cb_recipe_map,
            "wrapped_recipe_map": wrapped_recipe_map,
            "striped_recipe_map": striped_recipe_map,
            "two_hop_special_proximity_map": two_hop_proximity_map,
            "combo_synergy_probs": combo_synergy_probs,
        }

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def create_special_combo_alchemist(device: str = "cpu") -> SpecialComboAlchemistNet:
    model = SpecialComboAlchemistNet()
    return model.to(device)


if __name__ == "__main__":
    torch.manual_seed(42)
    model = create_special_combo_alchemist("cpu")
    params = model.count_parameters()
    print(f"[SpecialComboAlchemistNet] Trainable Parameters: {params:,} ({params / 1e6:.2f}M)")

    dummy_board = torch.zeros(2, 20, 9, 9)
    dummy_board[:, 18] = 1.0  # Active grid
    # 4-of-5 Red candies on row 4 (1 swap away from Color Bomb!)
    dummy_board[0, 0, 4, [1, 2, 4, 5]] = 1.0
    dummy_board[0, 0, 3, 3] = 1.0
    # Color Bomb at (6, 3) and Wrapped Candy 2 hops away at (4, 3)!
    dummy_board[0, 9, 6, 3] = 1.0
    dummy_board[0, 8, 4, 3] = 1.0

    out = model(dummy_board)
    for k, v in out.items():
        print(f"  {k}: {tuple(v.shape)}")
