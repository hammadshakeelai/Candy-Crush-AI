"""
AlphaCandy Model Zoo — Part 4.5: Endgame Clutch Miracle Network (`EndgameClutchMiracleNet`).
============================================================================================
Ultra-specialized architecture for Endgame Clutch States (<= 5 moves remaining).

Architectural Innovations:
  1. Maximum Cascade Volatility & Chain-Reaction Entropy Engine (`CascadeVolatilityEntropyAnalyzer`):
     - In desperate endgame states where standard 1-move expected value cannot clear the
       remaining objectives, switches from mean-return optimization to Upper-Confidence
       Cascade Volatility optimization (mu(s,a) + beta_clutch * sigma_cascade(s,a)).
     - Quantifies Bottom-Row Gravity Avalanche Leverage (rows 5..8 displace up to 8 rows of
       candies above them) and Staggered Column Color-Collision Entropy (measuring secondary
       match-3 chain reactions triggered when columns shift down by 1, 2, or 3 cells).
  2. 1-to-2 Move Lethal Win Detector (`OneToTwoMoveLethalDetector`):
     - Simultaneously scans all 144 actions for exact 1-move and 2-move deterministic lethal
       clears of the final remaining jellies, blockers, or ingredients.
     - Dynamically interpolates between Deterministic Lethal Sniper Mode (when lethal_prob is high)
       and Maximum Cascade Volatility Miracle Mode (when a multi-cascade chain reaction is required).
"""

import math
from typing import Dict, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class CascadeVolatilityEntropyAnalyzer(nn.Module):
    """
    Computes spatial Chain-Reaction Entropy and Cascade Volatility features:
      1. Bottom-Row Gravity Leverage: Deep rows (r -> 8) displace all rows 0..r above them.
      2. Staggered 1-Drop & 2-Drop Color Collision Density: Measures how many adjacent columns
         contain matching colors offset vertically by delta_r in {1, 2}, which automatically
         collide into secondary cascades when a match occurs below them!
      3. Sympathetic Special Detonation Field: Measures whether clearing a cell triggers
         chain-reaction blasts across nearby Striped/Wrapped/Color Bomb specials.
    """

    def __init__(self, height: int = 9, width: int = 9, d_out: int = 56):
        super().__init__()
        self.height = height
        self.width = width

        # Gravity avalanche leverage increases quadratically with row depth r in [0..8]
        depth_leverage = (torch.linspace(0.2, 1.0, steps=height) ** 1.5).view(1, 1, height, 1)
        depth_leverage = depth_leverage.expand(1, 1, height, width)

        # Lower-column cumulative sum matrix: sums column volatility strictly above or at row r
        above_mask = torch.tril(torch.ones(height, height, dtype=torch.float32), diagonal=0)

        self.register_buffer("depth_leverage", depth_leverage)
        self.register_buffer("above_mask", above_mask)

        self.proj = nn.Sequential(
            nn.Conv2d(8, d_out, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(d_out),
            nn.GELU(),
            nn.Conv2d(d_out, d_out, kernel_size=1),
            nn.GELU()
        )

    def forward(
        self, board_tensor: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            board_tensor: (B, 20, 9, 9)
        Returns:
            vol_feat: (B, d_out, 9, 9)
            chain_reaction_entropy_map: (B, 9, 9) expected cascade entropy triggered at (r, c)
            remaining_target_map: (B, 9, 9) spatial map of remaining endgame objectives
        """
        B, C, H, W = board_tensor.shape
        colors = board_tensor[:, :6]  # (B, 6, 9, 9)

        # 1. Staggered vertical color-collision potential:
        # Check if (r, c) has the same color as (r-1, c-1), (r-1, c+1), (r-2, c-1), or (r-2, c+1)
        c_pad = F.pad(colors, (1, 1, 2, 0))  # Pad left=1, right=1, top=2, bottom=0
        shift_u1_l = c_pad[:, :, 1:H + 1, 0:W]      # (r-1, c-1)
        shift_u1_r = c_pad[:, :, 1:H + 1, 2:W + 2]  # (r-1, c+1)
        shift_u2_l = c_pad[:, :, 0:H, 0:W]          # (r-2, c-1)
        shift_u2_r = c_pad[:, :, 0:H, 2:W + 2]      # (r-2, c+1)

        drop1_collision = (colors * (shift_u1_l + shift_u1_r)).sum(dim=1)  # (B, 9, 9)
        drop2_collision = (colors * (shift_u2_l + shift_u2_r)).sum(dim=1)  # (B, 9, 9)

        # Accumulate collision density above each cell (clearing row r shifts rows 0..r-1 down!)
        cascade_above_load = torch.einsum(
            "ij,bjw->biw", self.above_mask, drop1_collision + 0.6 * drop2_collision
        ) / float(H)

        # 2. Local color Shannon entropy across 3x3 patches (high color diversity vs clustering)
        local_color_prob = F.avg_pool2d(colors, kernel_size=3, stride=1, padding=1).clamp(1e-4, 1.0)
        shannon_entropy = -(local_color_prob * torch.log(local_color_prob)).sum(dim=1) / math.log(6.0)

        # 3. Special candy chain-detonation density (channels 6..9)
        special_chain = (
            1.2 * board_tensor[:, 6]
            + 1.2 * board_tensor[:, 7]
            + 1.6 * board_tensor[:, 8]
            + 2.5 * board_tensor[:, 9]
        )

        # 4. Remaining objective map (Jellies ch 15..16, Ingredients ch 17, Blockers ch 10..14)
        remaining_target_map = (
            1.5 * board_tensor[:, 15]
            + 2.5 * board_tensor[:, 16]
            + 2.0 * board_tensor[:, 17]
            + 0.8 * board_tensor[:, 14]
        )

        depth_plane = self.depth_leverage.expand(B, 1, H, W).squeeze(1)

        # Combined Chain-Reaction Entropy Map
        chain_reaction_entropy_map = (
            depth_plane * (0.45 * cascade_above_load + 0.35 * shannon_entropy)
            + 0.4 * special_chain
        )

        stack = torch.stack([
            depth_plane,
            drop1_collision,
            drop2_collision,
            cascade_above_load,
            shannon_entropy,
            special_chain,
            remaining_target_map,
            chain_reaction_entropy_map
        ], dim=1)  # (B, 8, 9, 9)

        vol_feat = self.proj(stack)
        return vol_feat, chain_reaction_entropy_map, remaining_target_map


class ClutchCascadeTransformerBlock(nn.Module):
    """
    Transformer Block with Cascade-Entropy & Lethal-Objective Gated Self-Attention.
    Propagates both 1-to-2 move lethal target positions and upward-to-downward cascade wave dynamics.
    """

    def __init__(self, d_model: int = 192, num_heads: int = 6):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)

        self.norm1 = nn.LayerNorm(d_model)
        self.qkv = nn.Linear(d_model, d_model * 3, bias=False)
        self.key_gate_proj = nn.Linear(2, num_heads, bias=False)
        self.out_proj = nn.Linear(d_model, d_model)

        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 3),
            nn.GELU(),
            nn.Linear(d_model * 3, d_model)
        )

    def forward(
        self,
        tokens: torch.Tensor,
        entropy_map: torch.Tensor,
        target_map: torch.Tensor
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

        # Bias keys by [chain_reaction_entropy, remaining_target_priority]
        key_feats = torch.stack([entropy_map.view(B, N), target_map.view(B, N)], dim=-1)  # (B, 81, 2)
        key_bias = self.key_gate_proj(key_feats).permute(0, 2, 1).unsqueeze(2)  # (B, H, 1, 81)

        attn = F.softmax(scores + key_bias, dim=-1)
        out = torch.matmul(attn, v).transpose(1, 2).contiguous().view(B, N, D)
        tokens = tokens + self.out_proj(out)
        tokens = tokens + self.ffn(self.norm2(tokens))
        return tokens


class OneToTwoMoveLethalDetector(nn.Module):
    """
    2-Step Latent Lookahead & Lethal Win Detector for <= 5 Moves Endgame States.
    Predicts:
      1. `lethal_1move_logits`: (B, 144) logit score that action `a` immediately wins in 1 move
      2. `lethal_2move_prob`: (B, 1) probability that an exact 1-to-2 move lethal sequence exists
      3. `cascade_volatility_scores`: (B, 144) predicted upper-tail cascade variance sigma_cascade(s, a)
    """

    def __init__(self, d_model: int = 192, num_actions: int = 144):
        super().__init__()
        self.d_model = d_model
        self.num_actions = num_actions

        # Pairwise swap encoder for all 144 actions
        self.horiz_pair_proj = nn.Linear(d_model * 2 + 4, d_model)
        self.vert_pair_proj = nn.Linear(d_model * 2 + 4, d_model)

        # 1-move lethal action classifier + 2-move latent lookahead scorer
        self.lethal_1move_head = nn.Sequential(
            nn.Linear(d_model * 2, 128),
            nn.GELU(),
            nn.Linear(128, 1)
        )
        # 2-step latent transition simulator over the top candidate representations
        self.step2_lookahead_mlp = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Linear(d_model, 64),
            nn.GELU(),
            nn.Linear(64, 1)
        )
        # Cascade volatility head predicting standard deviation / upper-tail chain length per action
        self.volatility_head = nn.Sequential(
            nn.Linear(d_model * 2, 128),
            nn.GELU(),
            nn.Linear(128, 1),
            nn.Softplus()
        )

    def forward(
        self,
        z_2d: torch.Tensor,
        global_ctx: torch.Tensor,
        entropy_map: torch.Tensor,
        target_map: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        B, D, H, W = z_2d.shape

        # Encode 72 horizontal swap pairs
        h_l = z_2d[:, :, :, :-1].permute(0, 2, 3, 1)
        h_r = z_2d[:, :, :, 1:].permute(0, 2, 3, 1)
        h_aux = torch.stack([
            entropy_map[:, :, :-1], entropy_map[:, :, 1:],
            target_map[:, :, :-1], target_map[:, :, 1:]
        ], dim=-1)
        h_emb = F.gelu(self.horiz_pair_proj(torch.cat([h_l, h_r, h_aux], dim=-1).reshape(B, 72, 2 * D + 4)))

        # Encode 72 vertical swap pairs
        v_t = z_2d[:, :, :-1, :].permute(0, 2, 3, 1)
        v_b = z_2d[:, :, 1:, :].permute(0, 2, 3, 1)
        v_aux = torch.stack([
            entropy_map[:, :-1, :], entropy_map[:, 1:, :],
            target_map[:, :-1, :], target_map[:, 1:, :]
        ], dim=-1)
        v_emb = F.gelu(self.vert_pair_proj(torch.cat([v_t, v_b, v_aux], dim=-1).reshape(B, 72, 2 * D + 4)))

        action_embs = torch.cat([h_emb, v_emb], dim=1)  # (B, 144, D)
        ctx_expanded = global_ctx.unsqueeze(1).expand(B, self.num_actions, D)
        joint_action_ctx = torch.cat([action_embs, ctx_expanded], dim=-1)  # (B, 144, 2*D)

        # 1. Immediate 1-Move Lethal Logits & Probabilities
        lethal_1move_logits = self.lethal_1move_head(joint_action_ctx).squeeze(-1)  # (B, 144)
        lethal_1move_probs = torch.sigmoid(lethal_1move_logits)

        # 2. 2-Move Lethal Lookahead Score per action & board-level 1-to-2 move lethal probability
        lethal_2move_action_logits = self.step2_lookahead_mlp(joint_action_ctx).squeeze(-1)  # (B, 144)
        lethal_2move_prob = torch.sigmoid(lethal_2move_action_logits.max(dim=-1, keepdim=True).values)  # (B, 1)

        # 3. Cascade Volatility Score sigma_cascade(s, a) per action
        cascade_volatility_scores = self.volatility_head(joint_action_ctx).squeeze(-1)  # (B, 144)

        return (
            lethal_1move_logits + 0.6 * lethal_2move_action_logits,
            lethal_1move_probs,
            lethal_2move_prob,
            cascade_volatility_scores,
        )


class EndgameClutchMiracleNet(nn.Module):
    """
    EndgameClutchMiracleNet: Specialized Deep Architecture for Endgame Clutch States (<= 5 Moves Left).
    Dynamically arbitrates between Exact 1-to-2 Move Lethal Win Detection and Maximum Cascade
    Volatility & Chain-Reaction Entropy (Hail-Mary Miracle Mode).
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

        # 1. Cascade Volatility & Chain-Reaction Entropy Analyzer
        vol_dim = 56
        self.volatility_analyzer = CascadeVolatilityEntropyAnalyzer(
            height=board_h, width=board_w, d_out=vol_dim
        )

        # 2. Stem + Positional Encoding
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels + vol_dim, d_model, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(d_model),
            nn.GELU(),
            nn.Conv2d(d_model, d_model, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(d_model),
            nn.GELU()
        )
        self.pos_embed = nn.Parameter(torch.randn(1, self.num_nodes, d_model) * 0.02)

        # 3. 6-Layer Clutch Cascade Transformer Backbone
        self.layers = nn.ModuleList([
            ClutchCascadeTransformerBlock(d_model=d_model, num_heads=num_heads)
            for _ in range(num_layers)
        ])
        self.final_norm = nn.LayerNorm(d_model)

        # 4. 1-to-2 Move Lethal Detector & Cascade Volatility Scorer
        self.lethal_detector = OneToTwoMoveLethalDetector(d_model=d_model, num_actions=num_actions)

        self.global_policy_head = nn.Sequential(
            nn.Conv2d(d_model, 96, kernel_size=3, padding=1),
            nn.BatchNorm2d(96),
            nn.GELU(),
            nn.Flatten(),
            nn.Linear(96 * self.num_nodes, 256),
            nn.GELU(),
            nn.Linear(256, num_actions)
        )

        # 5. Value & Win Probability Readout Heads
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

    def forward(
        self,
        board_tensor: torch.Tensor,
        legal_mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        B, C, H, W = board_tensor.shape

        # Step 1: Extract Cascade Volatility & Chain-Reaction Entropy features
        vol_feat, chain_reaction_entropy_map, remaining_target_map = (
            self.volatility_analyzer(board_tensor)
        )

        # Step 2: Stem + Clutch Cascade Transformer Layers
        stem_2d = self.stem(torch.cat([board_tensor, vol_feat], dim=1))
        tokens = stem_2d.flatten(2).transpose(1, 2) + self.pos_embed

        for layer in self.layers:
            tokens = layer(tokens, chain_reaction_entropy_map, remaining_target_map)

        tokens = self.final_norm(tokens)
        z_2d = tokens.transpose(1, 2).view(B, self.d_model, H, W)
        global_ctx = tokens.mean(dim=1)  # (B, d_model)

        # Step 3: Run 1-to-2 Move Lethal Win Detector & Cascade Volatility Scorer
        (
            lethal_logits,
            lethal_1move_probs,
            lethal_2move_prob,
            cascade_volatility_scores,
        ) = self.lethal_detector(z_2d, global_ctx, chain_reaction_entropy_map, remaining_target_map)

        # Step 4: Dynamic Clutch Policy Arbitration:
        # When lethal_2move_prob is high, prioritize exact lethal_logits;
        # When lethal_2move_prob is low, boost Upper-Tail Cascade Volatility (Hail-Mary Miracle Mode)!
        miracle_beta = 1.5 * (1.0 - lethal_2move_prob) + 0.35
        base_logits = self.global_policy_head(z_2d)
        policy_logits = (
            base_logits
            + (1.0 + lethal_2move_prob) * lethal_logits
            + miracle_beta * cascade_volatility_scores
        )

        if legal_mask is not None:
            policy_logits = policy_logits.masked_fill(~legal_mask.bool(), -1e9)

        # Step 5: Global Readout
        max_entropy = chain_reaction_entropy_map.flatten(1).max(dim=-1, keepdim=True).values
        readout = torch.cat([global_ctx, tokens.max(dim=1).values, lethal_2move_prob, max_entropy], dim=-1)
        value = self.value_head(readout)
        win_prob = self.win_head(readout)

        return {
            "policy_logits": policy_logits,
            "value": value,
            "win_prob": win_prob,
            "lethal_1move_action_probs": lethal_1move_probs,
            "lethal_2move_prob": lethal_2move_prob,
            "cascade_volatility_scores": cascade_volatility_scores,
            "chain_reaction_entropy_map": chain_reaction_entropy_map,
        }

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def create_endgame_clutch_specialist(device: str = "cpu") -> EndgameClutchMiracleNet:
    model = EndgameClutchMiracleNet()
    return model.to(device)


if __name__ == "__main__":
    torch.manual_seed(42)
    model = create_endgame_clutch_specialist("cpu")
    params = model.count_parameters()
    print(f"[EndgameClutchMiracleNet] Trainable Parameters: {params:,} ({params / 1e6:.2f}M)")

    dummy_board = torch.zeros(2, 20, 9, 9)
    dummy_board[:, 18] = 1.0   # Active grid
    dummy_board[:, 19] = 0.08  # <= 3 moves remaining (endgame clutch!)
    dummy_board[0, 15, 7, 4] = 1.0  # Last remaining jelly at (7, 4)
    dummy_board[0, 0, 7, [2, 3]] = 1.0
    dummy_board[0, 0, 6, 4] = 1.0   # 1-move lethal match at (7, 4)!

    out = model(dummy_board)
    for k, v in out.items():
        print(f"  {k}: {tuple(v.shape)}")
