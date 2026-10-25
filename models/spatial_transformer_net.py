"""
AlphaCandy Spatial Transformer & Graph-Attention Architecture for Candy Crush Saga.
Features Multi-Head Spatial Self-Attention across non-Euclidean board topologies,
disconnected islands, and long-range cascading flows.
Dual-head outputs: Policy Distribution P(a|s) over 144 canonical swaps,
and Distributional Value V(s) / P(Win|s).
"""

from typing import Tuple, List, Optional, Dict, Any
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from core.reverse_engine import (
    ReverseEngineState, Direction, OFFSETS, CandyColor, SpecialCandy, Blocker, Ingredient, LevelType
)

NUM_CANONICAL_ACTIONS = 144  # 72 horizontal (9x8) + 72 vertical (8x9)


def encode_action_index(r: int, c: int, direction: Direction, height: int = 9, width: int = 9) -> int:
    """
    Encodes a swap (r, c, direction) into a discrete canonical action index in [0, 143].
    - Horizontal swaps (RIGHT): r * (width - 1) + c  -> [0, 71]
    - Vertical swaps (DOWN): 72 + r * width + c       -> [72, 143]
    Swaps LEFT or UP are mapped to their canonical RIGHT or DOWN counterparts.
    """
    if direction == Direction.LEFT:
        r, c = r, c - 1
        direction = Direction.RIGHT
    elif direction == Direction.UP:
        r, c = r - 1, c
        direction = Direction.DOWN

    if direction == Direction.RIGHT:
        return r * (width - 1) + c
    elif direction == Direction.DOWN:
        return 72 + r * width + c
    raise ValueError(f"Invalid direction: {direction}")


def decode_action_index(idx: int, height: int = 9, width: int = 9) -> Tuple[int, int, Direction]:
    """Decodes canonical action index [0, 143] into (r, c, Direction)."""
    if idx < 72:
        r = idx // (width - 1)
        c = idx % (width - 1)
        return r, c, Direction.RIGHT
    else:
        v_idx = idx - 72
        r = v_idx // width
        c = v_idx % width
        return r, c, Direction.DOWN


def encode_spatial_board_tensor(state: ReverseEngineState) -> np.ndarray:
    """
    Encodes full board state into a 20-channel 9x9 spatial feature tensor:
    Channels 0..5   : Regular Candy Colors (1..6)
    Channels 6..9   : Special Candies (Striped H, Striped V, Wrapped, Color Bomb)
    Channels 10..14 : Blockers (Chocolate, Licorice, Frosting layers, Other)
    Channels 15..16 : Jellies (Single layer, Double layer)
    Channel 17      : Ingredients (Cherries/Hazelnuts)
    Channel 18      : Active Cell Mask (1=Playable, 0=Void/Hole)
    Channel 19      : Normalized Moves Remaining & Objectives Remaining
    """
    h, w = state.height, state.width
    tensor = np.zeros((20, h, w), dtype=np.float32)

    # 1. Colors (0..5)
    for color_val in range(1, 7):
        tensor[color_val - 1] = (state.colors == color_val).astype(np.float32)

    # 2. Specials (6..9)
    tensor[6] = (state.specials == SpecialCandy.STRIPED_HORIZONTAL).astype(np.float32)
    tensor[7] = (state.specials == SpecialCandy.STRIPED_VERTICAL).astype(np.float32)
    tensor[8] = (state.specials == SpecialCandy.WRAPPED).astype(np.float32)
    tensor[9] = (state.specials == SpecialCandy.COLOR_BOMB).astype(np.float32)

    # 3. Blockers (10..14)
    tensor[10] = (state.blockers == Blocker.CHOCOLATE).astype(np.float32)
    tensor[11] = (state.blockers == Blocker.LICORICE).astype(np.float32)
    tensor[12] = (state.blockers == Blocker.FROSTING).astype(np.float32)
    tensor[13] = (state.blockers >= 4).astype(np.float32)
    tensor[14] = (state.blockers != Blocker.NONE).astype(np.float32)

    # 4. Jellies (15..16)
    tensor[15] = (state.jellies == 1).astype(np.float32)
    tensor[16] = (state.jellies >= 2).astype(np.float32)

    # 5. Ingredients (17)
    tensor[17] = (state.ingredients != Ingredient.NONE).astype(np.float32)

    # 6. Active Grid Mask (18)
    tensor[18] = state.active_grid.astype(np.float32)

    # 7. Normalized Dynamics Context (19)
    max_m = max(1, getattr(state.config, "moves_limit", 30))
    moves_norm = state.moves_left / float(max_m)
    obj_norm = state.remaining_objectives_count() / 60.0
    tensor[19] = (moves_norm * 0.5) + (obj_norm * 0.5)

    return tensor


class SpatialAttentionBlock(nn.Module):
    """
    Multi-Head Spatial Self-Attention over board cells.
    Allows distant cells (across voids or portals) to directly exchange contextual information.
    """

    def __init__(self, channels: int = 128, num_heads: int = 4):
        super().__init__()
        self.num_heads = num_heads
        self.channels = channels
        self.head_dim = channels // num_heads

        self.q_proj = nn.Conv2d(channels, channels, kernel_size=1)
        self.k_proj = nn.Conv2d(channels, channels, kernel_size=1)
        self.v_proj = nn.Conv2d(channels, channels, kernel_size=1)
        self.out_proj = nn.Conv2d(channels, channels, kernel_size=1)
        self.norm = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        b, c, h, w = x.shape
        seq_len = h * w

        q = self.q_proj(x).view(b, self.num_heads, self.head_dim, seq_len).permute(0, 1, 3, 2)
        k = self.k_proj(x).view(b, self.num_heads, self.head_dim, seq_len).permute(0, 1, 3, 2)
        v = self.v_proj(x).view(b, self.num_heads, self.head_dim, seq_len).permute(0, 1, 3, 2)

        # Scaled Dot-Product Attention: (b, heads, seq, seq)
        scores = torch.matmul(q, k.transpose(-2, -1)) / np.sqrt(self.head_dim)

        if mask is not None:
            # Flatten spatial mask (b, 1, 1, seq_len)
            mask_flat = mask.view(b, 1, 1, seq_len)
            scores = scores.masked_fill(mask_flat == 0, -1e9)

        attn_weights = F.softmax(scores, dim=-1)
        out = torch.matmul(attn_weights, v)  # (b, heads, seq, head_dim)
        out = out.permute(0, 1, 3, 2).contiguous().view(b, c, h, w)

        return self.norm(x + self.out_proj(out))


class ResidualConvBlock(nn.Module):
    """Convolutional residual block with Mish activation and BatchNorm."""

    def __init__(self, channels: int = 128):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = F.mish(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.mish(out + residual)


class AlphaCandySpatialNet(nn.Module):
    """
    Super-Intelligent Deep Match-3 Dual-Head Neural Architecture.
    Combines Conv ResNet representation with Multi-Head Spatial Attention (Transformer)
    for non-Euclidean board reasoning, cascade propagation, and combo discovery.
    """

    def __init__(self, in_channels: int = 20, hidden_dim: int = 128, num_actions: int = 144):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_dim = hidden_dim
        self.num_actions = num_actions

        # Initial Feature Stem
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.Mish()
        )

        # Stage 1: Local Feature Processing
        self.res1 = ResidualConvBlock(hidden_dim)
        self.res2 = ResidualConvBlock(hidden_dim)

        # Stage 2: Spatial Multi-Head Self-Attention (Global Cross-Island Context)
        self.attention = SpatialAttentionBlock(channels=hidden_dim, num_heads=4)

        # Stage 3: High-Order Non-Local Processing
        self.res3 = ResidualConvBlock(hidden_dim)
        self.res4 = ResidualConvBlock(hidden_dim)

        # --- Policy Head: P(a|s) over 144 canonical swaps ---
        self.policy_conv = nn.Sequential(
            nn.Conv2d(hidden_dim, 64, kernel_size=1),
            nn.BatchNorm2d(64),
            nn.Mish()
        )
        self.policy_fc = nn.Sequential(
            nn.Linear(64 * 9 * 9, 256),
            nn.Mish(),
            nn.Dropout(0.1),
            nn.Linear(256, num_actions)
        )

        # --- Distributional Value Head: V(s) in [-1, 1] + P(Win|s) in [0, 1] ---
        self.value_conv = nn.Sequential(
            nn.Conv2d(hidden_dim, 32, kernel_size=1),
            nn.BatchNorm2d(32),
            nn.Mish()
        )
        self.value_fc = nn.Sequential(
            nn.Linear(32 * 9 * 9, 128),
            nn.Mish(),
            nn.Linear(128, 2)  # [0]: expected value scalar, [1]: win probability logit
        )

    def forward(self, x: torch.Tensor, board_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        Returns:
            policy_logits: (B, 144) unmasked action logits
            value_scalar : (B, 1) expected utility in [-1, 1]
            win_prob     : (B, 1) estimated win probability in [0, 1]
        """
        feat = self.stem(x)
        feat = self.res1(feat)
        feat = self.res2(feat)

        # Spatial Self-Attention across 81 cells
        feat = self.attention(feat, mask=board_mask)

        feat = self.res3(feat)
        feat = self.res4(feat)

        # Policy Head
        p_feat = self.policy_conv(feat).flatten(1)
        policy_logits = self.policy_fc(p_feat)

        # Value Head
        v_feat = self.value_conv(feat).flatten(1)
        v_out = self.value_fc(v_feat)
        value_scalar = torch.tanh(v_out[:, 0:1])
        win_prob = torch.sigmoid(v_out[:, 1:2])

        return policy_logits, value_scalar, win_prob

    def forward_with_mask(
        self, x: torch.Tensor, legal_moves: List[Tuple[int, int, Direction]]
    ) -> Tuple[torch.Tensor, torch.Tensor, float, float]:
        """
        Evaluates a single state tensor with strict legal action masking.
        Returns:
            masked_logits : (144,) masked logits (-inf for illegal)
            probs         : (144,) softmax action probabilities
            val_scalar    : float in [-1, 1]
            win_p         : float in [0, 1]
        """
        self.eval()
        with torch.no_grad():
            logits, val_scalar, win_prob = self.forward(x)
            logits_1d = logits.squeeze(0)

            # Build action mask
            mask = torch.full_like(logits_1d, -float("inf"))
            for r, c, d in legal_moves:
                a_idx = encode_action_index(r, c, d)
                if 0 <= a_idx < self.num_actions:
                    mask[a_idx] = 0.0

            masked_logits = logits_1d + mask
            probs = F.softmax(masked_logits, dim=-1)

        return masked_logits, probs, float(val_scalar.item()), float(win_prob.item())
