"""
2D Gravity-Aware Retentive Spatial Network (`RetNetCandy2D`) for Candy Crush Saga.
Based on:
  - Sun et al. (Microsoft Research 2023), "Retentive Network: A Successor to Transformer" (arXiv:2307.08621)
  - Fan et al. (CVPR 2024), "RMT: Retentive Networks Meet Vision Transformers"

Replaces softmax attention with Multi-Scale 2D Manhattan + Asymmetric Gravity Retention:
    Retention_h(X) = ((Q_h K_h^T) * D^{(h)}_{gravity}) V_h
where D^{(h)}_{nm} = gamma_h^{dist_{gravity}(n, m)} encodes both 2D spatial proximity,
downward gravity cascade flow, and orthogonal row/column blast-line alignment.
Supports both full parallel representation evaluation and O(1) row-recurrent state updates.
"""

import math
from typing import Dict, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiScale2DGravityDecayMask(nn.Module):
    """
    Constructs the multi-head 2D Manhattan + Asymmetric Gravity + Blast-Line Decay Matrix
    D in R^{H x 81 x 81} where:
        dist_{gravity}((r_n, c_n), (r_m, c_m)) =
            w_up * max(0, r_n - r_m) + w_down * max(0, r_m - r_n) + |c_n - c_m|
            - w_blast * I(r_n == r_m or c_n == c_m)
    and D_{h, n, m} = gamma_h^{dist_{gravity}(n, m)}.
    """
    def __init__(self, num_heads: int = 8, board_h: int = 9, board_w: int = 9):
        super().__init__()
        self.num_heads = num_heads
        self.board_h = board_h
        self.board_w = board_w
        self.num_tokens = board_h * board_w

        # Multi-scale retention decay rates gamma_h = 1 - 2^{-3 - h} in (0.75, 0.996)
        gammas = 1.0 - torch.pow(2.0, -3.0 - torch.arange(num_heads, dtype=torch.float32))
        self.register_buffer("gammas", gammas.view(num_heads, 1, 1))

        # Precompute 2D grid coordinates for all 81 cells
        coords = torch.stack(
            torch.meshgrid(
                torch.arange(board_h, dtype=torch.float32),
                torch.arange(board_w, dtype=torch.float32),
                indexing="ij"
            ),
            dim=-1
        ).view(-1, 2)  # (81, 2)

        r_diff = coords[:, 0].unsqueeze(1) - coords[:, 0].unsqueeze(0)  # (81, 81): r_n - r_m
        c_diff = (coords[:, 1].unsqueeze(1) - coords[:, 1].unsqueeze(0)).abs()  # (81, 81): |c_n - c_m|

        # Upward vs downward directional components (gravity pulls from r_m < r_n downward to r_n)
        upstream_dist = F.relu(r_diff)      # source m is above target n (gravity flows m -> n)
        downstream_dist = F.relu(-r_diff)   # source m is below target n (against gravity)

        # Orthogonal row/column blast-line indicator (Striped candy ray alignment)
        same_row_or_col = ((r_diff.abs() < 1e-4) | (c_diff < 1e-4)).float()

        self.register_buffer("upstream_dist", upstream_dist.unsqueeze(0))      # (1, 81, 81)
        self.register_buffer("downstream_dist", downstream_dist.unsqueeze(0))  # (1, 81, 81)
        self.register_buffer("col_dist", c_diff.unsqueeze(0))                  # (1, 81, 81)
        self.register_buffer("same_row_or_col", same_row_or_col.unsqueeze(0))  # (1, 81, 81)

        # Learnable per-head gravity anisotropy weights (initialized so upstream gravity decays slower)
        self.w_upstream = nn.Parameter(torch.ones(num_heads, 1, 1) * 0.65)
        self.w_downstream = nn.Parameter(torch.ones(num_heads, 1, 1) * 1.15)
        self.w_blast_bonus = nn.Parameter(torch.ones(num_heads, 1, 1) * 0.35)

    def forward(self) -> torch.Tensor:
        """Returns multi-scale 2D decay matrix D of shape (1, H, 81, 81)."""
        eff_dist = (
            F.softplus(self.w_upstream) * self.upstream_dist +
            F.softplus(self.w_downstream) * self.downstream_dist +
            self.col_dist -
            torch.sigmoid(self.w_blast_bonus) * self.same_row_or_col
        ).clamp(min=0.0)

        decay_matrix = torch.pow(self.gammas, eff_dist / 2.0)  # (H, 81, 81)
        # Normalize rows by L1/L2 factor as in Sun et al. (2023) Eq. 5 for numerical stability
        row_norm = decay_matrix.sum(dim=-1, keepdim=True).clamp(min=1.0).sqrt()
        decay_matrix = decay_matrix / row_norm
        return decay_matrix.unsqueeze(0)  # (1, H, 81, 81)


class MultiScale2DRetentionBlock(nn.Module):
    """
    2D Retentive Block implementing:
      1. Parallel 2D Retention with GroupNorm + Swish Gate (Sun et al., 2023)
      2. O(1) Row-Recurrent Retention state update from top row (r=0) to bottom row (r=8)
      3. Local 3x3 depthwise convolution + Gated FFN (SwiGLU)
    """
    def __init__(
        self,
        d_model: int = 256,
        num_heads: int = 8,
        board_h: int = 9,
        board_w: int = 9
    ):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.board_h = board_h
        self.board_w = board_w

        self.norm1 = nn.LayerNorm(d_model)
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.g_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

        # RetNet GroupNorm over heads (Sun et al., Section 2.2)
        self.group_norm = nn.GroupNorm(num_heads, d_model)
        self.decay_mask_gen = MultiScale2DGravityDecayMask(
            num_heads=num_heads,
            board_h=board_h,
            board_w=board_w
        )

        # Local 3x3 spatial convolution + SwiGLU Feed-Forward Network
        self.norm2 = nn.LayerNorm(d_model)
        self.dw_conv = nn.Conv2d(d_model, d_model, kernel_size=3, padding=1, groups=d_model, bias=False)
        self.ffn_gate = nn.Linear(d_model, d_model * 2)
        self.ffn_val = nn.Linear(d_model, d_model * 2)
        self.ffn_out = nn.Linear(d_model * 2, d_model)

    def forward_parallel(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Parallel 2D Retention pass over all 81 board tokens.
        Args:
            x: (B, 81, d_model)
        Returns:
            x_out: (B, 81, d_model)
            decay_matrix: (1, H, 81, 81)
        """
        B, N, D = x.shape
        x_norm = self.norm1(x)

        q = self.q_proj(x_norm).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x_norm).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x_norm).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        gate = F.silu(self.g_proj(x_norm))  # (B, N, D)

        # Scale queries and keys
        q = q * (self.head_dim ** -0.5)
        qk = torch.matmul(q, k.transpose(-2, -1))  # (B, H, 81, 81)

        decay_matrix = self.decay_mask_gen()  # (1, H, 81, 81)
        ret_weights = qk * decay_matrix
        ret_out = torch.matmul(ret_weights, v)  # (B, H, 81, head_dim)

        # Reshape and apply head-wise GroupNorm + Swish Gate
        ret_out = ret_out.transpose(1, 2).contiguous().view(B * N, D)
        ret_out = self.group_norm(ret_out).view(B, N, D)
        ret_out = self.out_proj(ret_out * gate)

        x = x + ret_out

        # Local depthwise conv + SwiGLU FFN
        x_norm2 = self.norm2(x)
        x_2d = x_norm2.transpose(1, 2).contiguous().view(B, D, self.board_h, self.board_w)
        x_local = self.dw_conv(x_2d).flatten(2).transpose(1, 2) + x_norm2
        ffn_out = self.ffn_out(F.silu(self.ffn_gate(x_local)) * self.ffn_val(x_local))
        x = x + ffn_out

        return x, decay_matrix

    def forward_recurrent_row_scan(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Chunkwise / Row-by-Row Recurrent Gravity Scan (top row r=0 -> bottom row r=8)
        maintaining an O(1) recurrent state tensor S_r in R^{B x H x head_dim x head_dim}:
            S_r = gamma_h * S_{r-1} + K_r^T V_r
            O_r = Q_r S_r
        """
        B, N, D = x.shape
        x_norm = self.norm1(x).view(B, self.board_h, self.board_w, D)
        gammas = self.decay_mask_gen.gammas.view(1, self.num_heads, 1, 1)

        state = torch.zeros(
            (B, self.num_heads, self.head_dim, self.head_dim),
            device=x.device,
            dtype=x.dtype
        )
        row_outputs = []
        for r in range(self.board_h):
            row_tokens = x_norm[:, r, :, :]  # (B, W, D)
            q_r = self.q_proj(row_tokens).view(B, self.board_w, self.num_heads, self.head_dim).transpose(1, 2)
            k_r = self.k_proj(row_tokens).view(B, self.board_w, self.num_heads, self.head_dim).transpose(1, 2)
            v_r = self.v_proj(row_tokens).view(B, self.board_w, self.num_heads, self.head_dim).transpose(1, 2)

            q_r = q_r * (self.head_dim ** -0.5)
            # Update recurrent retention state S_r = gamma * S_{r-1} + K_r^T @ V_r
            kv_r = torch.matmul(k_r.transpose(-2, -1), v_r) / float(self.board_w)
            state = gammas * state + kv_r
            out_r = torch.matmul(q_r, state)  # (B, H, W, head_dim)
            row_outputs.append(out_r.transpose(1, 2).contiguous().view(B, self.board_w, D))

        ret_seq = torch.cat(row_outputs, dim=1)  # (B, 81, D)
        ret_seq = self.group_norm(ret_seq.view(B * N, D)).view(B, N, D)
        gate = F.silu(self.g_proj(self.norm1(x)))
        x_out = x + self.out_proj(ret_seq * gate)
        return x_out, state


class RetNetCandy2D(nn.Module):
    """
    2D Gravity-Aware Retentive Network (`RetNetCandy2D`) for Candy Crush Saga.
    Combines parallel 2D spatial retention training with O(1) row-recurrent gravity state
    scanning. Accepts canonical (B, 20, 9, 9) board states and outputs (B, 144) swap logits
    and (B, 1) win probability / value estimates.
    """
    def __init__(
        self,
        in_channels: int = 20,
        d_model: int = 256,
        num_layers: int = 6,
        num_heads: int = 8,
        num_actions: int = 144,
        board_h: int = 9,
        board_w: int = 9
    ):
        super().__init__()
        self.in_channels = in_channels
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.num_actions = num_actions
        self.board_h = board_h
        self.board_w = board_w
        self.num_tokens = board_h * board_w

        # Spatial convolutional stem
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, d_model, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(d_model),
            nn.SiLU(),
            nn.Conv2d(d_model, d_model, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(d_model),
            nn.SiLU()
        )
        self.pos_embed = nn.Parameter(torch.randn(1, self.num_tokens, d_model) * 0.02)

        # Stack of 2D Gravity-Aware Retentive Blocks
        self.retnet_blocks = nn.ModuleList([
            MultiScale2DRetentionBlock(
                d_model=d_model,
                num_heads=num_heads,
                board_h=board_h,
                board_w=board_w
            )
            for _ in range(num_layers)
        ])
        self.final_norm = nn.LayerNorm(d_model)

        # Policy Head over 144 canonical swaps
        self.policy_head = nn.Sequential(
            nn.Conv2d(d_model, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.SiLU(),
            nn.Conv2d(128, 64, kernel_size=1),
            nn.SiLU(),
            nn.Flatten(),
            nn.Linear(64 * self.num_tokens, 256),
            nn.SiLU(),
            nn.Linear(256, num_actions)
        )

        # Value / Win Probability Head
        self.value_head = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.SiLU(),
            nn.Linear(128, 1)
        )

    def forward(
        self,
        board_tensor: torch.Tensor,
        legal_mask: Optional[torch.Tensor] = None,
        use_recurrent_scan: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Canonical forward pass accepting (B, 20, 9, 9) board tensor.
        Supports both parallel 2D retention (`use_recurrent_scan=False`) and O(1) row-recurrent
        gravity state propagation (`use_recurrent_scan=True`).
        """
        B, C, H, W = board_tensor.shape
        tokens = self.stem(board_tensor).flatten(2).transpose(1, 2) + self.pos_embed  # (B, 81, d_model)

        last_decay_matrix = None
        last_recurrent_state = None
        for block in self.retnet_blocks:
            if use_recurrent_scan:
                tokens, last_recurrent_state = block.forward_recurrent_row_scan(tokens)
            else:
                tokens, last_decay_matrix = block.forward_parallel(tokens)

        z = self.final_norm(tokens)  # (B, 81, d_model)
        z_2d = z.transpose(1, 2).contiguous().view(B, self.d_model, H, W)

        policy_logits = self.policy_head(z_2d)
        if legal_mask is not None:
            policy_logits = policy_logits.masked_fill(~legal_mask.bool(), -1e9)

        z_global = z.mean(dim=1)  # (B, d_model)
        val_raw = self.value_head(z_global)  # (B, 1)
        win_prob = torch.sigmoid(val_raw)
        value = torch.tanh(val_raw)

        if last_decay_matrix is None:
            last_decay_matrix = self.retnet_blocks[-1].decay_mask_gen()

        return {
            "policy_logits": policy_logits,
            "win_prob": win_prob,
            "value": value,
            "gravity_decay_matrix": last_decay_matrix,
            "recurrent_state": last_recurrent_state,
            "latent_z": z
        }

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def create_retnet_candy_2d(device: str = "cpu") -> RetNetCandy2D:
    """Factory function initializing RetNetCandy2D on the target device."""
    model = RetNetCandy2D(
        in_channels=20,
        d_model=256,
        num_layers=6,
        num_heads=8,
        num_actions=144
    )
    return model.to(device)


if __name__ == "__main__":
    torch.manual_seed(42)
    net = create_retnet_candy_2d("cpu")
    params = net.count_parameters()
    print(f"[RetNetCandy2D] Initialized with {params:,} trainable parameters ({params / 1e6:.2f}M)")

    dummy_board = torch.randn(2, 20, 9, 9)
    out_parallel = net(dummy_board, use_recurrent_scan=False)
    out_recurrent = net(dummy_board, use_recurrent_scan=True)
    print(f"  policy_logits (parallel):  {out_parallel['policy_logits'].shape}")
    print(f"  policy_logits (recurrent): {out_recurrent['policy_logits'].shape}")
    print(f"  win_prob:                  {out_parallel['win_prob'].shape}")
    print(f"  value:                     {out_parallel['value'].shape}")
    print(f"  gravity_decay_matrix:      {out_parallel['gravity_decay_matrix'].shape}")
