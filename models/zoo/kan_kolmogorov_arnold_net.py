"""
Kolmogorov-Arnold Network (KAN) for Candy Crush Saga (`KolmogorovArnoldCandyNet`).
Based on:
  - Liu et al. (2024), "KAN: Kolmogorov-Arnold Networks" (arXiv:2404.19756)
  - Li (2024), "FastKAN: Very Fast Kolmogorov-Arnold Networks using Gaussian Radial Basis Functions"

Replaces fixed-activation linear MLP layers in spatial token-channel mixers and policy/value
heads with learnable univariate B-spline / Gaussian Radial Basis Function (RBF) activation
grids (`FastKANLayer`), modeling sharp nonlinear blocker-cascade phase transitions with high
parameter efficiency.
"""

import math
from typing import Dict, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class FastKANLayer(nn.Module):
    """
    Learnable Univariate Basis Expansion Layer implementing the Kolmogorov-Arnold theorem:
        Phi(x)_j = sum_i [ w_base_{j,i} * SiLU(x_i) + sum_{k=1}^G c_{j,i,k} * B_k(LayerNorm(x)_i) ]
    where B_k(u) = exp(-((u - mu_k) / h)^2) are Gaussian Radial Basis Function (RBF) / B-spline
    equivalents anchored on a uniform knot grid over [grid_min, grid_max].
    """
    def __init__(
        self,
        in_features: int,
        out_features: int,
        grid_size: int = 8,
        grid_min: float = -2.5,
        grid_max: float = 2.5,
        use_base_update: bool = True,
        base_scale: float = 0.1,
        spline_scale: float = 0.1
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.grid_min = grid_min
        self.grid_max = grid_max
        self.use_base_update = use_base_update

        # Pre-normalization keeps input coordinates within the active spline grid domain
        self.norm = nn.LayerNorm(in_features)

        # Uniformly spaced knot centers mu_k in [grid_min, grid_max]
        grid = torch.linspace(grid_min, grid_max, grid_size)
        self.register_buffer("grid", grid)
        self.bandwidth = (grid_max - grid_min) / max(1, grid_size - 1)

        # Learnable spline control coefficients c_{out, in, grid_size}
        self.spline_weight = nn.Parameter(
            torch.randn(out_features, in_features * grid_size) * (spline_scale / math.sqrt(in_features * grid_size))
        )

        # Residual SiLU base linear path (Liu et al. Eq. 2.10)
        if use_base_update:
            self.base_linear = nn.Linear(in_features, out_features)
            nn.init.trunc_normal_(self.base_linear.weight, std=base_scale / math.sqrt(in_features))
            if self.base_linear.bias is not None:
                nn.init.zeros_(self.base_linear.bias)

    def compute_basis(self, x_norm: torch.Tensor) -> torch.Tensor:
        """
        Evaluates Gaussian RBF / B-spline basis functions on normalized input x_norm.
        Args:
            x_norm: (..., in_features)
        Returns:
            basis: (..., in_features * grid_size)
        """
        # Expand last dim against knot grid: (..., in_features, 1) - (grid_size,) -> (..., in_features, grid_size)
        diff = (x_norm.unsqueeze(-1) - self.grid) / self.bandwidth
        basis = torch.exp(-0.5 * diff.pow(2))
        return basis.flatten(start_dim=-2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (..., in_features)
        Returns:
            out: Output tensor of shape (..., out_features)
        """
        x_norm = self.norm(x)
        basis = self.compute_basis(x_norm)  # (..., in_features * grid_size)
        spline_out = F.linear(basis, self.spline_weight)

        if self.use_base_update:
            base_out = self.base_linear(F.silu(x))
            return base_out + spline_out
        return spline_out

    def l1_spline_regularization(self) -> torch.Tensor:
        """Sparsity and entropy penalty over spline coefficients (Liu et al., 2024)."""
        abs_w = self.spline_weight.abs()
        l1_norm = abs_w.mean()
        prob = abs_w / (abs_w.sum() + 1e-8)
        entropy = -(prob * torch.log(prob + 1e-8)).mean()
        return l1_norm + 0.1 * entropy


class SpatialKANMixerBlock(nn.Module):
    """
    2D Spatial-Channel Kolmogorov-Arnold Mixer Block for 9x9 match-3 boards.
    Combines depthwise local spatial convolution, token-wise KAN mixing across the 81 board cells,
    and channel-wise KAN spline mixing across feature channels.
    """
    def __init__(
        self,
        d_model: int = 192,
        num_tokens: int = 81,
        grid_size: int = 8
    ):
        super().__init__()
        self.d_model = d_model
        self.num_tokens = num_tokens

        # Local depthwise 3x3 match pattern filter
        self.local_dw_conv = nn.Conv2d(
            d_model, d_model, kernel_size=3, padding=1, groups=d_model, bias=False
        )
        self.local_bn = nn.BatchNorm2d(d_model)

        # Spatial Token KAN Mixer (mixes across the 81 grid cells)
        self.token_kan = FastKANLayer(
            in_features=num_tokens,
            out_features=num_tokens,
            grid_size=grid_size
        )

        # Channel Feature KAN Mixer (replaces standard 2-layer MLP with 2-layer KAN)
        self.channel_kan1 = FastKANLayer(
            in_features=d_model,
            out_features=d_model,
            grid_size=grid_size
        )
        self.channel_kan2 = FastKANLayer(
            in_features=d_model,
            out_features=d_model,
            grid_size=grid_size
        )

    def forward(self, x: torch.Tensor, board_h: int = 9, board_w: int = 9) -> torch.Tensor:
        """
        Args:
            x: Token tensor of shape (B, 81, d_model)
        Returns:
            x_out: Updated token tensor of shape (B, 81, d_model)
        """
        B, N, D = x.shape

        # 1. Local 2D spatial depthwise residual
        x_2d = x.transpose(1, 2).contiguous().view(B, D, board_h, board_w)
        local_feat = F.silu(self.local_bn(self.local_dw_conv(x_2d)))
        x = x + local_feat.flatten(2).transpose(1, 2)

        # 2. Non-local Spatial Token KAN Mixing: (B, D, 81) -> (B, D, 81)
        x_t = x.transpose(1, 2)
        x = x + self.token_kan(x_t).transpose(1, 2)

        # 3. Nonlinear Channel KAN Mixing: (B, 81, D) -> (B, 81, D)
        x = x + self.channel_kan2(self.channel_kan1(x))
        return x


class KolmogorovArnoldCandyNet(nn.Module):
    """
    Kolmogorov-Arnold Network (KAN) Architecture for Candy Crush Saga.
    Accepts canonical (B, 20, 9, 9) board tensors and outputs (B, 144) swap policy logits,
    (B, 1) win probability / value, spline activation statistics, and blocker synergy maps.
    """
    def __init__(
        self,
        in_channels: int = 20,
        d_model: int = 192,
        num_kan_blocks: int = 5,
        grid_size: int = 8,
        num_actions: int = 144,
        board_h: int = 9,
        board_w: int = 9
    ):
        super().__init__()
        self.in_channels = in_channels
        self.d_model = d_model
        self.num_kan_blocks = num_kan_blocks
        self.grid_size = grid_size
        self.num_actions = num_actions
        self.board_h = board_h
        self.board_w = board_w
        self.num_tokens = board_h * board_w

        # Input stem: per-cell KAN projection from 20 raw channels + 3x3 conv context
        self.conv_stem = nn.Sequential(
            nn.Conv2d(in_channels, d_model // 2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(d_model // 2),
            nn.SiLU()
        )
        self.cell_kan_stem = FastKANLayer(
            in_features=in_channels,
            out_features=d_model // 2,
            grid_size=grid_size
        )
        self.pos_embed = nn.Parameter(torch.randn(1, self.num_tokens, d_model) * 0.02)

        # Stacked Spatial-Channel KAN Mixer Backbone
        self.kan_blocks = nn.ModuleList([
            SpatialKANMixerBlock(
                d_model=d_model,
                num_tokens=self.num_tokens,
                grid_size=grid_size
            )
            for _ in range(num_kan_blocks)
        ])
        self.final_norm = nn.LayerNorm(d_model)

        # Policy Head using KAN univariate spline layers
        self.policy_spatial_reduce = nn.Sequential(
            nn.Conv2d(d_model, 32, kernel_size=1, bias=False),
            nn.BatchNorm2d(32),
            nn.SiLU()
        )
        self.policy_kan_head = nn.Sequential(
            FastKANLayer(32 * self.num_tokens, 256, grid_size=grid_size),
            FastKANLayer(256, num_actions, grid_size=grid_size)
        )

        # Value / Win Probability Head using KAN spline layers
        self.value_kan_head = nn.Sequential(
            FastKANLayer(d_model, 128, grid_size=grid_size),
            FastKANLayer(128, 1, grid_size=grid_size)
        )

        # Nonlinear Blocker-Cascade Synergy Map Head (predicts cell-level cascade potential)
        self.synergy_kan_head = FastKANLayer(d_model, 1, grid_size=grid_size)

    def encode_tokens(self, board_tensor: torch.Tensor) -> torch.Tensor:
        """Encodes (B, 20, 9, 9) board state into KAN latent tokens (B, 81, d_model)."""
        B, C, H, W = board_tensor.shape

        # Local 3x3 convolutional branch
        conv_feat = self.conv_stem(board_tensor).flatten(2).transpose(1, 2)  # (B, 81, d_model//2)

        # Pointwise KAN spline branch directly on the 20 board channels
        raw_tokens = board_tensor.flatten(2).transpose(1, 2)  # (B, 81, 20)
        kan_feat = self.cell_kan_stem(raw_tokens)  # (B, 81, d_model//2)

        tokens = torch.cat([conv_feat, kan_feat], dim=-1) + self.pos_embed

        for block in self.kan_blocks:
            tokens = block(tokens, board_h=H, board_w=W)

        return self.final_norm(tokens)

    def spline_regularization_loss(self) -> torch.Tensor:
        """Computes total L1 + entropy regularization across all FastKANLayer modules."""
        reg = torch.tensor(0.0, device=self.pos_embed.device)
        count = 0
        for module in self.modules():
            if isinstance(module, FastKANLayer):
                reg = reg + module.l1_spline_regularization()
                count += 1
        return reg / max(1, count)

    def forward(
        self,
        board_tensor: torch.Tensor,
        legal_mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Canonical forward pass accepting (B, 20, 9, 9) board tensor.
        Returns dictionary with `policy_logits` (B, 144), `win_prob` (B, 1), `value` (B, 1),
        and KAN-specific outputs (`cascade_synergy_map`, `spline_reg_loss`, `latent_z`).
        """
        B, C, H, W = board_tensor.shape
        z = self.encode_tokens(board_tensor)  # (B, 81, d_model)
        z_2d = z.transpose(1, 2).contiguous().view(B, self.d_model, H, W)

        # Policy logits over 144 canonical swaps
        p_feat = self.policy_spatial_reduce(z_2d).flatten(1)  # (B, 32 * 81)
        policy_logits = self.policy_kan_head(p_feat)  # (B, 144)

        if legal_mask is not None:
            policy_logits = policy_logits.masked_fill(~legal_mask.bool(), -1e9)

        # Global pooled KAN representation -> calibrated win probability & value
        z_global = z.mean(dim=1)  # (B, d_model)
        value_raw = self.value_kan_head(z_global)  # (B, 1)
        win_prob = torch.sigmoid(value_raw)
        value = torch.tanh(value_raw)

        # Per-cell cascade synergy heatmap (B, 1, 9, 9)
        cascade_synergy_map = self.synergy_kan_head(z).transpose(1, 2).view(B, 1, H, W)

        return {
            "policy_logits": policy_logits,
            "win_prob": win_prob,
            "value": value,
            "cascade_synergy_map": cascade_synergy_map,
            "spline_reg_loss": self.spline_regularization_loss(),
            "latent_z": z
        }

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def create_kan_candy_net(device: str = "cpu") -> KolmogorovArnoldCandyNet:
    """Factory function initializing KolmogorovArnoldCandyNet on the specified device."""
    model = KolmogorovArnoldCandyNet(
        in_channels=20,
        d_model=192,
        num_kan_blocks=5,
        grid_size=8,
        num_actions=144
    )
    return model.to(device)


if __name__ == "__main__":
    torch.manual_seed(42)
    net = create_kan_candy_net("cpu")
    params = net.count_parameters()
    print(f"[KolmogorovArnoldCandyNet] Initialized with {params:,} trainable parameters ({params / 1e6:.2f}M)")

    dummy_board = torch.randn(2, 20, 9, 9)
    out = net(dummy_board)
    print(f"  policy_logits:       {out['policy_logits'].shape}")
    print(f"  win_prob:            {out['win_prob'].shape}")
    print(f"  value:               {out['value'].shape}")
    print(f"  cascade_synergy_map: {out['cascade_synergy_map'].shape}")
    print(f"  spline_reg_loss:     {out['spline_reg_loss'].item():.6f}")
