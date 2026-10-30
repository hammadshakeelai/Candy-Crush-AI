"""
Level-Conditioned Dynamic HyperNetwork (`LevelConditionedHyperNet`) for Candy Crush Saga.
Based on:
  - Ha, Dai & Le (ICLR 2017), "HyperNetworks" (arXiv:1609.09106)
  - Chen et al. (CVPR 2020), "Dynamic Convolution: Attention Over Convolution Kernels"

Extracts a structural level signature (jelly density, double-jelly ratio, chocolate presence,
licorice cages, frosting thickness, portal topology, color diversity, moves remaining) and
dynamically synthesizes the convolutional filter weights, biases, and policy projection matrices
of the primary network on the fly! Every level geometry receives custom-tailored neural weights.
"""

import math
from typing import Dict, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class LevelStructuralSignatureExtractor(nn.Module):
    """
    Extracts an explicit 16-dimensional structural signature + learned topological embedding
    z_level in R^{B x d_cond} from the canonical 20-channel 9x9 board tensor:
      - Channels 0..6:   Candy color occupancy & entropy
      - Channels 7..11:  Special candy inventory (Striped H/V, Wrapped, Color Bomb)
      - Channels 12..14: Single vs. Double Jelly density & corner jelly exposure
      - Channel 15:      Expanding Chocolate hazard count & perimeter
      - Channel 16:      Licorice Lock density
      - Channel 17:      Multi-layer Frosting thickness
      - Channel 18:      Normalized Moves Remaining
      - Channel 19:      Portal / Teleporter entrance-exit presence
    """
    def __init__(self, in_channels: int = 20, d_cond: int = 128):
        super().__init__()
        self.in_channels = in_channels
        self.d_cond = d_cond

        # Spatial topology encoder
        self.spatial_encoder = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.AdaptiveAvgPool2d((3, 3)),
            nn.Flatten()
        )

        # Combine 16 explicit structural descriptors + 64*9 spatial pooled features
        self.signature_proj = nn.Sequential(
            nn.Linear(16 + 64 * 9, d_cond * 2),
            nn.LayerNorm(d_cond * 2),
            nn.GELU(),
            nn.Linear(d_cond * 2, d_cond),
            nn.LayerNorm(d_cond)
        )

    def extract_explicit_descriptors(self, board_tensor: torch.Tensor) -> torch.Tensor:
        """Computes 16 domain-grounded structural level metrics of shape (B, 16)."""
        B, C, H, W = board_tensor.shape
        num_cells = float(H * W)

        # Color distribution statistics (channels 0..6)
        color_counts = board_tensor[:, 0:7].sum(dim=(2, 3)) / num_cells  # (B, 7)
        color_probs = color_counts / (color_counts.sum(dim=-1, keepdim=True) + 1e-6)
        color_entropy = -(color_probs * torch.log(color_probs + 1e-6)).sum(dim=-1, keepdim=True)

        # Special candy counts (channels 7..11)
        striped_density = board_tensor[:, 8:10].sum(dim=(1, 2, 3), keepdim=False).unsqueeze(-1) / num_cells
        wrapped_density = board_tensor[:, 10].sum(dim=(1, 2)).unsqueeze(-1) / num_cells
        color_bomb_density = board_tensor[:, 11].sum(dim=(1, 2)).unsqueeze(-1) / num_cells

        # Jelly counts: single jelly (ch 13), double jelly (ch 14), corner jellies
        single_jelly = board_tensor[:, 13].sum(dim=(1, 2)).unsqueeze(-1) / num_cells
        double_jelly = board_tensor[:, 14].sum(dim=(1, 2)).unsqueeze(-1) / num_cells
        total_jelly = single_jelly + double_jelly

        corners = (
            board_tensor[:, 13:15, 0, 0] +
            board_tensor[:, 13:15, 0, -1] +
            board_tensor[:, 13:15, -1, 0] +
            board_tensor[:, 13:15, -1, -1]
        ).sum(dim=-1, keepdim=True) / 8.0

        # Blockers: Chocolate (ch 15), Licorice (ch 16), Frosting (ch 17)
        chocolate_density = board_tensor[:, 15].sum(dim=(1, 2)).unsqueeze(-1) / num_cells
        licorice_density = board_tensor[:, 16].sum(dim=(1, 2)).unsqueeze(-1) / num_cells
        frosting_density = board_tensor[:, 17].sum(dim=(1, 2)).unsqueeze(-1) / num_cells
        total_blockers = chocolate_density + licorice_density + frosting_density

        # Moves remaining (ch 18) & Portal presence (ch 19)
        moves_norm = board_tensor[:, 18].mean(dim=(1, 2)).unsqueeze(-1)
        portal_density = board_tensor[:, 19].abs().sum(dim=(1, 2)).unsqueeze(-1) / num_cells

        # Top-half vs bottom-half objective asymmetry
        top_jelly = board_tensor[:, 13:15, :4, :].sum(dim=(1, 2, 3)).unsqueeze(-1) / num_cells
        bot_jelly = board_tensor[:, 13:15, 5:, :].sum(dim=(1, 2, 3)).unsqueeze(-1) / num_cells

        descriptors = torch.cat([
            color_entropy,
            striped_density,
            wrapped_density,
            color_bomb_density,
            single_jelly,
            double_jelly,
            total_jelly,
            corners,
            chocolate_density,
            licorice_density,
            frosting_density,
            total_blockers,
            moves_norm,
            portal_density,
            top_jelly,
            bot_jelly
        ], dim=-1)  # (B, 16)
        return descriptors

    def forward(self, board_tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            z_cond: Conditioning hyper-embedding of shape (B, d_cond)
            descriptors: Explicit 16D structural signature of shape (B, 16)
        """
        descriptors = self.extract_explicit_descriptors(board_tensor)
        spatial_feat = self.spatial_encoder(board_tensor)
        z_cond = self.signature_proj(torch.cat([descriptors, spatial_feat], dim=-1))
        return z_cond, descriptors


class HyperGeneratedConvBlock(nn.Module):
    """
    Residual Block whose 3x3 depthwise + 1x1 pointwise convolutional filter weights and biases
    are dynamically synthesized per sample in the batch by a HyperNetwork generator!
    Uses low-rank tensor factorization + per-sample grouped 2D convolution (`groups=B * C`)
    for exact instance-specific weight application.
    """
    def __init__(
        self,
        channels: int = 256,
        d_cond: int = 128,
        num_basis_kernels: int = 8
    ):
        super().__init__()
        self.channels = channels
        self.d_cond = d_cond
        self.num_basis_kernels = num_basis_kernels

        # HyperNetwork generates sample-specific 3x3 depthwise filter weights (B, C, 1, 3, 3)
        self.dw_hyper_generator = nn.Sequential(
            nn.Linear(d_cond, 128),
            nn.GELU(),
            nn.Linear(128, channels * 9)
        )

        # HyperNetwork generates mixing coefficients over K basis 1x1xCxC pointwise kernels
        # W_pw(z) = sum_{k=1}^K alpha_k(z) * W_basis_k
        self.pw_basis_weights = nn.Parameter(
            torch.randn(num_basis_kernels, channels, channels) / math.sqrt(channels)
        )
        self.pw_hyper_coeff = nn.Sequential(
            nn.Linear(d_cond, 64),
            nn.GELU(),
            nn.Linear(64, num_basis_kernels)
        )

        # HyperNetwork generates sample-specific affine channel scale (gamma) and bias (beta)
        self.affine_hyper = nn.Sequential(
            nn.Linear(d_cond, channels * 2)
        )

        self.norm1 = nn.GroupNorm(16, channels)
        self.norm2 = nn.GroupNorm(16, channels)

    def forward(self, x: torch.Tensor, z_cond: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Spatial feature map (B, C, H, W)
            z_cond: Level conditioning vector (B, d_cond)
        Returns:
            out: Updated spatial feature map (B, C, H, W)
            weight_norm: Frobenius norm of dynamically generated filters (B, 1)
        """
        B, C, H, W = x.shape
        residual = x

        # 1. Generate per-sample 3x3 depthwise convolution filters: (B * C, 1, 3, 3)
        dw_weights = self.dw_hyper_generator(z_cond).view(B * C, 1, 3, 3)
        dw_weights = torch.tanh(dw_weights) * (1.0 / math.sqrt(9.0))

        # Apply instance-specific 3x3 depthwise filters via grouped conv2d
        x_norm = self.norm1(x)
        x_grouped = x_norm.view(1, B * C, H, W)
        out_dw = F.conv2d(x_grouped, dw_weights, padding=1, groups=B * C).view(B, C, H, W)
        out_dw = F.gelu(out_dw)

        # 2. Generate sample-specific 1x1 pointwise filter matrix W_pw(z) in R^{B x C x C}
        basis_alphas = F.softmax(self.pw_hyper_coeff(z_cond), dim=-1)  # (B, K)
        # Contract (B, K) with (K, C_out, C_in) -> (B, C_out, C_in)
        pw_weights = torch.einsum("bk,koc->boc", basis_alphas, self.pw_basis_weights)

        # Apply sample-specific 1x1 convolution across spatial tokens (B, C, H*W)
        out_flat = out_dw.view(B, C, H * W)
        out_pw = torch.bmm(pw_weights, out_flat).view(B, C, H, W)
        out_pw = self.norm2(out_pw)

        # 3. Apply hyper-generated affine modulation (FiLM scale + bias)
        gamma_beta = self.affine_hyper(z_cond).view(B, 2, C, 1, 1)
        gamma = 1.0 + 0.2 * torch.tanh(gamma_beta[:, 0])
        beta = 0.2 * gamma_beta[:, 1]
        out_modulated = F.gelu(gamma * out_pw + beta)

        # Compute Frobenius norm of generated weights for diagnostic telemetry
        dw_norm = dw_weights.view(B, -1).norm(dim=-1, keepdim=True)
        pw_norm = pw_weights.view(B, -1).norm(dim=-1, keepdim=True)
        total_weight_norm = dw_norm + pw_norm

        return residual + out_modulated, total_weight_norm


class LevelConditionedHyperNet(nn.Module):
    """
    Complete Level-Conditioned HyperNetwork Architecture for Candy Crush Saga.
    Extracts the level's structural regime signature and dynamically generates custom
    convolutional kernels and policy head weights on the fly for every board state.
    """
    def __init__(
        self,
        in_channels: int = 20,
        channels: int = 256,
        d_cond: int = 128,
        num_hyper_blocks: int = 5,
        num_basis_kernels: int = 8,
        num_actions: int = 144,
        board_h: int = 9,
        board_w: int = 9
    ):
        super().__init__()
        self.in_channels = in_channels
        self.channels = channels
        self.d_cond = d_cond
        self.num_hyper_blocks = num_hyper_blocks
        self.num_actions = num_actions
        self.board_h = board_h
        self.board_w = board_w

        # Context / Level Signature Extractor
        self.signature_extractor = LevelStructuralSignatureExtractor(
            in_channels=in_channels,
            d_cond=d_cond
        )

        # Primary network stem
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(16, channels),
            nn.GELU()
        )

        # Stack of Hyper-Generated Convolutional Blocks
        self.hyper_blocks = nn.ModuleList([
            HyperGeneratedConvBlock(
                channels=channels,
                d_cond=d_cond,
                num_basis_kernels=num_basis_kernels
            )
            for _ in range(num_hyper_blocks)
        ])

        # Dynamic Hyper-Generated Policy Head
        self.policy_spatial_conv = nn.Sequential(
            nn.Conv2d(channels, 64, kernel_size=1, bias=False),
            nn.GroupNorm(8, 64),
            nn.GELU()
        )
        self.policy_base_fc = nn.Linear(64 * board_h * board_w, 256)

        # HyperNetwork generates sample-specific final policy projection weights W_pol(z) in R^{B x 144 x 256}
        # via rank-16 factorized hyper-adaptation: W(z) = W_shared + U(z) @ V(z)^T
        self.policy_shared_out = nn.Linear(256, num_actions)
        self.hyper_policy_u = nn.Linear(d_cond, num_actions * 16)
        self.hyper_policy_v = nn.Linear(d_cond, 256 * 16)
        self.hyper_policy_bias = nn.Linear(d_cond, num_actions)

        # Dynamic Hyper-Conditioned Value / Win Probability Head
        self.value_head = nn.Sequential(
            nn.Linear(channels + d_cond, 128),
            nn.GELU(),
            nn.Linear(128, 1)
        )

    def forward(
        self,
        board_tensor: torch.Tensor,
        legal_mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Canonical forward pass accepting (B, 20, 9, 9) board tensor.
        Dynamically synthesizes level-conditioned convolutional and policy weights on the fly.
        """
        B, C, H, W = board_tensor.shape

        # 1. Extract structural level signature and hyper-conditioning vector z_cond
        z_cond, level_signature = self.signature_extractor(board_tensor)  # (B, d_cond), (B, 16)

        # 2. Forward through hyper-generated convolutional backbone
        h = self.stem(board_tensor)
        filter_norms = []
        for block in self.hyper_blocks:
            h, w_norm = block(h, z_cond)
            filter_norms.append(w_norm)

        generated_filter_norms = torch.cat(filter_norms, dim=-1)  # (B, num_hyper_blocks)

        # 3. Hyper-generated dynamic policy projection
        p_feat = F.gelu(self.policy_base_fc(self.policy_spatial_conv(h).flatten(1)))  # (B, 256)
        shared_logits = self.policy_shared_out(p_feat)  # (B, 144)

        # Low-rank dynamic weight perturbation: delta_W = U(z) @ V(z)^T / sqrt(16)
        u_mat = self.hyper_policy_u(z_cond).view(B, self.num_actions, 16)
        v_mat = self.hyper_policy_v(z_cond).view(B, 256, 16)
        # Compute U @ (V^T @ p_feat) in O(B * (256 + 144) * 16)
        v_proj = torch.bmm(v_mat.transpose(1, 2), p_feat.unsqueeze(-1))  # (B, 16, 1)
        dynamic_logits = torch.bmm(u_mat, v_proj).squeeze(-1) * (0.25 / math.sqrt(16.0))
        dynamic_bias = 0.1 * self.hyper_policy_bias(z_cond)

        policy_logits = shared_logits + dynamic_logits + dynamic_bias
        if legal_mask is not None:
            policy_logits = policy_logits.masked_fill(~legal_mask.bool(), -1e9)

        # 4. Level-conditioned Value & Win Probability
        h_global = h.mean(dim=(2, 3))  # (B, channels)
        val_raw = self.value_head(torch.cat([h_global, z_cond], dim=-1))  # (B, 1)
        win_prob = torch.sigmoid(val_raw)
        value = torch.tanh(val_raw)

        return {
            "policy_logits": policy_logits,
            "win_prob": win_prob,
            "value": value,
            "level_signature": level_signature,
            "regime_embedding": z_cond,
            "generated_filter_norms": generated_filter_norms
        }

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def create_hypernetwork_candy_net(device: str = "cpu") -> LevelConditionedHyperNet:
    """Factory function initializing LevelConditionedHyperNet on the target device."""
    model = LevelConditionedHyperNet(
        in_channels=20,
        channels=256,
        d_cond=128,
        num_hyper_blocks=5,
        num_basis_kernels=8,
        num_actions=144
    )
    return model.to(device)


if __name__ == "__main__":
    torch.manual_seed(42)
    net = create_hypernetwork_candy_net("cpu")
    params = net.count_parameters()
    print(f"[LevelConditionedHyperNet] Initialized with {params:,} trainable parameters ({params / 1e6:.2f}M)")

    dummy_board = torch.randn(2, 20, 9, 9).abs()
    out = net(dummy_board)
    print(f"  policy_logits:          {out['policy_logits'].shape}")
    print(f"  win_prob:               {out['win_prob'].shape}")
    print(f"  value:                  {out['value'].shape}")
    print(f"  level_signature:        {out['level_signature'].shape}")
    print(f"  regime_embedding:       {out['regime_embedding'].shape}")
    print(f"  generated_filter_norms: {out['generated_filter_norms'].shape}")
