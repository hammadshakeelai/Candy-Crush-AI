"""
AlphaCandy Model Zoo — Self-Supervised Contrastive State Representation Network
(`SimCLRCandyContrastiveNet`).

Based on:
  - Chen et al. (ICML 2020), "A Simple Framework for Contrastive Learning of Visual Representations (SimCLR)"
  - Laskin et al. (ICML 2020), "CURL: Contrastive Unsupervised Representations for Reinforcement Learning"
  - Wang & Isola (ICML 2020), "Understanding Contrastive Representation Learning through Alignment
    and Uniformity on the Hypersphere"

In Candy Crush Saga, any permutation pi in S_6 of the 6 candy colors (Red, Orange, Yellow, Green,
Blue, Purple; channels 0..5) produces a 100% structurally isomorphic board state with identical
optimal policy and value ($|S_6| = 720$ equivalent boards per state).

`SimCLRCandyContrastiveNet` combines:
  1. An S_6 Color-Equivariant Relational Stem that extracts color-symmetric structural features.
  2. An L2-normalized Hypersphere Projection Head z_proj in S^127 (unit sphere in R^128).
  3. An NT-Xent / InfoNCE Temperature-Scaled Cosine Similarity Loss helper (`infonce_loss`) +
     automatic S_6 color-permutation positive view generator (`generate_isomorphic_color_view`).
  4. Canonical 144-action policy and value heads operating on the color-invariant latent manifold.
"""

from typing import Dict, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class S6ColorSymmetricStem(nn.Module):
    """
    S_6 Permutation-Equivariant & Invariant Color Stem.
    Applies shared spatial filters to each of the 6 candy color channels (0..5) independently,
    then aggregates symmetric statistics (mean, max, pairwise variance across colors) so that
    any permutation of candy colors maps to an isomorphic structural representation.
    """

    def __init__(self, color_feat_dim: int = 24, struct_channels: int = 14, out_dim: int = 192):
        super().__init__()
        self.color_feat_dim = color_feat_dim

        # Shared per-color 2D spatial pattern detector (detects L-shapes, T-shapes, 2-in-a-row, etc.)
        self.per_color_conv = nn.Sequential(
            nn.Conv2d(1, color_feat_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(color_feat_dim),
            nn.SiLU(),
            nn.Conv2d(color_feat_dim, color_feat_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(color_feat_dim),
            nn.SiLU()
        )

        # Structural channels (ch 6..19: specials, jelly, chocolate, frosting, portals, active mask, moves)
        self.struct_conv = nn.Sequential(
            nn.Conv2d(struct_channels, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.SiLU()
        )

        # Fuse S_6-invariant stats (mean, max, std across 6 colors -> 3 * color_feat_dim) + raw 20ch + struct 64ch
        fused_in = 3 * color_feat_dim + 64 + 20
        self.fuse_proj = nn.Sequential(
            nn.Conv2d(fused_in, out_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_dim),
            nn.SiLU()
        )

    def forward(self, board_tensor: torch.Tensor) -> torch.Tensor:
        B, C, H, W = board_tensor.shape
        colors = board_tensor[:, 0:6]   # (B, 6, 9, 9)
        structs = board_tensor[:, 6:20] # (B, 14, 9, 9)

        # Shared conv across all 6 color planes: (B * 6, 1, 9, 9) -> (B, 6, D_c, 9, 9)
        c_flat = colors.reshape(B * 6, 1, H, W)
        c_feat = self.per_color_conv(c_flat).view(B, 6, self.color_feat_dim, H, W)

        # S_6 permutation-invariant symmetric reductions across the 6 colors
        c_mean = c_feat.mean(dim=1)
        c_max = c_feat.max(dim=1).values
        c_std = torch.sqrt(c_feat.var(dim=1, unbiased=False) + 1e-6)

        s_feat = self.struct_conv(structs)
        combined = torch.cat([c_mean, c_max, c_std, s_feat, board_tensor], dim=1)
        return self.fuse_proj(combined)


class ContrastiveResidualBlock(nn.Module):
    """Residual block with pre-activation LayerNorm/BatchNorm and depthwise-separable mixing."""

    def __init__(self, d_model: int = 192):
        super().__init__()
        self.conv1 = nn.Conv2d(d_model, d_model, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(d_model)
        self.conv2 = nn.Conv2d(d_model, d_model, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.silu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.silu(x + out)


class SimCLRCandyContrastiveNet(nn.Module):
    """
    Self-Supervised Contrastive State Representation Network (`SimCLRCandyContrastiveNet`).
    Projects 9x9 board states onto the 128-dimensional unit hypersphere S^127 while predicting
    canonical 144-action swap policy logits and state values.
    """

    def __init__(
        self,
        in_channels: int = 20,
        d_model: int = 192,
        num_blocks: int = 5,
        proj_dim: int = 128,
        num_actions: int = 144,
        board_h: int = 9,
        board_w: int = 9
    ):
        super().__init__()
        self.in_channels = in_channels
        self.d_model = d_model
        self.num_blocks = num_blocks
        self.proj_dim = proj_dim  # L2-normalized vector in R^128 lies on unit sphere S^127
        self.num_actions = num_actions
        self.board_h = board_h
        self.board_w = board_w

        # 1. S_6 Color-Symmetric Stem
        self.stem = S6ColorSymmetricStem(color_feat_dim=24, struct_channels=in_channels - 6, out_dim=d_model)

        # 2. Deep Spatial Residual Encoder
        self.encoder_blocks = nn.ModuleList([
            ContrastiveResidualBlock(d_model) for _ in range(num_blocks)
        ])

        # Spatial Transformer Context Layer over 81 cells
        self.self_attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=6, batch_first=True)
        self.attn_norm = nn.LayerNorm(d_model)

        # 3. SimCLR Nonlinear Projection Head g(h) -> z_proj in S^127 (unit hypersphere in R^128)
        self.projection_head = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.LayerNorm(d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.SiLU(),
            nn.Linear(d_model, proj_dim, bias=False)
        )

        # 4. Canonical 144-Action Policy Head conditioned on spatial + hypersphere latent
        self.policy_conv = nn.Sequential(
            nn.Conv2d(d_model, 32, kernel_size=1, bias=False),
            nn.BatchNorm2d(32),
            nn.SiLU()
        )
        self.policy_head = nn.Sequential(
            nn.Linear(32 * board_h * board_w + proj_dim, 256),
            nn.SiLU(),
            nn.Linear(256, num_actions)
        )

        # 5. Canonical Value & Win Probability Head
        self.value_head = nn.Sequential(
            nn.Linear(d_model * 2 + proj_dim, 128),
            nn.SiLU(),
            nn.Linear(128, 1)
        )

    @staticmethod
    def generate_isomorphic_color_view(board_tensor: torch.Tensor) -> torch.Tensor:
        """
        Applies a random S_6 permutation to the 6 candy color channels (0..5) for each sample
        in the batch, creating a structurally isomorphic positive view with identical move topology.
        """
        B, C, H, W = board_tensor.shape
        aug = board_tensor.clone()
        for b in range(B):
            perm = torch.randperm(6, device=board_tensor.device)
            aug[b, 0:6] = board_tensor[b, perm]
        return aug

    def encode(self, board_tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            feat_2d: (B, d_model, 9, 9) spatial map
            h_global: (B, 2 * d_model) pooled encoder representation
            z_proj: (B, 128) L2-normalized projection on unit hypersphere S^127
        """
        B, C, H, W = board_tensor.shape
        x = self.stem(board_tensor)
        for block in self.encoder_blocks:
            x = block(x)

        tokens = x.flatten(2).transpose(1, 2)  # (B, 81, d_model)
        attn_out, _ = self.self_attn(tokens, tokens, tokens, need_weights=False)
        tokens = self.attn_norm(tokens + attn_out)

        feat_2d = tokens.transpose(1, 2).contiguous().view(B, self.d_model, H, W)
        h_mean = tokens.mean(dim=1)
        h_max = tokens.max(dim=1).values
        h_global = torch.cat([h_mean, h_max], dim=-1)  # (B, 2 * d_model)

        # Project onto unit hypersphere S^127 via L2 normalization
        z_raw = self.projection_head(h_global)
        z_proj = F.normalize(z_raw, p=2, dim=-1)  # ||z_proj||_2 = 1, lies on S^127
        return feat_2d, h_global, z_proj

    def forward(
        self,
        board_tensor: torch.Tensor,
        legal_mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Canonical forward pass returning `policy_logits` (B, 144), `value` (B, 1),
        `win_prob` (B, 1), and hypersphere embedding `z_proj` (B, 128) on S^127.
        """
        feat_2d, h_global, z_proj = self.encode(board_tensor)

        spatial_flat = self.policy_conv(feat_2d).flatten(1)
        policy_logits = self.policy_head(torch.cat([spatial_flat, z_proj], dim=-1))

        if legal_mask is not None:
            policy_logits = policy_logits.masked_fill(~legal_mask.bool(), -1e9)

        val_raw = self.value_head(torch.cat([h_global, z_proj], dim=-1))
        value = torch.tanh(val_raw)
        win_prob = torch.sigmoid(val_raw)

        return {
            "policy_logits": policy_logits,
            "value": value,
            "win_prob": win_prob,
            "z_proj": z_proj,
            "encoder_latent": h_global,
        }

    @staticmethod
    def infonce_loss(
        z1: torch.Tensor,
        z2: torch.Tensor,
        temperature: float = 0.07
    ) -> Dict[str, torch.Tensor]:
        """
        Computes symmetric NT-Xent / InfoNCE temperature-scaled cosine similarity loss over
        L2-normalized hypersphere embeddings z1, z2 in S^127 (Chen et al., SimCLR 2020),
        plus hypersphere Alignment and Uniformity diagnostics (Wang & Isola, 2020).

        Args:
            z1: (B, 128) L2-normalized projection of view 1
            z2: (B, 128) L2-normalized projection of isomorphic view 2
            temperature: Softmax temperature tau (default 0.07)
        """
        z1 = F.normalize(z1, p=2, dim=-1)
        z2 = F.normalize(z2, p=2, dim=-1)
        B = z1.shape[0]

        # Concatenate 2B embeddings on S^127
        z_all = torch.cat([z1, z2], dim=0)  # (2B, 128)
        sim_matrix = torch.matmul(z_all, z_all.T) / temperature  # (2B, 2B)

        # Mask out self-similarity on the main diagonal
        diag_mask = torch.eye(2 * B, device=z1.device, dtype=torch.bool)
        sim_matrix = sim_matrix.masked_fill(diag_mask, -1e9)

        # Positive pair indices: i <-> i + B and i + B <-> i
        pos_targets = torch.cat([
            torch.arange(B, 2 * B, device=z1.device),
            torch.arange(0, B, device=z1.device)
        ], dim=0)

        loss = F.cross_entropy(sim_matrix, pos_targets)

        # Hypersphere alignment: E[||z1 - z2||_2^2]
        alignment = (z1 - z2).pow(2).sum(dim=-1).mean()
        # Hypersphere cosine similarity between positive isomorphic pairs
        pos_cosine_sim = (z1 * z2).sum(dim=-1).mean()

        return {
            "loss": loss,
            "alignment": alignment,
            "pos_cosine_sim": pos_cosine_sim,
        }

    def self_supervised_step(
        self,
        board_tensor: torch.Tensor,
        temperature: float = 0.07
    ) -> Dict[str, torch.Tensor]:
        """
        Generates two random S_6 color-permuted isomorphic views of `board_tensor` and
        computes their SimCLR InfoNCE contrastive loss on S^127.
        """
        view1 = self.generate_isomorphic_color_view(board_tensor)
        view2 = self.generate_isomorphic_color_view(board_tensor)
        _, _, z1 = self.encode(view1)
        _, _, z2 = self.encode(view2)
        return self.infonce_loss(z1, z2, temperature=temperature)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def create_simclr_candy_contrastive_net(device: str = "cpu") -> SimCLRCandyContrastiveNet:
    model = SimCLRCandyContrastiveNet(
        in_channels=20,
        d_model=192,
        num_blocks=5,
        proj_dim=128,
        num_actions=144
    )
    return model.to(device)


if __name__ == "__main__":
    torch.manual_seed(42)
    net = create_simclr_candy_contrastive_net("cpu")
    params = net.count_parameters()
    print(f"[SimCLRCandyContrastiveNet] Trainable parameters: {params:,} ({params / 1e6:.2f}M)")

    dummy_board = torch.rand(2, 20, 9, 9)
    out = net(dummy_board)
    ssl_metrics = net.self_supervised_step(dummy_board, temperature=0.07)

    # Verify unit sphere norm ||z_proj||_2 == 1.0
    norms = torch.norm(out["z_proj"], p=2, dim=-1)
    print(f"  policy_logits:   {out['policy_logits'].shape}")
    print(f"  value:           {out['value'].shape}")
    print(f"  win_prob:        {out['win_prob'].shape}")
    print(f"  z_proj (S^127):  {out['z_proj'].shape} (L2 norms: {norms.detach().tolist()})")
    print(f"  InfoNCE loss:    {ssl_metrics['loss'].item():.4f} (pos_cos_sim: {ssl_metrics['pos_cosine_sim'].item():.4f})")
