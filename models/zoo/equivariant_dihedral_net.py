"""
Gravity-Preserving Dihedral (Z_2) & Color-Permutation (S_6) Equivariant Network for Candy Crush Saga.
Exploits the two exact mathematical symmetry groups of Match-3 physics under downward gravity:
  1. Horizontal Reflection Equivariance (Z_2 Group = {I, sigma_h}):
     - Flipping the board left-to-right (r, c) <-> (r, 8 - c) commutes with downward gravity.
     - Horizontal swaps (r, c, RIGHT) equivariantly map to (r, 7 - c, RIGHT).
     - Vertical swaps (r, c, DOWN) equivariantly map to (r, 8 - c, DOWN).
  2. Color-Channel Permutation Invariance (Symmetric Group S_6, |S_6| = 720):
     - Channels 0..5 encode the 6 regular candy colors.
     - Uses weight-shared per-color equivariant branches with symmetric (mean + max) cross-color
       pooling so any permutation of the 6 candy colors produces identical policy and value outputs.
"""

from typing import Dict, Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


def build_horizontal_flip_action_permutation(board_h: int = 9, board_w: int = 9) -> torch.Tensor:
    """
    Constructs the exact 144-element index permutation tensor `perm` under a left-right
    board reflection (r, c) <-> (r, board_w - 1 - c):
      - Horizontal swaps [0..71]: idx = r * (W - 1) + c  <->  r * (W - 1) + (W - 2 - c)
      - Vertical swaps  [72..143]: idx = 72 + r * W + c  <->  72 + r * W + (W - 1 - c)
    """
    perm = torch.zeros(144, dtype=torch.long)
    # 1. Horizontal swaps (9 rows x 8 horizontal edges)
    for r in range(board_h):
        for c in range(board_w - 1):
            orig_idx = r * (board_w - 1) + c
            flipped_c = (board_w - 2) - c
            flipped_idx = r * (board_w - 1) + flipped_c
            perm[orig_idx] = flipped_idx

    # 2. Vertical swaps (8 vertical edges x 9 columns)
    for r in range(board_h - 1):
        for c in range(board_w):
            orig_idx = 72 + r * board_w + c
            flipped_c = (board_w - 1) - c
            flipped_idx = 72 + r * board_w + flipped_c
            perm[orig_idx] = flipped_idx

    return perm


class Z2SymmetricConv2d(nn.Module):
    """
    2D Convolution layer strictly equivariant to horizontal reflection (Z_2 left-right flip).
    Enforces W_sym(:, :, :, k) = 0.5 * (W(:, :, :, k) + W(:, :, :, K - 1 - k)) during forward pass.
    Vertical weights remain unconstrained to capture asymmetric downward gravity physics.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, padding: int = 1, bias: bool = True):
        super().__init__()
        self.padding = padding
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels, kernel_size, kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels))
        else:
            self.register_parameter("bias", None)
        nn.init.kaiming_normal_(self.weight, nonlinearity="relu")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w_sym = 0.5 * (self.weight + torch.flip(self.weight, dims=[-1]))
        return F.conv2d(x, w_sym, self.bias, padding=self.padding)


class S6ColorEquivariantBlock(nn.Module):
    """
    S_6 x Z_2 Equivariant Interaction Block over the 6 Candy Color Channels.
    Processes each color c in {0..5} with shared Z_2-symmetric weights and exchanges
    information across colors via permutation-invariant mean and max aggregations.
    """
    def __init__(self, d_color: int = 64):
        super().__init__()
        self.conv_self = Z2SymmetricConv2d(d_color, d_color, kernel_size=3, padding=1, bias=False)
        self.conv_mean = Z2SymmetricConv2d(d_color, d_color, kernel_size=3, padding=1, bias=False)
        self.conv_max = Z2SymmetricConv2d(d_color, d_color, kernel_size=3, padding=1, bias=False)
        self.norm = nn.GroupNorm(8, d_color)
        self.conv_out = Z2SymmetricConv2d(d_color, d_color, kernel_size=3, padding=1, bias=True)

    def forward(self, c_feat: torch.Tensor) -> torch.Tensor:
        """
        Args:
            c_feat: Tensor of shape (B, 6, d_color, H, W)
        Returns:
            Updated S_6 x Z_2 equivariant tensor of shape (B, 6, d_color, H, W)
        """
        B, num_colors, C, H, W = c_feat.shape
        flat = c_feat.view(B * num_colors, C, H, W)

        # Permutation-invariant cross-color statistics
        mean_pool = c_feat.mean(dim=1, keepdim=True).expand(-1, num_colors, -1, -1, -1).reshape(B * num_colors, C, H, W)
        max_pool = c_feat.max(dim=1, keepdim=True)[0].expand(-1, num_colors, -1, -1, -1).reshape(B * num_colors, C, H, W)

        h = self.conv_self(flat) + self.conv_mean(mean_pool) + self.conv_max(max_pool)
        h = F.gelu(self.norm(h))
        h = self.conv_out(h)

        return c_feat + h.view(B, num_colors, C, H, W)


class Z2EquivariantResBlock(nn.Module):
    """Residual block preserving exact Z_2 horizontal reflection equivariance."""
    def __init__(self, channels: int = 256):
        super().__init__()
        self.conv1 = Z2SymmetricConv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.norm1 = nn.GroupNorm(16, channels)
        self.conv2 = Z2SymmetricConv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.norm2 = nn.GroupNorm(16, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.gelu(self.norm1(self.conv1(x)))
        h = self.norm2(self.conv2(h))
        return F.gelu(x + h)


class GravityEquivariantCandyNet(nn.Module):
    """
    Gravity-Preserving Z_2 x S_6 Equivariant Policy-Value Network.
    Guarantees:
      1. Color Permutation Invariance: Permuting any of the first 6 color channels
         leaves policy_logits and win_prob 100% unchanged.
      2. Horizontal Reflection Equivariance: Flipping the board horizontally
         flips policy_logits according to `build_horizontal_flip_action_permutation`
         and leaves win_prob 100% unchanged.
    """
    def __init__(
        self,
        in_channels: int = 20,
        d_color: int = 64,
        d_model: int = 256,
        num_s6_blocks: int = 3,
        num_z2_blocks: int = 6,
        board_h: int = 9,
        board_w: int = 9
    ):
        super().__init__()
        self.in_channels = in_channels
        self.d_color = d_color
        self.d_model = d_model
        self.board_h = board_h
        self.board_w = board_w

        # Register the exact 144-action horizontal reflection permutation buffer
        self.register_buffer("flip_action_perm", build_horizontal_flip_action_permutation(board_h, board_w))

        # Shared per-color stem: takes 1 color channel + 14 non-color structural channels = 15 channels
        struct_channels = in_channels - 6
        self.color_stem = nn.Sequential(
            Z2SymmetricConv2d(1 + struct_channels, d_color, kernel_size=3, padding=1),
            nn.GroupNorm(8, d_color),
            nn.GELU()
        )

        # S_6 x Z_2 Equivariant Color Interaction Tower
        self.s6_tower = nn.ModuleList([S6ColorEquivariantBlock(d_color) for _ in range(num_s6_blocks)])

        # Fuse S_6-invariant symmetric statistics (mean, max, second-moment) + structural stem into d_model
        self.struct_stem = nn.Sequential(
            Z2SymmetricConv2d(struct_channels, d_color, kernel_size=3, padding=1),
            nn.GroupNorm(8, d_color),
            nn.GELU()
        )
        self.fuse_proj = nn.Sequential(
            Z2SymmetricConv2d(d_color * 4, d_model, kernel_size=3, padding=1),
            nn.GroupNorm(16, d_model),
            nn.GELU()
        )

        # Z_2 Equivariant Backbone Tower (captures asymmetric vertical gravity + symmetric horizontal geometry)
        self.z2_tower = nn.ModuleList([Z2EquivariantResBlock(d_model) for _ in range(num_z2_blocks)])

        # Edge-Aligned Horizontal Swap Scorer: evaluates undirected horizontal edge {(r, c), (r, c+1)}
        # Using symmetric sum (f_L + f_R) and absolute difference |f_L - f_R| guarantees exact Z_2 equivariance!
        self.h_edge_scorer = nn.Sequential(
            Z2SymmetricConv2d(d_model * 2, 128, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(128, 1, kernel_size=1)
        )

        # Edge-Aligned Vertical Swap Scorer: evaluates directed vertical edge ((r, c) -> (r+1, c))
        self.v_edge_scorer = nn.Sequential(
            Z2SymmetricConv2d(d_model * 2, 128, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(128, 1, kernel_size=1)
        )

        # Z_2 & S_6 Invariant Win Probability Head
        self.win_head = nn.Sequential(
            nn.Linear(d_model * 2, 128),
            nn.GELU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def extract_equivariant_features(self, board_tensor: torch.Tensor) -> torch.Tensor:
        """
        Extracts a spatial feature map F in R^{B x d_model x 9 x 9} that is:
          - Strictly invariant to any S_6 permutation of channels 0..5
          - Strictly equivariant to Z_2 horizontal reflection (dim=-1 flip)
        """
        B, C, H, W = board_tensor.shape
        colors = board_tensor[:, :6, :, :]   # (B, 6, H, W)
        struct = board_tensor[:, 6:, :, :]   # (B, 14, H, W)

        # Concatenate each color i with the structural channels
        struct_expanded = struct.unsqueeze(1).expand(-1, 6, -1, -1, -1)
        per_color_in = torch.cat([colors.unsqueeze(2), struct_expanded], dim=2)  # (B, 6, 15, H, W)

        c_feat = self.color_stem(per_color_in.view(B * 6, -1, H, W)).view(B, 6, self.d_color, H, W)
        for block in self.s6_tower:
            c_feat = block(c_feat)

        # Symmetric S_6 pooling across the 6 color branches (mean, max, L2 deviation)
        c_mean = c_feat.mean(dim=1)
        c_max = c_feat.max(dim=1)[0]
        c_var = ((c_feat - c_mean.unsqueeze(1)) ** 2).mean(dim=1).sqrt()
        s_feat = self.struct_stem(struct)

        fused = self.fuse_proj(torch.cat([c_mean, c_max, c_var, s_feat], dim=1))
        for block in self.z2_tower:
            fused = block(fused)

        return fused

    def _raw_forward(self, board_tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, C, H, W = board_tensor.shape
        feat = self.extract_equivariant_features(board_tensor)  # (B, d_model, 9, 9)

        # 1. Horizontal edge features for swaps (r, c, RIGHT) where c in 0..7 -> shape (B, 2*d_model, 9, 8)
        f_left = feat[:, :, :, :-1]
        f_right = feat[:, :, :, 1:]
        h_edge_in = torch.cat([f_left + f_right, torch.abs(f_left - f_right)], dim=1)
        h_logits = self.h_edge_scorer(h_edge_in).view(B, H * (W - 1))  # (B, 72)

        # 2. Vertical edge features for swaps (r, c, DOWN) where r in 0..7 -> shape (B, 2*d_model, 8, 9)
        f_top = feat[:, :, :-1, :]
        f_bot = feat[:, :, 1:, :]
        v_edge_in = torch.cat([f_top, f_bot], dim=1)
        v_logits = self.v_edge_scorer(v_edge_in).view(B, (H - 1) * W)  # (B, 72)

        policy_logits = torch.cat([h_logits, v_logits], dim=1)  # (B, 144)

        # 3. Global Z_2 & S_6 invariant state pooling for win probability
        g_mean = feat.mean(dim=(2, 3))
        g_max = feat.amax(dim=(2, 3))
        win_prob = self.win_head(torch.cat([g_mean, g_max], dim=-1))
        return policy_logits, win_prob, feat

    def forward(self, board_tensor: torch.Tensor, exact_orbit_projection: bool = True) -> Dict[str, torch.Tensor]:
        """
        Args:
            board_tensor: Canonical state tensor of shape (B, 20, 9, 9)
            exact_orbit_projection: Eliminates IEEE-754 float32 summation order drift via Z_2 Reynolds operator
        Returns:
            Dictionary with:
              - `policy_logits`: (B, 144) Z_2-equivariant & S_6-invariant swap logits
              - `win_prob`: (B, 1) Z_2-invariant & S_6-invariant win probability
              - `value`: (B, 1) alias for win_prob
              - `spatial_features`: (B, d_model, 9, 9) equivariant feature map
        """
        policy_logits, win_prob, feat = self._raw_forward(board_tensor)

        if exact_orbit_projection:
            flipped_board = self.flip_board_horizontally(board_tensor)
            flipped_logits, flipped_win, flipped_feat = self._raw_forward(flipped_board)
            policy_logits = 0.5 * (policy_logits + self.flip_policy_horizontally(flipped_logits))
            win_prob = 0.5 * (win_prob + flipped_win)
            feat = 0.5 * (feat + torch.flip(flipped_feat, dims=[-1]))

        return {
            "policy_logits": policy_logits,
            "win_prob": win_prob,
            "value": win_prob,
            "spatial_features": feat
        }

    def flip_board_horizontally(self, board_tensor: torch.Tensor) -> torch.Tensor:
        """Flips a (B, 20, 9, 9) board tensor along the horizontal column axis (dim=-1)."""
        return torch.flip(board_tensor, dims=[-1])

    def flip_policy_horizontally(self, policy_logits: torch.Tensor) -> torch.Tensor:
        """Applies the canonical 144-action horizontal reflection permutation to policy logits."""
        return policy_logits[:, self.flip_action_perm]

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def create_gravity_equivariant_net(device: str = "cpu") -> GravityEquivariantCandyNet:
    model = GravityEquivariantCandyNet(
        in_channels=20,
        d_color=64,
        d_model=256,
        num_s6_blocks=3,
        num_z2_blocks=6
    )
    return model.to(device)


if __name__ == "__main__":
    torch.manual_seed(42)
    net = create_gravity_equivariant_net("cpu")
    net.eval()
    params = net.count_parameters()
    print(f"[GravityEquivariantCandyNet] Initialized with {params:,} trainable parameters ({params / 1e6:.2f}M)")

    dummy_board = torch.randn(2, 20, 9, 9)
    out_orig = net(dummy_board)
    print(f"  policy_logits shape: {out_orig['policy_logits'].shape}")
    print(f"  win_prob shape:      {out_orig['win_prob'].shape}")

    # Mathematical Verification 1: Z_2 Horizontal Reflection Equivariance
    flipped_board = net.flip_board_horizontally(dummy_board)
    out_flipped = net(flipped_board)
    expected_flipped_policy = net.flip_policy_horizontally(out_orig["policy_logits"])
    z2_policy_error = (out_flipped["policy_logits"] - expected_flipped_policy).abs().max().item()
    z2_value_error = (out_flipped["win_prob"] - out_orig["win_prob"]).abs().max().item()
    print(f"  [Z_2 Reflection Check] Max Policy Equivariance Error: {z2_policy_error:.2e} | Value Error: {z2_value_error:.2e}")
    assert z2_policy_error < 1e-5, f"Z_2 equivariance violated: {z2_policy_error}"
    assert z2_value_error < 1e-5, f"Z_2 value invariance violated: {z2_value_error}"

    # Mathematical Verification 2: S_6 Color-Channel Permutation Invariance
    color_perm = torch.tensor([3, 0, 5, 1, 4, 2], dtype=torch.long)
    permuted_board = dummy_board.clone()
    permuted_board[:, :6, :, :] = dummy_board[:, color_perm, :, :]
    out_color_perm = net(permuted_board)
    s6_policy_error = (out_color_perm["policy_logits"] - out_orig["policy_logits"]).abs().max().item()
    s6_value_error = (out_color_perm["win_prob"] - out_orig["win_prob"]).abs().max().item()
    print(f"  [S_6 Color Perm Check] Max Policy Invariance Error:   {s6_policy_error:.2e} | Value Error: {s6_value_error:.2e}")
    assert s6_policy_error < 1e-5, f"S_6 color invariance violated: {s6_policy_error}"
    print("  [PASS] Exact Z_2 Horizontal Equivariance & S_6 Color Permutation Symmetry Verified!")
