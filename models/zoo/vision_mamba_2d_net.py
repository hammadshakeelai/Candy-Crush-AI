"""
Vision Mamba 2D (CandyMamba2DNet): 4-Way Directional Selective State Space Model for Candy Crush Saga.
Tailored for Match-3 Physics Propagation:
  1. Top-to-Bottom Scan: Tracks gravity drop trajectories, falling candies, and ingredient chutes.
  2. Bottom-to-Top Scan: Models column support compression and upward cascade chain reactions.
  3. Left-to-Right Scan: Captures horizontal match-3/4/5 alignments and rightward striped laser blasts.
  4. Right-to-Left Scan: Captures leftward striped laser blasts and cross-column portal dependencies.
"""

from typing import Dict, Tuple, Optional, Union
import torch
import torch.nn as nn
import torch.nn.functional as F


class MambaOutput(tuple):
    """
    Hybrid output container that supports BOTH tuple unpacking:
        policy_logits, win_prob, cascade_potential_map = model(x)
    AND dictionary-style key lookup:
        out = model(x); out["policy_logits"], out["win_prob"], out["cascade_potential_map"]
    """
    _KEYS = ("policy_logits", "win_prob", "cascade_potential_map")

    def __new__(cls, policy_logits: torch.Tensor, win_prob: torch.Tensor, cascade_potential_map: torch.Tensor):
        return super().__new__(cls, (policy_logits, win_prob, cascade_potential_map))

    def __getitem__(self, key: Union[int, slice, str]):
        if isinstance(key, str):
            if key == "value":
                return super().__getitem__(1)
            if key in self._KEYS:
                return super().__getitem__(self._KEYS.index(key))
            raise KeyError(f"Unknown key '{key}'. Available keys: {self._KEYS}")
        return super().__getitem__(key)

    def keys(self):
        return self._KEYS

    def to_dict(self) -> Dict[str, torch.Tensor]:
        return {
            "policy_logits": self[0],
            "win_prob": self[1],
            "value": self[1],
            "cascade_potential_map": self[2]
        }


class SelectiveScan1D(nn.Module):
    """
    Input-Dependent Selective State Space Recurrence (S6):
        h_t = exp(Delta_t * A) * h_{t-1} + (Delta_t * B_t) * x_t
        y_t = C_t * h_t + D * x_t
    Vectorized across batch and orthogonal spatial axis for fast hardware-agnostic execution.
    """
    def __init__(self, d_inner: int, d_state: int = 16):
        super().__init__()
        self.d_inner = d_inner
        self.d_state = d_state

        # Project input to selective parameters: Delta, B, C
        self.x_proj = nn.Linear(d_inner, d_inner + 2 * d_state, bias=True)

        # Diagonal state matrix A initialized with HiPPO-inspired negative log-spectrum
        A_init = torch.arange(1, d_state + 1, dtype=torch.float32).unsqueeze(0).expand(d_inner, -1)
        self.A_log = nn.Parameter(torch.log(A_init))
        self.D = nn.Parameter(torch.ones(d_inner))

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        """
        Args:
            u: Sequence tensor of shape (Batch_Axis, Seq_Len, d_inner)
        Returns:
            y: Selective state space output of shape (Batch_Axis, Seq_Len, d_inner)
        """
        batch_n, seq_len, d_in = u.shape
        proj = self.x_proj(u)
        delta, B_sel, C_sel = torch.split(proj, [self.d_inner, self.d_state, self.d_state], dim=-1)

        # Positive discretization step size Delta_t in (0, inf)
        delta = F.softplus(delta)  # (batch_n, L, d_inner)
        A = -torch.exp(self.A_log)  # (d_inner, d_state)

        # Discretize A and B via zero-order hold / Euler approximation
        # delta_A: (batch_n, L, d_inner, d_state)
        delta_A = torch.exp(delta.unsqueeze(-1) * A.unsqueeze(0).unsqueeze(0))
        delta_B_u = (delta.unsqueeze(-1) * B_sel.unsqueeze(2)) * u.unsqueeze(-1)

        h = torch.zeros((batch_n, self.d_inner, self.d_state), device=u.device, dtype=u.dtype)
        outputs = []
        for t in range(seq_len):
            h = delta_A[:, t] * h + delta_B_u[:, t]
            y_t = torch.einsum("bds,bs->bd", h, C_sel[:, t]) + self.D * u[:, t]
            outputs.append(y_t)

        return torch.stack(outputs, dim=1)


class FourWayDirectionalSSMBlock(nn.Module):
    """
    2D Selective State Space Block scanning the 9x9 grid along all 4 physical propagation axes:
      - Axis 1 (Top -> Bottom): Gravity drop and ingredient chute flow along columns
      - Axis 2 (Bottom -> Top): Column support compression and upward cascade waves
      - Axis 3 (Left -> Right): Horizontal striped candy laser lines and row matches
      - Axis 4 (Right -> Left): Reverse horizontal interactions and cross-column constraints
    """
    def __init__(self, d_model: int = 256, d_state: int = 16, expand: int = 2, dropout: float = 0.05):
        super().__init__()
        self.d_model = d_model
        self.d_inner = d_model * expand
        self.branch_dim = self.d_inner // 4

        self.norm1 = nn.LayerNorm(d_model)
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)

        # Depthwise 2D convolution for local 3x3 match-3 geometry awareness before directional scan
        self.dw_conv2d = nn.Conv2d(
            self.d_inner, self.d_inner, kernel_size=3, padding=1, groups=self.d_inner, bias=True
        )

        # 4 Dedicated Selective Scan Engines for the 4 physical propagation directions
        self.ssm_top_down = SelectiveScan1D(self.branch_dim, d_state=d_state)
        self.ssm_bottom_up = SelectiveScan1D(self.branch_dim, d_state=d_state)
        self.ssm_left_right = SelectiveScan1D(self.branch_dim, d_state=d_state)
        self.ssm_right_left = SelectiveScan1D(self.branch_dim, d_state=d_state)

        self.merge_norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Sequential(
            nn.Linear(self.d_inner, d_model, bias=False),
            nn.Dropout(dropout)
        )

        # Channel-mixing Feed-Forward Network
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
            nn.Dropout(dropout)
        )

    def forward(self, x: torch.Tensor, H: int = 9, W: int = 9) -> torch.Tensor:
        """
        Args:
            x: Spatial token tensor of shape (B, H * W, d_model)
        Returns:
            Updated spatial token tensor of shape (B, H * W, d_model)
        """
        B, N, C = x.shape
        residual = x
        x_norm = self.norm1(x)

        xz = self.in_proj(x_norm)
        x_branch, z_gate = xz.chunk(2, dim=-1)  # Each (B, N, d_inner)

        # Local 2D depthwise conv
        x_2d = x_branch.transpose(1, 2).contiguous().view(B, self.d_inner, H, W)
        x_2d = F.silu(self.dw_conv2d(x_2d))  # (B, d_inner, H, W)

        # Partition channels across the 4 physical scan directions
        x_td, x_bu, x_lr, x_rl = torch.chunk(x_2d, 4, dim=1)  # Each (B, branch_dim, H, W)

        # 1. Top-to-Bottom scan along H (columns)
        seq_td = x_td.permute(0, 3, 2, 1).contiguous().view(B * W, H, self.branch_dim)
        out_td = self.ssm_top_down(seq_td).view(B, W, H, self.branch_dim).permute(0, 3, 2, 1)

        # 2. Bottom-to-Top scan along reversed H (columns bottom-up)
        seq_bu = torch.flip(x_bu, dims=[2]).permute(0, 3, 2, 1).contiguous().view(B * W, H, self.branch_dim)
        out_bu = self.ssm_bottom_up(seq_bu).view(B, W, H, self.branch_dim).permute(0, 3, 2, 1)
        out_bu = torch.flip(out_bu, dims=[2])

        # 3. Left-to-Right scan along W (rows)
        seq_lr = x_lr.permute(0, 2, 3, 1).contiguous().view(B * H, W, self.branch_dim)
        out_lr = self.ssm_left_right(seq_lr).view(B, H, W, self.branch_dim).permute(0, 3, 1, 2)

        # 4. Right-to-Left scan along reversed W (rows right-to-left)
        seq_rl = torch.flip(x_rl, dims=[3]).permute(0, 2, 3, 1).contiguous().view(B * H, W, self.branch_dim)
        out_rl = self.ssm_right_left(seq_rl).view(B, H, W, self.branch_dim).permute(0, 3, 1, 2)
        out_rl = torch.flip(out_rl, dims=[3])

        # Recombine 4 directional scans into (B, N, d_inner)
        merged_2d = torch.cat([out_td, out_bu, out_lr, out_rl], dim=1)
        merged_tokens = merged_2d.flatten(2).transpose(1, 2)

        # Gated SiLU modulation (Mamba architecture)
        gated = self.merge_norm(merged_tokens) * F.silu(z_gate)
        x = residual + self.out_proj(gated)

        # FFN block
        x = x + self.ffn(self.norm2(x))
        return x


class CandyMamba2DNet(nn.Module):
    """
    8-Layer 2D Vision Mamba Architecture (d=256) for Match-3 Physics & Cascade Prediction.
    Accepts canonical (B, 20, 9, 9) board tensor and outputs:
      - `policy_logits`: (B, 144) canonical swap logits
      - `win_prob`: (B, 1) calibrated win probability
      - `cascade_potential_map`: (B, 1, 9, 9) spatial heatmap of expected chain-reaction intensity
    """
    def __init__(
        self,
        in_channels: int = 20,
        d_model: int = 256,
        num_layers: int = 8,
        d_state: int = 16,
        expand: int = 2,
        num_actions: int = 144,
        board_h: int = 9,
        board_w: int = 9
    ):
        super().__init__()
        self.in_channels = in_channels
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_actions = num_actions
        self.board_h = board_h
        self.board_w = board_w

        # Spatial Patch / Cell Embedding Stem
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, d_model, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(d_model),
            nn.GELU(),
            nn.Conv2d(d_model, d_model, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(d_model),
            nn.GELU()
        )
        self.pos_embed = nn.Parameter(torch.randn(1, board_h * board_w, d_model) * 0.02)

        # 8 Stacked FourWayDirectionalSSMBlocks
        self.layers = nn.ModuleList([
            FourWayDirectionalSSMBlock(d_model=d_model, d_state=d_state, expand=expand)
            for _ in range(num_layers)
        ])
        self.final_norm = nn.LayerNorm(d_model)

        # Head 1: Canonical 144-Action Policy Head
        self.policy_head = nn.Sequential(
            nn.Conv2d(d_model, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.Conv2d(128, 64, kernel_size=1),
            nn.GELU(),
            nn.Flatten(),
            nn.Linear(64 * board_h * board_w, 256),
            nn.GELU(),
            nn.Linear(256, num_actions)
        )

        # Head 2: Calibrated Win Probability Head
        self.win_head = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.GELU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

        # Head 3: Spatial Cascade Potential Map (B, 1, 9, 9)
        # Predicts per-cell probability of participating in a multi-step gravity cascade
        self.cascade_map_head = nn.Sequential(
            nn.Conv2d(d_model, 64, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(64, 1, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, board_tensor: torch.Tensor) -> MambaOutput:
        """
        Args:
            board_tensor: (B, 20, 9, 9) canonical state tensor
        Returns:
            MambaOutput (supports both `(policy_logits, win_prob, cascade_potential_map) = net(x)`
            and dictionary key access `out["policy_logits"]`).
        """
        B, C, H, W = board_tensor.shape
        feat = self.stem(board_tensor)
        tokens = feat.flatten(2).transpose(1, 2) + self.pos_embed

        for layer in self.layers:
            tokens = layer(tokens, H=H, W=W)

        tokens = self.final_norm(tokens)
        feat_2d = tokens.transpose(1, 2).contiguous().view(B, self.d_model, H, W)

        policy_logits = self.policy_head(feat_2d)
        win_prob = self.win_head(tokens.mean(dim=1))
        cascade_potential_map = self.cascade_map_head(feat_2d)

        return MambaOutput(policy_logits, win_prob, cascade_potential_map)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def create_candy_mamba_2d_net(device: str = "cpu") -> CandyMamba2DNet:
    model = CandyMamba2DNet(
        in_channels=20,
        d_model=256,
        num_layers=8,
        d_state=16,
        expand=2,
        num_actions=144
    )
    return model.to(device)


if __name__ == "__main__":
    torch.manual_seed(42)
    net = create_candy_mamba_2d_net("cpu")
    params = net.count_parameters()
    print(f"[CandyMamba2DNet] Initialized with {params:,} trainable parameters ({params / 1e6:.2f}M)")

    dummy = torch.randn(2, 20, 9, 9)
    policy_logits, win_prob, cascade_map = net(dummy)
    print(f"  policy_logits shape:         {policy_logits.shape}")
    print(f"  win_prob shape:              {win_prob.shape}")
    print(f"  cascade_potential_map shape: {cascade_map.shape}")
