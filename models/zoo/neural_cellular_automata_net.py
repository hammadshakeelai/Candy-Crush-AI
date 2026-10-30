r"""
AlphaCandy Model Zoo — Part 6.3: Growing Neural Cellular Automata Cascade Predictor Network
(`NeuralCACascadePredictorNet`).
============================================================================================
Based on:
  - Mordvintsev, Randazzo, Niklasson, & Levin (Distill 2020), "Growing Neural Cellular Automata"
  - Gilpin (NeurIPS 2019), "Cellular Automata as Convolutional Neural Networks"

Why Neural Cellular Automata (NCA) for Candy Crush?
---------------------------------------------------
Candy Crush physics is fundamentally a **2D Discrete Cellular Automaton** governed by local
transition rules that repeat iteratively until a fixed point (quiescent settled board) is reached:
  1. **Local Match Detection**: Horizontal and vertical 3/4/5-collinear neighbor checks.
  2. **Special Blast Propagation**: Row/column laser sweeps and $3\times 3$ Wrapped explosions.
  3. **Gravity Relaxation**: Unsupported candies fall straight down into empty cells below
     ($r \to r+1$) or slide diagonally around blockers while top-row dispensers spawn new candies.
  4. **Secondary Chain Cascades**: Settled candies trigger new matches for $T$ recursive ticks.

Standard feed-forward CNNs apply distinct weights at each layer and fail to generalize when a
board triggers a 6-step or 10-step gravity chain reaction. `NeuralCACascadePredictorNet` embeds
the board into an $S$-channel Cellular Automaton state (20 visible physical channels +
hidden chemical/momentum channels) and runs **$T = 12$ iterative local $3\times 3$ Sobel / Moore
neighborhood perception + update steps** with **stochastic cell-update masks** and **alive-cell
grid masking** to simulate multi-step gravity relaxation and chain-reaction cascades directly
inside the neural forward pass!
"""

import math
from typing import Dict, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class MooreSobelPerceptionFilter(nn.Module):
    """
    Mordvintsev et al. (2020) Fixed + Learnable $3\times 3$ Moore-Neighborhood Perception Bank.

    Applies depthwise $3\times 3$ perception kernels to every channel of the CA state $s_t \in \mathbb{R}^{B \times C \times 9 \times 9}$:
      1. Identity filter $K_{\text{id}}$ (self state at $(r, c)$)
      2. Horizontal Sobel gradient $K_x$ (detects left-right match boundaries & horizontal swaps)
      3. Vertical Sobel gradient $K_y$ (detects top-bottom gravity drop gradients)
      4. Moore isotropic Laplacian $K_{\Delta}$ (detects 8-neighbor Wrapped blast & local cluster density)
      5. Asymmetric Upward-Gravity sensor $K_{\text{grav}}$ (senses unsupported candies directly above at $r-1$)
      6. Learnable depthwise $3\times 3$ perception filters for adaptive blocker/special interactions.
    """

    def __init__(self, state_channels: int = 96):
        super().__init__()
        self.state_channels = state_channels

        # 5 canonical fixed physics filters (Identity, Sobel-X, Sobel-Y, Moore-Laplacian, Gravity-Above)
        k_id = torch.tensor([
            [0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0]
        ], dtype=torch.float32)

        k_sobel_x = torch.tensor([
            [-1.0, 0.0, 1.0],
            [-2.0, 0.0, 2.0],
            [-1.0, 0.0, 1.0]
        ], dtype=torch.float32) / 8.0

        k_sobel_y = torch.tensor([
            [-1.0, -2.0, -1.0],
            [ 0.0,  0.0,  0.0],
            [ 1.0,  2.0,  1.0]
        ], dtype=torch.float32) / 8.0

        k_laplacian = torch.tensor([
            [0.25,  0.50, 0.25],
            [0.50, -3.00, 0.50],
            [0.25,  0.50, 0.25]
        ], dtype=torch.float32) / 3.0

        # Asymmetric gravity kernel sensing candies immediately above (r-1, c) and below (r+1, c)
        k_gravity = torch.tensor([
            [ 0.15,  0.70,  0.15],
            [ 0.00, -0.50,  0.00],
            [-0.10, -0.30, -0.10]
        ], dtype=torch.float32)

        # Stack fixed kernels: (5, 1, 3, 3) -> repeat across all state_channels: (5 * C, 1, 3, 3)
        fixed_stack = torch.stack([k_id, k_sobel_x, k_sobel_y, k_laplacian, k_gravity], dim=0)  # (5, 3, 3)
        fixed_weight = fixed_stack.unsqueeze(0).repeat(state_channels, 1, 1, 1)                  # (C, 5, 3, 3)
        fixed_weight = fixed_weight.view(state_channels * 5, 1, 3, 3)
        self.register_buffer("fixed_kernels", fixed_weight)

        # Additional learnable depthwise 3x3 perception filter (1 per channel)
        self.learnable_dw = nn.Conv2d(
            state_channels,
            state_channels,
            kernel_size=3,
            padding=1,
            groups=state_channels,
            bias=False
        )

    def forward(self, ca_state: torch.Tensor) -> torch.Tensor:
        """
        Args:
            ca_state: (B, C, 9, 9) cellular automaton grid state
        Returns:
            perceived: (B, 6 * C, 9, 9) local Moore/Sobel perception tensor
        """
        B, C, H, W = ca_state.shape
        fixed_perc = F.conv2d(ca_state, self.fixed_kernels, padding=1, groups=C)  # (B, 5*C, 9, 9)
        learn_perc = self.learnable_dw(ca_state)                                   # (B, C, 9, 9)
        return torch.cat([fixed_perc, learn_perc], dim=1)                          # (B, 6*C, 9, 9)


class NeuralCAUpdateRule(nn.Module):
    """
    Shared Local Cellular Automaton Update Rule $\Delta s_t = f_\theta(\text{Perceive}(s_t))$
    with:
      1. $1\times 1$ Pointwise Reaction MLP (emulating local cell reaction-diffusion + gravity laws).
      2. Non-local Striped Laser & Portal Row/Column Broadcast Gate (so special candies in the
         CA state propagate across their row/column during cascade relaxation).
      3. Mordvintsev **Stochastic Cell-Update Mask** ($p_{\text{fire}} \in (0, 1]$) breaking synchronous
         oscillation artifacts and emulating asynchronous physics relaxation.
      4. **Alive-Cell Grid Masking** preventing void cutouts from leaking phantom candies.
    """

    def __init__(
        self,
        state_channels: int = 96,
        hidden_dim: int = 192,
        fire_rate: float = 0.85
    ):
        super().__init__()
        self.state_channels = state_channels
        self.fire_rate = fire_rate

        self.perception = MooreSobelPerceptionFilter(state_channels=state_channels)

        # 1x1 local reaction-update MLP (6 * C -> hidden_dim -> C)
        self.update_mlp = nn.Sequential(
            nn.Conv2d(state_channels * 6, hidden_dim, kernel_size=1, bias=False),
            nn.GroupNorm(8, hidden_dim),
            nn.GELU(),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1),
            nn.GELU()
        )
        # Final update delta projection initialized near zero for stable T=12 unrolling
        self.delta_out = nn.Conv2d(hidden_dim, state_channels, kernel_size=1)
        nn.init.zeros_(self.delta_out.weight)
        nn.init.zeros_(self.delta_out.bias)

        # Row + Column axial pooling gate to let striped candy explosions travel across lines
        self.line_blast_gate = nn.Sequential(
            nn.Conv2d(state_channels * 2, state_channels, kernel_size=1),
            nn.Sigmoid()
        )
        # Step-conditioned dampening gate (allows CA to transition from fast match burst to settling)
        self.step_embed = nn.Embedding(16, state_channels)

    def forward(
        self,
        ca_state: torch.Tensor,
        alive_mask: torch.Tensor,
        step_idx: int,
        stochastic_update: Optional[bool] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Executes a single Cellular Automaton perception + stochastic update step $t \to t+1$.

        Args:
            ca_state: (B, C, 9, 9) current CA grid state
            alive_mask: (B, 1, 9, 9) active playable board cell mask
            step_idx: integer tick $t \in \{0, \dots, T-1\}$
            stochastic_update: Whether to apply stochastic per-cell fire mask (defaults to self.training)

        Returns:
            next_state: (B, C, 9, 9) updated CA grid state
            step_activity: (B,) mean L2 state update magnitude (measures cascade intensity at tick t)
        """
        B, C, H, W = ca_state.shape
        if stochastic_update is None:
            stochastic_update = self.training

        # 1. Local 3x3 Sobel + Moore + Gravity perception
        perceived = self.perception(ca_state)  # (B, 6*C, 9, 9)
        raw_delta = self.delta_out(self.update_mlp(perceived))  # (B, C, 9, 9)

        # 2. Axial row/column broadcast modulation (models full-row/col striped sweeps)
        row_ctx = ca_state.mean(dim=3, keepdim=True).expand(B, C, H, W)
        col_ctx = ca_state.mean(dim=2, keepdim=True).expand(B, C, H, W)
        line_gate = self.line_blast_gate(torch.cat([row_ctx, col_ctx], dim=1))

        # Inject step-specific time embedding
        t_tensor = torch.tensor(min(step_idx, 15), device=ca_state.device, dtype=torch.long)
        t_mod = torch.tanh(self.step_embed(t_tensor)).view(1, C, 1, 1)

        delta = raw_delta * (1.0 + 0.3 * line_gate + 0.15 * t_mod)

        # 3. Mordvintsev Stochastic Cell-Update Mask (asynchronous cellular relaxation)
        if stochastic_update and self.fire_rate < 1.0:
            rand_mask = (
                torch.rand(B, 1, H, W, device=ca_state.device, dtype=ca_state.dtype) <= self.fire_rate
            ).float() / self.fire_rate
            delta = delta * rand_mask

        # 4. Alive-cell masking (void cutouts remain quiescent)
        delta = delta * alive_mask
        next_state = (ca_state + delta) * alive_mask
        # Clamp hidden state dynamic range for unconditional stability across long rollouts
        next_state = torch.clamp(next_state, min=-10.0, max=10.0)

        # Measure cascade kinetic activity at tick t
        step_activity = delta.pow(2).mean(dim=(1, 2, 3)).sqrt()  # (B,)
        return next_state, step_activity


class NeuralCACascadePredictorNet(nn.Module):
    """
    Growing Neural Cellular Automata Cascade Predictor Network (`NeuralCACascadePredictorNet`).

    Runs $T = 12$ iterative local $3\times 3$ Sobel/Moore-neighborhood perception + update steps
    with stochastic cell-update masks to simulate multi-step gravity relaxation and chain-reaction
    cascades directly inside the neural forward pass.

    Outputs:
      - `policy_logits`: (B, 144) canonical swap action logits informed by $T=12$ simulated cascades
      - `value`: (B, 1) expected board value in [-1, 1]
      - `win_prob`: (B, 1) calibrated win probability in [0, 1]
      - `settled_board_logits`: (B, 20, 9, 9) predicted post-cascade 20-channel board state after $T=12$ ticks
      - `cascade_activity_trajectory`: (B, 12) kinetic update energy at each CA tick $t=1..12$
      - `cascade_ignition_heatmap`: (B, 1, 9, 9) spatial map of cells triggering the largest chain reactions
      - `final_ca_state`: (B, state_channels, 9, 9) terminal $T=12$ cellular automaton state
    """

    def __init__(
        self,
        in_channels: int = 20,
        board_h: int = 9,
        board_w: int = 9,
        state_channels: int = 96,
        ca_hidden_dim: int = 192,
        num_ca_steps: int = 12,
        fire_rate: float = 0.85,
        num_actions: int = 144
    ):
        super().__init__()
        self.in_channels = in_channels
        self.board_h = board_h
        self.board_w = board_w
        self.num_nodes = board_h * board_w
        self.state_channels = state_channels
        self.num_ca_steps = num_ca_steps
        self.num_actions = num_actions

        # 1. Seed Encoder: lifts 20 visible channels into 96-channel CA seed state
        #    First 20 channels preserve residual correspondence with physical board channels
        self.seed_encoder = nn.Sequential(
            nn.Conv2d(in_channels, state_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, state_channels),
            nn.GELU(),
            nn.Conv2d(state_channels, state_channels, kernel_size=1)
        )

        # 2. Recurrent Shared Neural Cellular Automaton Update Rule (unrolled for T=12 steps)
        self.ca_rule = NeuralCAUpdateRule(
            state_channels=state_channels,
            hidden_dim=ca_hidden_dim,
            fire_rate=fire_rate
        )

        # 3. Auxiliary Heads for Post-Cascade Board Reconstruction & Ignition Heatmap
        self.settled_board_decoder = nn.Sequential(
            nn.Conv2d(state_channels, 64, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(64, in_channels, kernel_size=1)
        )
        self.ignition_head = nn.Sequential(
            nn.Conv2d(state_channels * 2, 32, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(32, 1, kernel_size=1),
            nn.Sigmoid()
        )

        # 4. Pre- vs. Post-Cascade Swap Policy Head (144 actions)
        # Compares initial seed state s_0, midpoint cascade state s_{T/2}, and settled state s_T
        multi_tick_channels = state_channels * 3 + 1
        self.policy_conv = nn.Sequential(
            nn.Conv2d(multi_tick_channels, 128, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, 128),
            nn.GELU(),
            nn.Conv2d(128, 48, kernel_size=1),
            nn.GELU()
        )
        pair_dim = state_channels * 4  # (s0_u, s0_v, sT_u, sT_v)
        self.horiz_swap_scorer = nn.Sequential(
            nn.Linear(pair_dim, 128),
            nn.GELU(),
            nn.Linear(128, 1)
        )
        self.vert_swap_scorer = nn.Sequential(
            nn.Linear(pair_dim, 128),
            nn.GELU(),
            nn.Linear(128, 1)
        )
        self.policy_fuse = nn.Linear(48 * self.num_nodes + num_actions, num_actions)

        # 5. Value & Win Probability Readout Heads (conditioned on settled CA state + cascade trajectory)
        value_in_dim = state_channels * 2 + num_ca_steps
        self.value_head = nn.Sequential(
            nn.Linear(value_in_dim, 128),
            nn.GELU(),
            nn.Linear(128, 1),
            nn.Tanh()
        )
        self.win_head = nn.Sequential(
            nn.Linear(value_in_dim, 128),
            nn.GELU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def _score_cascade_swaps(self, s_init: torch.Tensor, s_final: torch.Tensor) -> torch.Tensor:
        """
        Scores all 72 horizontal swaps and 72 vertical swaps using both initial cell states
        $s_0$ and settled post-cascade cell states $s_T$.
        """
        B, C, H, W = s_init.shape
        cat_grid = torch.cat([s_init, s_final], dim=1)  # (B, 2*C, 9, 9)

        # Horizontal adjacent swaps (r, c) <-> (r, c+1): 9x8 = 72
        h_u = cat_grid[:, :, :, :-1].permute(0, 2, 3, 1)  # (B, 9, 8, 2*C)
        h_v = cat_grid[:, :, :, 1:].permute(0, 2, 3, 1)   # (B, 9, 8, 2*C)
        h_in = torch.cat([h_u, h_v], dim=-1).reshape(B, 72, 4 * C)
        horiz_logits = self.horiz_swap_scorer(h_in).squeeze(-1)  # (B, 72)

        # Vertical adjacent swaps (r, c) <-> (r+1, c): 8x9 = 72
        v_u = cat_grid[:, :, :-1, :].permute(0, 2, 3, 1)  # (B, 8, 9, 2*C)
        v_v = cat_grid[:, :, 1:, :].permute(0, 2, 3, 1)   # (B, 8, 9, 2*C)
        v_in = torch.cat([v_u, v_v], dim=-1).reshape(B, 72, 4 * C)
        vert_logits = self.vert_swap_scorer(v_in).squeeze(-1)    # (B, 72)

        return torch.cat([horiz_logits, vert_logits], dim=-1)    # (B, 144)

    def forward(
        self,
        board_tensor: torch.Tensor,
        legal_mask: Optional[torch.Tensor] = None,
        num_steps: Optional[int] = None,
        stochastic_update: Optional[bool] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Runs $T = 12$ iterative Neural Cellular Automata perception + update ticks to simulate
        gravity relaxation and chain-reaction cascades before predicting policy and value.
        """
        B, C, H, W = board_tensor.shape
        T = num_steps if num_steps is not None else self.num_ca_steps

        # Extract active playable cell mask (channel 18); fallback to all 1s if zero-initialized
        raw_alive = board_tensor[:, 18:19]  # (B, 1, 9, 9)
        has_alive = (raw_alive.sum(dim=(1, 2, 3), keepdim=True) > 0).float()
        alive_mask = (raw_alive > 0.1).float() * has_alive + (1.0 - has_alive)

        # Step 1: Initialize CA seed state s_0 (anchoring first 20 channels to physical board)
        s_0 = self.seed_encoder(board_tensor)
        s_0 = torch.cat([board_tensor + s_0[:, :C], s_0[:, C:]], dim=1) * alive_mask  # (B, 96, 9, 9)

        # Step 2: Unroll T=12 iterative local Sobel/Moore CA perception + update steps
        ca_state = s_0
        s_mid = s_0
        mid_step = max(1, T // 2)
        activity_list = []

        for t in range(T):
            ca_state, step_act = self.ca_rule(
                ca_state,
                alive_mask=alive_mask,
                step_idx=t,
                stochastic_update=stochastic_update
            )
            activity_list.append(step_act)
            if t + 1 == mid_step:
                s_mid = ca_state

        # Pad or slice activity trajectory to canonical length self.num_ca_steps (12)
        cascade_activity_trajectory = torch.stack(activity_list, dim=-1)  # (B, T)
        if T < self.num_ca_steps:
            pad = torch.zeros(B, self.num_ca_steps - T, device=board_tensor.device, dtype=board_tensor.dtype)
            cascade_feat = torch.cat([cascade_activity_trajectory, pad], dim=-1)
        else:
            cascade_feat = cascade_activity_trajectory[:, :self.num_ca_steps]

        # Step 3: Predict settled 20-channel board state & spatial cascade ignition heatmap
        settled_board_logits = self.settled_board_decoder(ca_state) * alive_mask
        cascade_ignition_heatmap = self.ignition_head(torch.cat([s_0, ca_state - s_0], dim=1)) * alive_mask

        # Step 4: Policy logits (B, 144) combining initial s_0, midpoint s_{T/2}, and settled s_T
        multi_tick = torch.cat([s_0, s_mid, ca_state, cascade_ignition_heatmap], dim=1)
        conv_flat = self.policy_conv(multi_tick).flatten(1)
        pair_logits = self._score_cascade_swaps(s_0, ca_state)
        policy_logits = pair_logits + self.policy_fuse(torch.cat([conv_flat, pair_logits], dim=-1))

        if legal_mask is not None:
            policy_logits = policy_logits.masked_fill(~legal_mask.bool(), -1e9)

        # Step 5: Global Value & Win Probability Readout
        s_mean = ca_state.mean(dim=(2, 3))
        s_max = ca_state.amax(dim=(2, 3))
        global_readout = torch.cat([s_mean, s_max, cascade_feat], dim=-1)

        value = self.value_head(global_readout)
        win_prob = self.win_head(global_readout)

        return {
            "policy_logits": policy_logits,
            "value": value,
            "win_prob": win_prob,
            "settled_board_logits": settled_board_logits,
            "cascade_activity_trajectory": cascade_activity_trajectory,
            "cascade_ignition_heatmap": cascade_ignition_heatmap,
            "final_ca_state": ca_state,
        }

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


if __name__ == "__main__":
    torch.manual_seed(42)
    model = NeuralCACascadePredictorNet(num_ca_steps=12)
    params = model.count_parameters()
    print(f"[NeuralCACascadePredictorNet] Trainable Parameters: {params:,}")

    dummy_board = torch.randn(2, 20, 9, 9)
    dummy_board[:, 18] = 1.0  # Active grid

    out = model(dummy_board, stochastic_update=True)
    print("  policy_logits:", out["policy_logits"].shape)
    print("  value:", out["value"].shape)
    print("  win_prob:", out["win_prob"].shape)
    print("  settled_board_logits:", out["settled_board_logits"].shape)
    print("  cascade_activity_trajectory (T=12):", out["cascade_activity_trajectory"].shape)
    print("  cascade_ignition_heatmap:", out["cascade_ignition_heatmap"].shape)
    print("  final_ca_state:", out["final_ca_state"].shape)
