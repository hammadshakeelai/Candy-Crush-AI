"""
Liquid Time-Constant (LTC) & Closed-Form Continuous-Time (CfC) Neural Network (`LiquidNeuralCandyNet`).
Based on:
  - Hasani et al. (AAAI 2021), "Liquid Time-constant Networks"
  - Hasani et al. (Nature Machine Intelligence 2022), "Closed-form continuous-time neural networks"

Models Candy Crush board dynamics via a continuous-time liquid ordinary differential equation (ODE):
    dx/dt = -[w_tau + f(x, I(t))] * x(t) + A * f(x, I(t))
solved via both multi-step semi-implicit Euler ODE integration and the exact Closed-Form
Continuous-Time (CfC) exponential decay relaxation conditioned on move urgency and cascade volatility.
"""

import math
from typing import Dict, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class UrgencyVolatilityEstimator(nn.Module):
    """
    Computes adaptive integration time-steps Delta_t and cascade volatility from the raw
    20-channel board tensor:
      - Channels 7..11: Special candies (high cascade volatility)
      - Channels 12..14: Jelly density (objective pressure)
      - Channels 15..17: Chocolate, Licorice, Frosting blockers (hazard urgency)
      - Channel 18: Normalized moves remaining
    """
    def __init__(self, in_channels: int = 20, d_model: int = 256):
        super().__init__()
        self.spatial_urgency_net = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.SiLU(),
            nn.Conv2d(64, 1, kernel_size=1),
            nn.Softplus()
        )
        self.global_urgency_mlp = nn.Sequential(
            nn.Linear(4, 32),
            nn.SiLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, board_tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            board_tensor: (B, 20, 9, 9)
        Returns:
            delta_t: Spatial integration time step (B, 81, 1) in (0.05, 2.5)
            urgency_scalar: Global urgency index (B, 1) in [0, 1]
        """
        B, C, H, W = board_tensor.shape

        # Extract domain-specific structural signals
        special_density = board_tensor[:, 7:12].sum(dim=1).mean(dim=(1, 2), keepdim=False)
        jelly_density = board_tensor[:, 12:15].sum(dim=1).mean(dim=(1, 2), keepdim=False)
        blocker_density = board_tensor[:, 15:18].sum(dim=1).mean(dim=(1, 2), keepdim=False)
        moves_norm = board_tensor[:, 18].mean(dim=(1, 2), keepdim=False)

        stats = torch.stack([special_density, jelly_density, blocker_density, 1.0 - moves_norm], dim=-1)
        urgency_scalar = self.global_urgency_mlp(stats)  # (B, 1)

        # Spatial Delta_t map modulated by global urgency
        local_dt = self.spatial_urgency_net(board_tensor).flatten(2).transpose(1, 2)  # (B, 81, 1)
        delta_t = 0.05 + (0.5 + urgency_scalar.unsqueeze(-1)) * local_dt.clamp(max=2.0)
        return delta_t, urgency_scalar


class SpatialCfCCell(nn.Module):
    """
    Closed-form Continuous-time (CfC) + Liquid ODE Gating Cell (Hasani et al., Nature MI 2022).
    Simultaneously computes:
      1. Liquid ODE state derivative:
            dx/dt = -[w_tau + f(x, I)] * x + A * f(x, I)
      2. Closed-form continuous-time trajectory relaxation:
            x(t + dt) = sigma(-f(x, I) * dt) * g(x, I) + [1 - sigma(-[w_tau + f(x, I)] * dt)] * h(x, I)
    """
    def __init__(self, d_model: int = 256, board_h: int = 9, board_w: int = 9):
        super().__init__()
        self.d_model = d_model
        self.board_h = board_h
        self.board_w = board_w

        # Intrinsic positive base time-constant parameter w_tau > 0
        self.log_w_tau = nn.Parameter(torch.zeros(1, 1, d_model))
        # Equilibrium reversal potential A for the liquid synapse
        self.sensory_reversal_A = nn.Parameter(torch.ones(1, 1, d_model) * 1.0)

        # Spatial sensory backbone combining current state x(t) and external input I(t)
        self.sensory_conv = nn.Sequential(
            nn.Conv2d(d_model * 2, d_model, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(d_model),
            nn.SiLU()
        )
        self.norm = nn.LayerNorm(d_model)

        # Three CfC functional heads: f (liquid time-constant gate), g (backbone state), h (target asymptote)
        self.f_gate = nn.Linear(d_model, d_model)
        self.g_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.Tanh()
        )
        self.h_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.Tanh()
        )

    def forward(
        self,
        x_state: torch.Tensor,
        sensory_input: torch.Tensor,
        delta_t: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x_state: Current liquid hidden state (B, 81, d_model)
            sensory_input: Driving spatial input I(t) (B, 81, d_model)
            delta_t: Adaptive step size (B, 81, 1)
        Returns:
            x_next: Updated liquid state (B, 81, d_model)
            tau_eff: Effective input-dependent time constant 1 / (w_tau + |f|) of shape (B, 81, d_model)
        """
        B, N, D = x_state.shape
        x_2d = x_state.transpose(1, 2).contiguous().view(B, D, self.board_h, self.board_w)
        i_2d = sensory_input.transpose(1, 2).contiguous().view(B, D, self.board_h, self.board_w)

        joint_2d = self.sensory_conv(torch.cat([x_2d, i_2d], dim=1))
        joint = self.norm(joint_2d.flatten(2).transpose(1, 2))  # (B, 81, D)

        w_tau = F.softplus(self.log_w_tau) + 1e-3  # strictly positive base decay rate
        f_val = F.softplus(self.f_gate(joint))     # strictly positive state-dependent conductance
        g_val = self.g_head(joint)
        h_val = self.h_head(joint)

        # Effective liquid conductance rate: alpha = w_tau + f(x, I)
        conductance = w_tau + f_val
        tau_eff = 1.0 / (conductance + 1e-5)

        # 1. Closed-form Continuous-time (CfC) solution (Hasani et al. 2022 Eq. 6)
        gate_interp = torch.sigmoid(-conductance * delta_t)
        x_cfc = gate_interp * g_val + (1.0 - gate_interp) * h_val

        # 2. Semi-implicit Euler step of the exact LTC ODE:
        #    dx/dt = -conductance * x + A * f_val  =>  x_{t+dt} = (x + dt * A * f_val) / (1 + dt * conductance)
        x_ltc_ode = (x_state + delta_t * (self.sensory_reversal_A * f_val)) / (1.0 + delta_t * conductance)

        # Fuse exact CfC trajectory with semi-implicit LTC ODE integration
        x_next = 0.5 * (x_cfc + x_ltc_ode)
        return x_next, tau_eff


class LiquidNeuralCandyNet(nn.Module):
    """
    Liquid Time-Constant (LTC / CfC) Network for Candy Crush Saga.
    Evolves a continuous-time hidden state over K liquid ODE solver steps per board observation,
    dynamically adjusting its time constants based on move urgency and special candy volatility.
    """
    def __init__(
        self,
        in_channels: int = 20,
        d_model: int = 256,
        ode_unroll_steps: int = 4,
        num_actions: int = 144,
        board_h: int = 9,
        board_w: int = 9
    ):
        super().__init__()
        self.in_channels = in_channels
        self.d_model = d_model
        self.ode_unroll_steps = ode_unroll_steps
        self.num_actions = num_actions
        self.board_h = board_h
        self.board_w = board_w
        self.num_tokens = board_h * board_w

        # Sensory Encoder I(t)
        self.sensory_stem = nn.Sequential(
            nn.Conv2d(in_channels, d_model, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(d_model),
            nn.SiLU(),
            nn.Conv2d(d_model, d_model, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(d_model),
            nn.SiLU()
        )
        self.pos_embed = nn.Parameter(torch.randn(1, self.num_tokens, d_model) * 0.02)

        # Urgency & Cascade Volatility Time-Step Estimator
        self.urgency_estimator = UrgencyVolatilityEstimator(in_channels=in_channels, d_model=d_model)

        # Multi-step Liquid CfC / ODE Solver Cells
        self.liquid_cells = nn.ModuleList([
            SpatialCfCCell(d_model=d_model, board_h=board_h, board_w=board_w)
            for _ in range(ode_unroll_steps)
        ])
        self.final_norm = nn.LayerNorm(d_model)

        # Policy Head over 144 canonical swap actions
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

        # Calibrated Win Probability / Value Head conditioned on liquid state + urgency
        self.value_head = nn.Sequential(
            nn.Linear(d_model + 1, 128),
            nn.SiLU(),
            nn.Linear(128, 1)
        )

    def forward(
        self,
        board_tensor: torch.Tensor,
        initial_hidden: Optional[torch.Tensor] = None,
        legal_mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Canonical forward pass accepting (B, 20, 9, 9) board tensor.
        Evolves liquid state through `ode_unroll_steps` adaptive continuous-time steps.
        """
        B, C, H, W = board_tensor.shape

        # 1. Compute sensory drive I(t)
        sensory_2d = self.sensory_stem(board_tensor)
        sensory_tokens = sensory_2d.flatten(2).transpose(1, 2) + self.pos_embed  # (B, 81, d_model)

        # 2. Estimate adaptive integration time-steps Delta_t from urgency and cascade volatility
        delta_t, urgency_scalar = self.urgency_estimator(board_tensor)  # (B, 81, 1), (B, 1)

        # 3. Initialize or continue liquid hidden state x(0)
        if initial_hidden is None:
            x_state = sensory_tokens.clone()
        else:
            x_state = initial_hidden

        # 4. Evolve liquid ODE across unroll steps
        tau_list = []
        sub_dt = delta_t / float(self.ode_unroll_steps)
        for cell in self.liquid_cells:
            x_state, tau_eff = cell(x_state, sensory_tokens, sub_dt)
            tau_list.append(tau_eff)

        x_final = self.final_norm(x_state)  # (B, 81, d_model)
        x_2d = x_final.transpose(1, 2).contiguous().view(B, self.d_model, H, W)

        # 5. Compute policy logits
        policy_logits = self.policy_head(x_2d)
        if legal_mask is not None:
            policy_logits = policy_logits.masked_fill(~legal_mask.bool(), -1e9)

        # 6. Compute win probability and value
        x_global = x_final.mean(dim=1)  # (B, d_model)
        val_raw = self.value_head(torch.cat([x_global, urgency_scalar], dim=-1))  # (B, 1)
        win_prob = torch.sigmoid(val_raw)
        value = torch.tanh(val_raw)

        mean_tau = torch.stack(tau_list, dim=0).mean(dim=0)  # (B, 81, d_model)

        return {
            "policy_logits": policy_logits,
            "win_prob": win_prob,
            "value": value,
            "liquid_hidden_state": x_final,
            "liquid_time_constants": mean_tau,
            "volatility_dt": delta_t,
            "urgency_index": urgency_scalar
        }

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def create_liquid_candy_net(device: str = "cpu") -> LiquidNeuralCandyNet:
    """Factory function initializing LiquidNeuralCandyNet on the target device."""
    model = LiquidNeuralCandyNet(
        in_channels=20,
        d_model=256,
        ode_unroll_steps=4,
        num_actions=144
    )
    return model.to(device)


if __name__ == "__main__":
    torch.manual_seed(42)
    net = create_liquid_candy_net("cpu")
    params = net.count_parameters()
    print(f"[LiquidNeuralCandyNet] Initialized with {params:,} trainable parameters ({params / 1e6:.2f}M)")

    dummy_board = torch.randn(2, 20, 9, 9).abs()
    out = net(dummy_board)
    print(f"  policy_logits:         {out['policy_logits'].shape}")
    print(f"  win_prob:              {out['win_prob'].shape}")
    print(f"  value:                 {out['value'].shape}")
    print(f"  liquid_hidden_state:   {out['liquid_hidden_state'].shape}")
    print(f"  liquid_time_constants: {out['liquid_time_constants'].shape}")
    print(f"  volatility_dt:         {out['volatility_dt'].shape}")
