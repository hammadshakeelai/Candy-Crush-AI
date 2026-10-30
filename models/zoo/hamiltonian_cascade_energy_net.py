"""
AlphaCandy Model Zoo — Physics-Informed Hamiltonian Cascade Potential Energy Network
(`CascadePotentialFieldNet`).

Based on:
  - Greydanus et al. (NeurIPS 2019), "Hamiltonian Neural Networks"
  - Raissi et al. (JCP 2019), "Physics-Informed Neural Networks"

Models the 9x9 Candy Crush board as a discrete 2D Gravitational Potential Energy Field:
  1. Column-Wise Potential Energy Field U(r, c):
     Computes analytical + learned gravitational potential energy for each cell (r, c) based on:
       - Color-match alignment between upper rows (0 .. r-1) and lower rows (r+1 .. 8) across
         column c and adjacent columns (c-1, c+1) that would collide under downward gravity drop.
       - Gravitational elevation head g * (r + 1) (bottom-row triggers carry maximum column overburden).
       - Stored chemical/explosive potential of special candies (Striped, Wrapped, Color Bomb).
  2. Symplectic Hamiltonian Evolution Blocks:
     Evolves generalized board coordinates q (spatial configuration) and generalized cascade momenta p
     (downward avalanche velocity) via symplectic leapfrog updates on H(q, p) = U(q) + K(p).
  3. Kinetic Cascade Release Head Delta_K(s, a) in R^(B x 144):
     Predicts the downward avalanche kinetic energy unleashed by each of the 144 canonical swaps,
     identifying bottom-row triggers that maximize cascade chain reactions.
"""

from typing import Dict, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class GravitationalAlignmentPotentialModule(nn.Module):
    """
    Physics-Informed Analytical + Learnable Column-Wise Potential Energy Extractor.
    For every cell (r, c) on the 9x9 grid, evaluates how much color-match alignment exists
    between upper cells (0 .. r-1) that will fall under gravity and lower/lateral cells (r .. 8)
    when cell (r, c) is perturbed or cleared.
    """

    def __init__(self, board_h: int = 9, board_w: int = 9, d_out: int = 32):
        super().__init__()
        self.board_h = board_h
        self.board_w = board_w

        # Precompute strictly upper mask (r_up < r) and lower/same mask (r_low >= r)
        # upper_weights[r, r_up] decays smoothly with vertical drop distance (r - r_up)
        upper_weights = torch.zeros(board_h, board_h, dtype=torch.float32)
        lower_weights = torch.zeros(board_h, board_h, dtype=torch.float32)
        for r in range(board_h):
            for r_up in range(r):
                dist = float(r - r_up)
                upper_weights[r, r_up] = 1.0 / dist  # 1-row, 2-row, 3-row drop resonance
            for r_low in range(r, board_h):
                dist = float(r_low - r + 1)
                lower_weights[r, r_low] = 1.0 / dist

        # Column overburden weight: bottom rows (r closer to 8) support taller columns above them
        elevation_burden = torch.linspace(0.25, 1.0, steps=board_h).view(1, 1, board_h, 1)

        self.register_buffer("upper_weights", upper_weights)
        self.register_buffer("lower_weights", lower_weights)
        self.register_buffer("elevation_burden", elevation_burden)

        # Project physical potential features into d_out channels
        self.potential_proj = nn.Sequential(
            nn.Conv2d(10, d_out, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(d_out),
            nn.SiLU(),
            nn.Conv2d(d_out, d_out, kernel_size=1)
        )

    def forward(self, board_tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            board_tensor: (B, 20, 9, 9) canonical state tensor
        Returns:
            analytic_U: (B, 1, 9, 9) analytical gravitational potential energy field U_0(r, c)
            potential_feat: (B, d_out, 9, 9) multi-channel potential field representation
        """
        B, C, H, W = board_tensor.shape

        # Channels 0..5: 6 candy color one-hot planes (B, 6, 9, 9)
        colors = board_tensor[:, 0:6]
        active_mask = board_tensor[:, 18:19] if C > 18 else torch.ones(B, 1, H, W, device=board_tensor.device)

        # Weighted color profile of upper rows (0 .. r-1) above each row r:
        # upper_weights: (H, H_up), colors: (B, 6, H_up, W) -> upper_color_field: (B, 6, H, W)
        upper_color_field = torch.einsum("ru,bcuk->bcrk", self.upper_weights, colors)

        # Weighted color profile of lower rows (r .. 8) at and below each row r:
        lower_color_field = torch.einsum("rl,bclk->bcrl", self.lower_weights, colors)

        # Also include horizontal neighbor color profiles from columns (c-1, c+1) for lateral match-3 completion
        lower_left = F.pad(lower_color_field[:, :, :, :-1], (1, 0, 0, 0))
        lower_right = F.pad(lower_color_field[:, :, :, 1:], (0, 1, 0, 0))
        lateral_lower_field = 0.5 * lower_color_field + 0.25 * (lower_left + lower_right)

        # Color-match alignment resonance between falling upper candies and receiving lower/lateral cells
        # Inner product across the 6 color channels -> (B, 1, 9, 9)
        alignment_resonance = (upper_color_field * lateral_lower_field).sum(dim=1, keepdim=True)

        # Stored chemical/explosive potential from special candies (Striped ch 6,7, Wrapped ch 8, Color Bomb ch 9)
        special_energy = (
            1.5 * board_tensor[:, 6:7]
            + 1.5 * board_tensor[:, 7:8]
            + 2.0 * board_tensor[:, 8:9]
            + 3.0 * board_tensor[:, 9:10]
        )

        # Blocker fracture potential (Jelly ch 10, Frosting/Lock/Chocolate ch 11..14)
        blocker_density = board_tensor[:, 10:15].sum(dim=1, keepdim=True)

        # Total analytical gravitational potential U_0(r, c) weighted by column overburden g * h(r)
        analytic_U = (
            (alignment_resonance + 0.6 * special_energy + 0.3 * blocker_density)
            * self.elevation_burden
            * active_mask
        )

        # Concatenate 6 per-color alignment planes + alignment_resonance + special_energy + blocker_density + analytic_U = 10 ch
        per_color_resonance = upper_color_field * lateral_lower_field  # (B, 6, 9, 9)
        phys_stack = torch.cat(
            [per_color_resonance, alignment_resonance, special_energy, blocker_density, analytic_U],
            dim=1
        )
        potential_feat = self.potential_proj(phys_stack)
        return analytic_U, potential_feat


class SymplecticHamiltonianBlock(nn.Module):
    """
    Symplectic Leapfrog Integrator Block evolving generalized board coordinates q (potential configuration)
    and generalized cascade momenta p (kinetic avalanche propagation) along downward gravity columns:
        p_{t + 1/2} = p_t - (dt / 2) * dU/dq(q_t)
        q_{t + 1}   = q_t + dt * dK/dp(p_{t + 1/2})
        p_{t + 1}   = p_{t + 1/2} - (dt / 2) * dU/dq(q_{t + 1})
    """

    def __init__(self, half_dim: int = 96, step_size: float = 0.25):
        super().__init__()
        self.half_dim = half_dim
        self.step_size = nn.Parameter(torch.tensor(step_size))

        # Gradient of Potential Energy dU/dq modeled via vertical + 2D convolutional force operator
        self.grad_U = nn.Sequential(
            nn.Conv2d(half_dim, half_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(half_dim),
            nn.SiLU(),
            nn.Conv2d(half_dim, half_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(half_dim)
        )

        # Gradient of Kinetic Energy dK/dp (models downward momentum propagation along columns)
        self.grad_K = nn.Sequential(
            nn.Conv2d(half_dim, half_dim, kernel_size=(3, 1), padding=(1, 0), bias=False),
            nn.BatchNorm2d(half_dim),
            nn.SiLU(),
            nn.Conv2d(half_dim, half_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(half_dim)
        )

    def forward(self, q: torch.Tensor, p: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        dt = torch.clamp(self.step_size, min=0.05, max=0.5)
        # Half-step momentum kick
        p_half = p - 0.5 * dt * self.grad_U(q)
        # Full-step position drift
        q_next = q + dt * self.grad_K(p_half)
        # Half-step momentum kick
        p_next = p_half - 0.5 * dt * self.grad_U(q_next)
        return q_next, p_next


class CascadePotentialFieldNet(nn.Module):
    """
    Physics-Informed Hamiltonian Cascade Potential Field Network (`CascadePotentialFieldNet`).
    Accepts canonical (B, 20, 9, 9) board state tensor and outputs:
      - `policy_logits` (B, 144): Swap logits boosted by predicted kinetic cascade release
      - `value` (B, 1): Expected game outcome in [-1, 1]
      - `win_prob` (B, 1): Calibrated win probability in [0, 1]
      - `potential_energy_field` (B, 1, 9, 9): Column-wise gravitational potential U(r, c)
      - `kinetic_cascade_release` (B, 144): Predicted downward avalanche energy Delta_K(s, a) per swap
      - `total_board_potential` (B, 1): Integrated board potential energy sum_{r,c} U(r, c)
    """

    def __init__(
        self,
        in_channels: int = 20,
        d_model: int = 192,
        num_symplectic_steps: int = 5,
        num_actions: int = 144,
        board_h: int = 9,
        board_w: int = 9
    ):
        super().__init__()
        self.in_channels = in_channels
        self.d_model = d_model
        self.half_dim = d_model // 2
        self.num_symplectic_steps = num_symplectic_steps
        self.num_actions = num_actions
        self.board_h = board_h
        self.board_w = board_w

        # 1. Physics-Informed Column Potential Alignment Extractor
        self.gravity_potential_module = GravitationalAlignmentPotentialModule(
            board_h=board_h,
            board_w=board_w,
            d_out=32
        )

        # 2. Initial State Encoder into Generalized Coordinates q_0 and Momenta p_0
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels + 32, d_model, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(d_model),
            nn.SiLU(),
            nn.Conv2d(d_model, d_model, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(d_model),
            nn.SiLU()
        )

        # 3. Symplectic Hamiltonian Cascade Integrator Tower
        self.hamiltonian_blocks = nn.ModuleList([
            SymplecticHamiltonianBlock(half_dim=self.half_dim, step_size=0.25)
            for _ in range(num_symplectic_steps)
        ])

        # 4. Learned Refined Potential Field Head U(r, c) in R^(B x 1 x 9 x 9)
        self.potential_field_head = nn.Sequential(
            nn.Conv2d(self.half_dim, 32, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(32, 1, kernel_size=1),
            nn.Softplus()
        )

        # 5. Edge-Level Kinetic Cascade Release Head Delta_K(s, a) for all 144 swaps
        # Scores 72 horizontal swaps (r, c)<->(r, c+1) and 72 vertical swaps (r, c)<->(r+1, c)
        self.horiz_kinetic_head = nn.Sequential(
            nn.Linear(d_model * 2 + 2, 128),
            nn.SiLU(),
            nn.Linear(128, 1),
            nn.Softplus()
        )
        self.vert_kinetic_head = nn.Sequential(
            nn.Linear(d_model * 2 + 2, 128),
            nn.SiLU(),
            nn.Linear(128, 1),
            nn.Softplus()
        )

        # 6. Tactical Policy Head fused with Kinetic Cascade Release Delta_K(s, a)
        self.policy_conv = nn.Sequential(
            nn.Conv2d(d_model, 48, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(48),
            nn.SiLU()
        )
        self.policy_mlp = nn.Sequential(
            nn.Linear(48 * board_h * board_w + num_actions, 256),
            nn.SiLU(),
            nn.Linear(256, num_actions)
        )
        self.kinetic_boost_scale = nn.Parameter(torch.tensor(0.5))

        # 7. Hamiltonian Energy Value & Win Probability Heads
        self.value_head = nn.Sequential(
            nn.Linear(d_model + 2, 128),
            nn.SiLU(),
            nn.Linear(128, 1)
        )

    def _compute_swap_kinetic_release(
        self,
        phase_state: torch.Tensor,
        potential_field: torch.Tensor
    ) -> torch.Tensor:
        """
        Computes Kinetic Cascade Release Delta_K(s, a) for all 144 swaps:
          - 0..71:   Horizontal swaps (r, c) <-> (r, c+1)
          - 72..143: Vertical swaps   (r, c) <-> (r+1, c)
        Explicitly incorporates the local potential energy U(r, c) of the swapped pair so
        bottom-row swaps under high-potential columns yield maximum kinetic release.
        """
        B, D, H, W = phase_state.shape

        # Horizontal pairs: (B, 9, 8, 2*D + 2)
        h_left = phase_state[:, :, :, :-1].permute(0, 2, 3, 1)
        h_right = phase_state[:, :, :, 1:].permute(0, 2, 3, 1)
        u_left = potential_field[:, :, :, :-1].permute(0, 2, 3, 1)
        u_right = potential_field[:, :, :, 1:].permute(0, 2, 3, 1)
        h_feat = torch.cat([h_left, h_right, u_left, u_right], dim=-1).reshape(B, 72, 2 * D + 2)
        delta_k_horiz = self.horiz_kinetic_head(h_feat).squeeze(-1)  # (B, 72)

        # Vertical pairs: (B, 8, 9, 2*D + 2)
        v_top = phase_state[:, :, :-1, :].permute(0, 2, 3, 1)
        v_bot = phase_state[:, :, 1:, :].permute(0, 2, 3, 1)
        u_top = potential_field[:, :, :-1, :].permute(0, 2, 3, 1)
        u_bot = potential_field[:, :, 1:, :].permute(0, 2, 3, 1)
        v_feat = torch.cat([v_top, v_bot, u_top, u_bot], dim=-1).reshape(B, 72, 2 * D + 2)
        delta_k_vert = self.vert_kinetic_head(v_feat).squeeze(-1)   # (B, 72)

        return torch.cat([delta_k_horiz, delta_k_vert], dim=-1)      # (B, 144)

    def forward(
        self,
        board_tensor: torch.Tensor,
        legal_mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Executes physics-informed potential field extraction, symplectic Hamiltonian integration,
        and kinetic cascade release prediction.
        """
        B, C, H, W = board_tensor.shape

        # Step 1: Analytical & multi-channel column-wise potential energy field
        analytic_U, potential_feat = self.gravity_potential_module(board_tensor)

        # Step 2: Encode initial phase-space state (q_0, p_0)
        x = self.stem(torch.cat([board_tensor, potential_feat], dim=1))
        q, p = torch.chunk(x, 2, dim=1)  # Each (B, half_dim, 9, 9)

        # Track Hamiltonian energy along trajectory
        q_init_norm = q.pow(2).mean(dim=(1, 2, 3), keepdim=False)
        p_init_norm = p.pow(2).mean(dim=(1, 2, 3), keepdim=False)

        # Step 3: Symplectic Leapfrog Evolution
        for block in self.hamiltonian_blocks:
            q, p = block(q, p)

        phase_state = torch.cat([q, p], dim=1)  # (B, d_model, 9, 9)

        # Step 4: Total Potential Energy Field U(r, c) combining analytical + learned potential
        learned_U = self.potential_field_head(q)
        potential_energy_field = 0.5 * analytic_U + 0.5 * learned_U  # (B, 1, 9, 9)
        total_board_potential = potential_energy_field.sum(dim=(2, 3))  # (B, 1)

        # Step 5: Predict Kinetic Cascade Release Delta_K(s, a) for all 144 swaps
        kinetic_cascade_release = self._compute_swap_kinetic_release(phase_state, potential_energy_field)
        total_kinetic_capacity = kinetic_cascade_release.mean(dim=-1, keepdim=True)  # (B, 1)

        # Step 6: Policy logits fused with Kinetic Cascade Release boost
        spatial_p = self.policy_conv(phase_state).flatten(1)
        base_logits = self.policy_mlp(torch.cat([spatial_p, kinetic_cascade_release], dim=-1))
        policy_logits = base_logits + self.kinetic_boost_scale * kinetic_cascade_release

        if legal_mask is not None:
            policy_logits = policy_logits.masked_fill(~legal_mask.bool(), -1e9)

        # Step 7: Value & Win Probability conditioned on total Hamiltonian energy H = U + K
        global_phase = phase_state.mean(dim=(2, 3))
        val_in = torch.cat([global_phase, total_board_potential, total_kinetic_capacity], dim=-1)
        val_raw = self.value_head(val_in)
        value = torch.tanh(val_raw)
        win_prob = torch.sigmoid(val_raw)

        # Symplectic phase-space energy conservation drift metric
        q_final_norm = q.pow(2).mean(dim=(1, 2, 3), keepdim=False)
        p_final_norm = p.pow(2).mean(dim=(1, 2, 3), keepdim=False)
        hamiltonian_drift = ((q_final_norm + p_final_norm) - (q_init_norm + p_init_norm)).pow(2).mean()

        return {
            "policy_logits": policy_logits,
            "value": value,
            "win_prob": win_prob,
            "potential_energy_field": potential_energy_field,
            "kinetic_cascade_release": kinetic_cascade_release,
            "total_board_potential": total_board_potential,
            "hamiltonian_drift_loss": hamiltonian_drift,
        }

    def physics_energy_conservation_loss(
        self,
        out_t: Dict[str, torch.Tensor],
        taken_actions: torch.Tensor,
        observed_cascade_score: torch.Tensor
    ) -> torch.Tensor:
        """
        Physics-informed supervision loss aligning predicted Kinetic Cascade Release Delta_K(s, a)
        with observed cascade energy / multi-match chain reaction magnitude, regularized by
        symplectic Hamiltonian drift.
        """
        pred_delta_k = out_t["kinetic_cascade_release"].gather(1, taken_actions.unsqueeze(-1)).squeeze(-1)
        cascade_mse = F.smooth_l1_loss(pred_delta_k, observed_cascade_score.float())
        return cascade_mse + 0.05 * out_t["hamiltonian_drift_loss"]

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def create_cascade_potential_field_net(device: str = "cpu") -> CascadePotentialFieldNet:
    model = CascadePotentialFieldNet(
        in_channels=20,
        d_model=192,
        num_symplectic_steps=5,
        num_actions=144
    )
    return model.to(device)


if __name__ == "__main__":
    torch.manual_seed(42)
    net = create_cascade_potential_field_net("cpu")
    params = net.count_parameters()
    print(f"[CascadePotentialFieldNet] Trainable parameters: {params:,} ({params / 1e6:.2f}M)")

    dummy_board = torch.rand(2, 20, 9, 9)
    out = net(dummy_board)
    phys_loss = net.physics_energy_conservation_loss(
        out, taken_actions=torch.tensor([10, 85]), observed_cascade_score=torch.tensor([3.5, 1.2])
    )
    print(f"  policy_logits:           {out['policy_logits'].shape}")
    print(f"  value:                   {out['value'].shape}")
    print(f"  win_prob:                {out['win_prob'].shape}")
    print(f"  potential_energy_field:  {out['potential_energy_field'].shape}")
    print(f"  kinetic_cascade_release: {out['kinetic_cascade_release'].shape}")
    print(f"  total_board_potential:   {out['total_board_potential'].shape}")
    print(f"  physics_energy_loss:     {phys_loss.item():.4f}")
