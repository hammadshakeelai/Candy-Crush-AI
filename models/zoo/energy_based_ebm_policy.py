"""
LeCun-Style Energy-Based Policy & Helmholtz Free-Energy Network (`EnergyBasedCandyNet`).
Based on:
  - LeCun et al. (2006), "A Tutorial on Energy-Based Learning"
  - Haarnoja et al. (ICML 2017), "Reinforcement Learning with Deep Energy-Based Policies"
  - Florence et al. (CoRL 2021), "Implicit Behavioral Cloning"

Assigns a scalar compatibility energy E_theta(s, a) in R^{B x 144} to every (board_state, swap_action)
pair via bilinear state-action energy coupling + local swap-dipole potential, inducing the Gibbs /
Boltzmann distribution:
    P_theta(a | s) = exp(-E_theta(s, a) / T) / sum_{a'} exp(-E_theta(s, a') / T)
and Helmholtz Free Energy state value potential:
    F_theta(s) = -T * log sum_{a'} exp(-E_theta(s, a') / T).
"""

import math
from typing import Dict, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class EnergyResidualBlock(nn.Module):
    """Spectral-smooth Residual Block with Swish/SiLU activations for smooth energy landscapes."""
    def __init__(self, channels: int = 256):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.gn1 = nn.GroupNorm(16, channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.gn2 = nn.GroupNorm(16, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.silu(self.gn1(self.conv1(x)))
        out = self.gn2(self.conv2(out))
        return F.silu(x + out)


class SwapDipoleEnergyHead(nn.Module):
    """
    Computes the local pairwise interaction energy E_local(s, a) for each of the 144 swap dipoles
    (r1, c1) <-> (r2, c2), plus a global state-action bilinear compatibility energy E_global(s, a).
    """
    def __init__(
        self,
        d_model: int = 256,
        num_actions: int = 144,
        board_h: int = 9,
        board_w: int = 9
    ):
        super().__init__()
        self.d_model = d_model
        self.num_actions = num_actions
        self.board_h = board_h
        self.board_w = board_w

        # Precompute canonical swap coordinate pairs (72 horizontal + 72 vertical)
        r1_list, c1_list, r2_list, c2_list, orient_list = [], [], [], [], []
        for r in range(board_h):
            for c in range(board_w - 1):
                r1_list.append(r)
                c1_list.append(c)
                r2_list.append(r)
                c2_list.append(c + 1)
                orient_list.append(0)  # Horizontal
        for r in range(board_h - 1):
            for c in range(board_w):
                r1_list.append(r)
                c1_list.append(c)
                r2_list.append(r + 1)
                c2_list.append(c)
                orient_list.append(1)  # Vertical

        self.register_buffer("idx_r1", torch.tensor(r1_list, dtype=torch.long))
        self.register_buffer("idx_c1", torch.tensor(c1_list, dtype=torch.long))
        self.register_buffer("idx_r2", torch.tensor(r2_list, dtype=torch.long))
        self.register_buffer("idx_c2", torch.tensor(c2_list, dtype=torch.long))
        self.register_buffer("idx_orient", torch.tensor(orient_list, dtype=torch.long))

        # Action coordinate & orientation embedding e(a) in R^{144 x d_model}
        self.action_embed = nn.Embedding(num_actions, d_model)
        self.orient_embed = nn.Embedding(2, d_model)

        # Pairwise dipole energy network operating on [f(r1,c1), f(r2,c2), |f1 - f2|, f1 * f2, e(a)]
        self.dipole_energy_mlp = nn.Sequential(
            nn.Linear(d_model * 3, d_model),
            nn.LayerNorm(d_model),
            nn.SiLU(),
            nn.Linear(d_model, 64),
            nn.SiLU(),
            nn.Linear(64, 1)
        )

        # Global board context energy coupling
        self.global_context_proj = nn.Sequential(
            nn.Conv2d(d_model, 64, kernel_size=1, bias=False),
            nn.GroupNorm(8, 64),
            nn.SiLU(),
            nn.Flatten(),
            nn.Linear(64 * board_h * board_w, d_model),
            nn.LayerNorm(d_model),
            nn.SiLU()
        )
        self.global_energy_bilinear = nn.Linear(d_model, num_actions)

    def forward(self, spatial_feat: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            spatial_feat: (B, d_model, 9, 9)
        Returns:
            total_energy: Scalar compatibility energy E_theta(s, a) of shape (B, 144)
            local_dipole_energy: Local swap dipole energy (B, 144)
            global_coupling_energy: Global context energy (B, 144)
        """
        B, D, H, W = spatial_feat.shape

        # Gather endpoint representations f1, f2 in R^{B x 144 x D}
        f1 = spatial_feat[:, :, self.idx_r1, self.idx_c1].transpose(1, 2)
        f2 = spatial_feat[:, :, self.idx_r2, self.idx_c2].transpose(1, 2)

        # Action embeddings
        action_ids = torch.arange(self.num_actions, device=spatial_feat.device)
        a_emb = (
            self.action_embed(action_ids) + self.orient_embed(self.idx_orient)
        ).unsqueeze(0).expand(B, -1, -1)  # (B, 144, D)

        # Symmetric + antisymmetric dipole features
        dipole_feat = torch.cat([f1 + f2, (f1 - f2).abs(), a_emb], dim=-1)  # (B, 144, 3*D)
        local_dipole_energy = self.dipole_energy_mlp(dipole_feat).squeeze(-1)  # (B, 144)

        # Global context coupling energy
        g_ctx = self.global_context_proj(spatial_feat)  # (B, D)
        global_coupling_energy = self.global_energy_bilinear(g_ctx)  # (B, 144)

        total_energy = local_dipole_energy + global_coupling_energy
        return total_energy, local_dipole_energy, global_coupling_energy


class EnergyBasedCandyNet(nn.Module):
    """
    LeCun-style Energy-Based Model (EBM) for Candy Crush Saga.
    Computes compatibility energy E_theta(s, a) in R^{B x 144}, Boltzmann policy logits
    -E_theta(s, a) / T, and Helmholtz Free Energy F_theta(s) for value estimation.
    """
    def __init__(
        self,
        in_channels: int = 20,
        d_model: int = 256,
        num_res_blocks: int = 6,
        num_actions: int = 144,
        init_temperature: float = 1.0,
        board_h: int = 9,
        board_w: int = 9
    ):
        super().__init__()
        self.in_channels = in_channels
        self.d_model = d_model
        self.num_res_blocks = num_res_blocks
        self.num_actions = num_actions
        self.board_h = board_h
        self.board_w = board_w

        # Learnable log-temperature log(T) for Boltzmann distribution
        self.log_temperature = nn.Parameter(torch.tensor(math.log(init_temperature)))

        # Deep spatial feature extractor
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, d_model, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(16, d_model),
            nn.SiLU()
        )
        self.res_tower = nn.ModuleList([
            EnergyResidualBlock(d_model) for _ in range(num_res_blocks)
        ])

        # State-Action Energy Head E_theta(s, a)
        self.energy_head = SwapDipoleEnergyHead(
            d_model=d_model,
            num_actions=num_actions,
            board_h=board_h,
            board_w=board_w
        )

        # Calibrated Win Probability Head combining Helmholtz Free Energy F_theta(s) and pooled state
        self.value_refine = nn.Sequential(
            nn.Linear(d_model + 2, 128),
            nn.SiLU(),
            nn.Linear(128, 1)
        )

    @property
    def temperature(self) -> torch.Tensor:
        return torch.exp(self.log_temperature).clamp(min=0.05, max=10.0)

    def compute_energies(self, board_tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns (energies, local_energy, global_energy, spatial_feat)."""
        h = self.stem(board_tensor)
        for block in self.res_tower:
            h = block(h)
        energies, local_e, global_e = self.energy_head(h)
        return energies, local_e, global_e, h

    def forward(
        self,
        board_tensor: torch.Tensor,
        legal_mask: Optional[torch.Tensor] = None,
        temperature_override: Optional[float] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Canonical forward pass accepting (B, 20, 9, 9) board tensor.
        Lower energy E_theta(s, a) corresponds to higher action probability:
            policy_logits = -E_theta(s, a) / T
        """
        energies, local_e, global_e, spatial_feat = self.compute_energies(board_tensor)

        temp = temperature_override if temperature_override is not None else self.temperature
        if legal_mask is not None:
            # Assign high barrier energy (+1e9) to illegal swaps
            masked_energies = energies.masked_fill(~legal_mask.bool(), 1e9)
        else:
            masked_energies = energies

        # Policy logits are negative scaled energies: -E(s, a) / T
        policy_logits = -masked_energies / temp
        boltzmann_probs = F.softmax(policy_logits, dim=-1)

        # Helmholtz Free Energy: F(s) = -T * log sum_a exp(-E(s, a) / T)
        free_energy = -temp * torch.logsumexp(-energies / temp, dim=-1, keepdim=True)  # (B, 1)
        min_energy = energies.min(dim=-1, keepdim=True)[0]  # (B, 1)

        # Calibrated win probability and value from Free Energy + global representation
        h_global = spatial_feat.mean(dim=(2, 3))  # (B, d_model)
        val_raw = self.value_refine(torch.cat([h_global, -free_energy, -min_energy], dim=-1))
        win_prob = torch.sigmoid(val_raw)
        value = torch.tanh(val_raw)

        return {
            "policy_logits": policy_logits,
            "win_prob": win_prob,
            "value": value,
            "energies": energies,
            "local_dipole_energy": local_e,
            "global_coupling_energy": global_e,
            "boltzmann_probs": boltzmann_probs,
            "helmholtz_free_energy": free_energy
        }

    def contrastive_energy_loss(
        self,
        board_tensor: torch.Tensor,
        target_actions: torch.Tensor,
        legal_mask: Optional[torch.Tensor] = None,
        margin: float = 1.0,
        l2_energy_reg: float = 1e-3
    ) -> Dict[str, torch.Tensor]:
        """
        LeCun Contrastive Energy Loss combining:
          1. Boltzmann InfoNCE Negative Log-Likelihood: E(s, a^+) + T * log sum_a' exp(-E(s, a')/T)
          2. Hinge Margin Ranking Loss: mean_{a^- != a^+} max(0, margin + E(s, a^+) - E(s, a^-))
          3. L2 Energy Magnitude Regularizer: E(s, a)^2 to prevent energy drift
        """
        out = self.forward(board_tensor, legal_mask=legal_mask)
        energies = out["energies"]  # (B, 144)
        policy_logits = out["policy_logits"]  # (B, 144)
        B = energies.shape[0]

        # 1. Boltzmann NLL Loss
        nll_loss = F.cross_entropy(policy_logits, target_actions)

        # 2. Contrastive Margin Loss pushing positive action energy below all negative actions
        pos_energy = energies.gather(1, target_actions.unsqueeze(-1))  # (B, 1)
        margin_violations = F.relu(margin + pos_energy - energies)     # (B, 144)
        # Zero out the target action's own comparison
        target_one_hot = F.one_hot(target_actions, num_classes=self.num_actions).bool()
        margin_violations = margin_violations.masked_fill(target_one_hot, 0.0)
        hinge_loss = margin_violations.mean()

        # 3. Energy square regularization
        reg_loss = l2_energy_reg * (energies.pow(2).mean())

        total_loss = nll_loss + 0.5 * hinge_loss + reg_loss
        return {
            "total_ebm_loss": total_loss,
            "boltzmann_nll_loss": nll_loss,
            "margin_hinge_loss": hinge_loss,
            "energy_l2_reg": reg_loss
        }

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def create_ebm_candy_net(device: str = "cpu") -> EnergyBasedCandyNet:
    """Factory function initializing EnergyBasedCandyNet on the target device."""
    model = EnergyBasedCandyNet(
        in_channels=20,
        d_model=256,
        num_res_blocks=6,
        num_actions=144,
        init_temperature=1.0
    )
    return model.to(device)


if __name__ == "__main__":
    torch.manual_seed(42)
    net = create_ebm_candy_net("cpu")
    params = net.count_parameters()
    print(f"[EnergyBasedCandyNet] Initialized with {params:,} trainable parameters ({params / 1e6:.2f}M)")

    dummy_board = torch.randn(2, 20, 9, 9)
    dummy_targets = torch.tensor([12, 85], dtype=torch.long)
    out = net(dummy_board)
    loss_dict = net.contrastive_energy_loss(dummy_board, dummy_targets)
    print(f"  policy_logits:         {out['policy_logits'].shape}")
    print(f"  win_prob:              {out['win_prob'].shape}")
    print(f"  value:                 {out['value'].shape}")
    print(f"  energies:              {out['energies'].shape}")
    print(f"  helmholtz_free_energy: {out['helmholtz_free_energy'].shape}")
    print(f"  total_ebm_loss:        {loss_dict['total_ebm_loss'].item():.4f}")
