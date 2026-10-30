"""DreamerV3 Discrete World-Model Baseline (Hafner et al., Nature 2023 / 2025).

Implements the canonical 16x16 categorical discrete latent bottleneck with
1% uniform unimix regularization, straight-through gradients, RMSNorm + SiLU
blocks, and symlog scalar value decoding adapted to 9x9 Candy Crush boards.
"""

from typing import Dict, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


def symlog(x: torch.Tensor) -> torch.Tensor:
    return torch.sign(x) * torch.log1p(torch.abs(x))


def symexp(x: torch.Tensor) -> torch.Tensor:
    return torch.sign(x) * (torch.expm1(torch.abs(x)))


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization used throughout DreamerV3."""

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return x * rms * self.scale


class DreamerV3DiscreteRSSMNet(nn.Module):
    """Hafner et al. (2023) DreamerV3 architecture with 16x16 categorical latents.

    Encodes `(B, 20, 9, 9)` into 16 categorical variables of 16 classes each
    (256-dim discrete latent bottleneck) using 1% uniform unimix and
    straight-through estimator, then decodes actor `policy_logits` and symlog `win_prob`.
    """

    def __init__(
        self,
        in_channels: int = 20,
        deter_dim: int = 256,
        stoch_vars: int = 16,
        stoch_classes: int = 16,
        num_actions: int = 144,
        unimix_ratio: float = 0.01,
    ) -> None:
        super().__init__()
        self.deter_dim = deter_dim
        self.stoch_vars = stoch_vars
        self.stoch_classes = stoch_classes
        self.latent_dim = stoch_vars * stoch_classes
        self.unimix_ratio = unimix_ratio

        self.cnn_encoder = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(64, 96, kernel_size=3, stride=3),  # 9x9 -> 3x3
            nn.SiLU(inplace=True),
            nn.Flatten(),
        )

        self.deter_proj = nn.Sequential(
            nn.Linear(96 * 9, deter_dim),
            RMSNorm(deter_dim),
            nn.SiLU(inplace=True),
        )

        # Posterior z ~ q(z | h, x) and Prior z_hat ~ p(z | h) for KL regularization
        self.posterior_head = nn.Linear(deter_dim, self.latent_dim)
        self.prior_head = nn.Sequential(
            nn.Linear(deter_dim, deter_dim),
            RMSNorm(deter_dim),
            nn.SiLU(inplace=True),
            nn.Linear(deter_dim, self.latent_dim),
        )

        joint_dim = deter_dim + self.latent_dim
        self.actor_trunk = nn.Sequential(
            nn.Linear(joint_dim, 256),
            RMSNorm(256),
            nn.SiLU(inplace=True),
            nn.Linear(256, num_actions),
        )
        self.critic_symlog_head = nn.Sequential(
            nn.Linear(joint_dim, 128),
            RMSNorm(128),
            nn.SiLU(inplace=True),
            nn.Linear(128, 1),
        )

    def _sample_discrete_latent(self, logits: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        b = logits.size(0)
        logits = logits.view(b, self.stoch_vars, self.stoch_classes)
        probs = F.softmax(logits, dim=-1)
        # 1% uniform unimix to prevent dead categorical latents (Hafner et al. 2023)
        uniform = torch.full_like(probs, 1.0 / self.stoch_classes)
        probs = (1.0 - self.unimix_ratio) * probs + self.unimix_ratio * uniform

        hard_idx = probs.argmax(dim=-1)
        one_hot = F.one_hot(hard_idx, num_classes=self.stoch_classes).float()
        # Straight-through gradient estimator
        z_st = one_hot + probs - probs.detach()
        return z_st.view(b, self.latent_dim), probs

    def forward(
        self,
        board_tensor: torch.Tensor,
        legal_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        embed = self.cnn_encoder(board_tensor)
        h = self.deter_proj(embed)

        post_logits = self.posterior_head(h)
        prior_logits = self.prior_head(h)
        z, post_probs = self._sample_discrete_latent(post_logits)
        _, prior_probs = self._sample_discrete_latent(prior_logits)

        state_feat = torch.cat([h, z], dim=-1)
        policy_logits = self.actor_trunk(state_feat)
        if legal_mask is not None:
            policy_logits = policy_logits.masked_fill(~legal_mask.bool(), -1e9)

        symlog_val = self.critic_symlog_head(state_feat)
        win_prob = torch.sigmoid(symexp(symlog_val))

        kl_div = (
            post_probs * (torch.log(post_probs + 1e-8) - torch.log(prior_probs + 1e-8))
        ).sum(dim=(-1, -2)).mean()

        return {
            "policy_logits": policy_logits,
            "win_prob": win_prob,
            "discrete_latent_z": z,
            "dyn_kl_loss": kl_div,
        }
