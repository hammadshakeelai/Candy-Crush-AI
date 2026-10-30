"""Bayesian Thompson Sampling Multi-Armed Bandit Root Solver Network.

Models each of the 144 swap actions as a Beta-Bernoulli / Normal-Inverse-Gamma
arm with epistemic uncertainty parameters (alpha, beta), enabling principled
Thompson Sampling root exploration across multi-retry level attempts.
"""

from typing import Dict, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class BayesianBanditRootSolverNet(nn.Module):
    """Predicts Beta(alpha, beta) posterior parameters over all 144 swap actions.

    Outputs:
        - policy_logits: (B, 144) expected log-odds / mean reward logits
        - win_prob: (B, 1) global board win probability from Beta mean
        - alpha: (B, 144) pseudo-success concentration parameter (>= 1.0)
        - beta: (B, 144) pseudo-failure concentration parameter (>= 1.0)
        - epistemic_uncertainty: (B, 144) posterior variance of each action arm
    """

    def __init__(
        self,
        in_channels: int = 20,
        hidden_dim: int = 128,
        num_actions: int = 144,
        prior_alpha: float = 1.0,
        prior_beta: float = 1.0,
    ) -> None:
        super().__init__()
        self.num_actions = num_actions
        self.prior_alpha = prior_alpha
        self.prior_beta = prior_beta

        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim // 2, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden_dim // 2),
            nn.SiLU(inplace=True),
        )

        flat_dim = (hidden_dim // 2) * 81
        self.trunk = nn.Sequential(
            nn.Linear(flat_dim, 256),
            nn.LayerNorm(256),
            nn.SiLU(inplace=True),
        )

        self.alpha_head = nn.Linear(256, num_actions)
        self.beta_head = nn.Linear(256, num_actions)
        self.logit_scale = nn.Parameter(torch.tensor(4.0))
        self.value_head = nn.Sequential(
            nn.Linear(256, 64),
            nn.SiLU(inplace=True),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def _compute_posterior(
        self, board_tensor: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        feat = self.encoder(board_tensor).flatten(1)
        latent = self.trunk(feat)
        alpha = self.prior_alpha + F.softplus(self.alpha_head(latent))
        beta = self.prior_beta + F.softplus(self.beta_head(latent))
        return latent, alpha, beta

    def forward(
        self,
        board_tensor: torch.Tensor,
        legal_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        latent, alpha, beta = self._compute_posterior(board_tensor)
        total = alpha + beta
        arm_mean = alpha / total
        epistemic_var = (alpha * beta) / (total.pow(2) * (total + 1.0))

        # Log-odds policy representation scaled by learned inverse temperature
        policy_logits = self.logit_scale * torch.log(arm_mean / (1.0 - arm_mean + 1e-6) + 1e-6)
        if legal_mask is not None:
            policy_logits = policy_logits.masked_fill(~legal_mask.bool(), -1e9)

        win_prob = self.value_head(latent)
        return {
            "policy_logits": policy_logits,
            "win_prob": win_prob,
            "alpha": alpha,
            "beta": beta,
            "epistemic_uncertainty": epistemic_var,
        }

    @torch.no_grad()
    def thompson_sample_action(
        self,
        board_tensor: torch.Tensor,
        legal_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Draws one stochastic Thompson sample from Beta(alpha, beta) per action arm."""
        _, alpha, beta = self._compute_posterior(board_tensor)
        dist = torch.distributions.Beta(alpha.clamp(min=1e-2), beta.clamp(min=1e-2))
        sampled_thetas = dist.sample()  # (B, 144)
        if legal_mask is not None:
            sampled_thetas = sampled_thetas.masked_fill(~legal_mask.bool(), -1.0)
        chosen_actions = sampled_thetas.argmax(dim=-1)
        return chosen_actions, sampled_thetas
