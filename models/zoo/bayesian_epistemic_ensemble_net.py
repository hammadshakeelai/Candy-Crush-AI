"""
AlphaCandy Model Zoo — Bayesian Evidential & Bootstrapped Epistemic Ensemble Network
(`EpistemicUncertaintyCandyNet`).

Based on:
  - Sensoy et al. (NeurIPS 2018), "Evidential Deep Learning to Quantify Classification Uncertainty"
  - Osband et al. (NeurIPS 2016 / 2018), "Deep Exploration via Bootstrapped DQN with Randomized Priors"
  - Kendall & Gal (NeurIPS 2017), "What Uncertainties Do We Need in Bayesian Deep Learning for Computer Vision?"

Decomposes predictive uncertainty on 9x9 Candy Crush boards into:
  1. Aleatoric Uncertainty (irreducible environment randomness caused by stochastic top-row candy
     dispenser drops and random special-candy bounce trajectories).
  2. Epistemic Uncertainty (reducible model ignorance on out-of-distribution blocker configurations,
     rare teleporter topologies, or unexplored late-stage board states).

Provides `ucb_exploration_action(state, legal_mask, beta=1.5)` using Upper Confidence Bound (UCB)
over epistemic uncertainty to systematically escape deterministic dead-end loops on level retries.
"""

import math
from typing import Dict, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class SpatialSEBlock(nn.Module):
    """Residual Conv2d Block with Squeeze-and-Excitation channel recalibration."""

    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

        mid = max(16, channels // reduction)
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, mid),
            nn.SiLU(),
            nn.Linear(mid, channels),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = F.silu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        scale = self.se(out).unsqueeze(-1).unsqueeze(-1)
        return F.silu(residual + out * scale)


class BootstrappedEnsembleHead(nn.Module):
    """
    Single Bootstrapped Policy-Value-Aleatoric Head with a frozen Randomized Prior Network
    (Osband et al., 2018) to guarantee non-collapsing epistemic variance in unvisited states.
    """

    def __init__(
        self,
        in_channels: int = 192,
        board_h: int = 9,
        board_w: int = 9,
        num_actions: int = 144,
        prior_scale: float = 0.15
    ):
        super().__init__()
        self.prior_scale = prior_scale
        flat_dim = 32 * board_h * board_w

        # Trainable policy & heteroscedastic aleatoric log-variance pathway
        self.spatial_reduce = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=1, bias=False),
            nn.BatchNorm2d(32),
            nn.SiLU()
        )
        self.policy_mlp = nn.Sequential(
            nn.Linear(flat_dim, 256),
            nn.SiLU(),
            nn.Linear(256, num_actions)
        )
        # Predicts per-action aleatoric log-variance (irreducible stochasticity of dispenser drops)
        self.aleatoric_logvar_mlp = nn.Sequential(
            nn.Linear(flat_dim, 128),
            nn.SiLU(),
            nn.Linear(128, num_actions)
        )

        # Trainable value head
        self.value_mlp = nn.Sequential(
            nn.Linear(in_channels, 128),
            nn.SiLU(),
            nn.Linear(128, 1)
        )

        # Frozen untrainable randomized prior network for Bayesian anchor in novel states
        self.prior_reduce = nn.Conv2d(in_channels, 16, kernel_size=1, bias=False)
        self.prior_policy = nn.Linear(16 * board_h * board_w, num_actions)
        self.prior_value = nn.Linear(in_channels, 1)
        for p in [
            *self.prior_reduce.parameters(),
            *self.prior_policy.parameters(),
            *self.prior_value.parameters()
        ]:
            p.requires_grad = False

    def forward(self, feat_2d: torch.Tensor, global_feat: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            logits: (B, 144) bootstrapped policy logits + randomized prior
            aleatoric_var: (B, 144) positive heteroscedastic aleatoric variance per action
            value_raw: (B, 1) unbounded value logit + randomized prior
        """
        red = self.spatial_reduce(feat_2d).flatten(1)
        with torch.no_grad():
            prior_red = F.silu(self.prior_reduce(feat_2d)).flatten(1)
            p_prior = self.prior_policy(prior_red)
            v_prior = self.prior_value(global_feat)

        logits = self.policy_mlp(red) + self.prior_scale * p_prior
        logvar = torch.clamp(self.aleatoric_logvar_mlp(red), min=-6.0, max=3.0)
        aleatoric_var = F.softplus(logvar) + 1e-4
        value_raw = self.value_mlp(global_feat) + self.prior_scale * v_prior
        return logits, aleatoric_var, value_raw


class DirichletEvidentialHead(nn.Module):
    """
    Dirichlet Evidential Deep Learning Head (Sensoy et al., NeurIPS 2018).
    Places a higher-order Dirichlet distribution Dir(p | alpha) over the 144-action categorical
    policy distribution, where alpha_a = evidence_a + 1.
    """

    def __init__(self, in_channels: int = 192, board_h: int = 9, board_w: int = 9, num_actions: int = 144):
        super().__init__()
        self.num_actions = num_actions
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 48, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(48),
            nn.SiLU()
        )
        self.evidence_proj = nn.Sequential(
            nn.Linear(48 * board_h * board_w, 256),
            nn.LayerNorm(256),
            nn.SiLU(),
            nn.Linear(256, num_actions)
        )

    def forward(
        self,
        feat_2d: torch.Tensor,
        legal_mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Computes Dirichlet parameters alpha, expected policy probabilities, vacuity,
        and per-action Dirichlet epistemic variance.
        """
        B = feat_2d.shape[0]
        h = self.conv(feat_2d).flatten(1)
        raw_evidence = self.evidence_proj(h)

        # Non-negative evidence e_a >= 0
        evidence = F.softplus(raw_evidence)
        if legal_mask is not None:
            mask_f = legal_mask.float()
            evidence = evidence * mask_f
            alpha = evidence + mask_f  # Concentration alpha_a = e_a + 1 on legal moves
            num_valid = mask_f.sum(dim=-1, keepdim=True).clamp(min=1.0)
        else:
            alpha = evidence + 1.0
            num_valid = torch.full((B, 1), float(self.num_actions), device=feat_2d.device, dtype=feat_2d.dtype)

        # Total Dirichlet strength S = sum_a alpha_a
        strength = alpha.sum(dim=-1, keepdim=True).clamp(min=1e-6)
        expected_prob = alpha / strength

        # Vacuity (epistemic lack of evidence): K_valid / S in (0, 1]
        vacuity = num_valid / strength

        # Analytic Dirichlet variance per action: alpha_a * (S - alpha_a) / (S^2 * (S + 1))
        dirichlet_var = (alpha * (strength - alpha)) / (strength.pow(2) * (strength + 1.0) + 1e-8)

        return {
            "evidence": evidence,
            "dirichlet_alpha": alpha,
            "dirichlet_strength": strength,
            "dirichlet_prob": expected_prob,
            "epistemic_vacuity": vacuity,
            "dirichlet_action_var": dirichlet_var,
        }


class EpistemicUncertaintyCandyNet(nn.Module):
    """
    Bayesian Epistemic Uncertainty & Multi-Head Bootstrapped Ensemble Network for Candy Crush.
    Combines:
      1. Dirichlet Evidential Deep Learning Head
      2. 5-Head Bootstrapped Policy-Value Ensemble with Randomized Prior Functions
      3. Explicit Aleatoric vs. Epistemic Uncertainty Decomposition
      4. `ucb_exploration_action` for escaping deterministic retry loops
    """

    def __init__(
        self,
        in_channels: int = 20,
        d_model: int = 192,
        num_res_blocks: int = 5,
        num_ensemble_heads: int = 5,
        num_actions: int = 144,
        board_h: int = 9,
        board_w: int = 9
    ):
        super().__init__()
        self.in_channels = in_channels
        self.d_model = d_model
        self.num_res_blocks = num_res_blocks
        self.num_ensemble_heads = num_ensemble_heads
        self.num_actions = num_actions
        self.board_h = board_h
        self.board_w = board_w

        # Shared spatial encoder stem
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, d_model, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(d_model),
            nn.SiLU()
        )
        self.res_tower = nn.ModuleList([
            SpatialSEBlock(d_model) for _ in range(num_res_blocks)
        ])

        # Global self-attention refinement over 81 cells
        self.spatial_attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=6, batch_first=True)
        self.attn_norm = nn.LayerNorm(d_model)

        # 1. Dirichlet Evidential Head
        self.evidential_head = DirichletEvidentialHead(
            in_channels=d_model,
            board_h=board_h,
            board_w=board_w,
            num_actions=num_actions
        )

        # 2. 5-Head Bootstrapped Ensemble
        self.ensemble_heads = nn.ModuleList([
            BootstrappedEnsembleHead(
                in_channels=d_model,
                board_h=board_h,
                board_w=board_w,
                num_actions=num_actions,
                prior_scale=0.15
            )
            for _ in range(num_ensemble_heads)
        ])

    def encode_board(self, board_tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encodes (B, 20, 9, 9) board into spatial feature map (B, D, 9, 9) and global vector (B, D)."""
        B, C, H, W = board_tensor.shape
        x = self.stem(board_tensor)
        for block in self.res_tower:
            x = block(x)

        tokens = x.flatten(2).transpose(1, 2)  # (B, 81, D)
        attn_out, _ = self.spatial_attn(tokens, tokens, tokens, need_weights=False)
        tokens = self.attn_norm(tokens + attn_out)

        feat_2d = tokens.transpose(1, 2).contiguous().view(B, self.d_model, H, W)
        global_feat = tokens.mean(dim=1)
        return feat_2d, global_feat

    def forward(
        self,
        board_tensor: torch.Tensor,
        legal_mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass returning canonical policy_logits (B, 144), value (B, 1), win_prob (B, 1),
        plus full uncertainty decomposition (aleatoric vs. epistemic).
        """
        feat_2d, global_feat = self.encode_board(board_tensor)

        # Evaluate 5 bootstrapped ensemble heads
        ens_logits_list = []
        ens_aleatoric_list = []
        ens_val_raw_list = []

        for head in self.ensemble_heads:
            logits_m, aleatoric_m, val_raw_m = head(feat_2d, global_feat)
            ens_logits_list.append(logits_m)
            ens_aleatoric_list.append(aleatoric_m)
            ens_val_raw_list.append(val_raw_m)

        # Shapes: (B, M, 144) and (B, M, 1)
        ens_logits = torch.stack(ens_logits_list, dim=1)
        ens_aleatoric_var = torch.stack(ens_aleatoric_list, dim=1)
        ens_val_raw = torch.stack(ens_val_raw_list, dim=1)

        # Evaluate Dirichlet Evidential Head
        evid_out = self.evidential_head(feat_2d, legal_mask=legal_mask)

        # Combine ensemble mean logits with log-evidence shift
        mean_ens_logits = ens_logits.mean(dim=1)
        evidential_logits = torch.log(evid_out["dirichlet_alpha"] + 1e-6)
        policy_logits = 0.7 * mean_ens_logits + 0.3 * evidential_logits

        if legal_mask is not None:
            mask_bool = legal_mask.bool()
            policy_logits = policy_logits.masked_fill(~mask_bool, -1e9)
            masked_ens_logits = ens_logits.masked_fill(~mask_bool.unsqueeze(1), -1e9)
        else:
            masked_ens_logits = ens_logits

        # Compute ensemble probability distributions p_m(a|s)
        ens_probs = F.softmax(masked_ens_logits, dim=-1)  # (B, M, 144)
        mean_prob = ens_probs.mean(dim=1)                 # (B, 144)

        # 1. Aleatoric Uncertainty (irreducible environment drop randomness)
        # Action-wise heteroscedastic aleatoric std: (B, 144)
        action_aleatoric_uncertainty = torch.sqrt(ens_aleatoric_var.mean(dim=1) + 1e-8)
        # Global expected entropy E_m[H(p_m)] across ensemble heads: (B, 1)
        head_entropies = -(ens_probs * torch.log(ens_probs.clamp(min=1e-8))).sum(dim=-1)  # (B, M)
        aleatoric_entropy = head_entropies.mean(dim=1, keepdim=True)                      # (B, 1)

        # 2. Epistemic Uncertainty (reducible model ignorance on novel topologies)
        # Action-wise epistemic std combining ensemble disagreement + Dirichlet variance: (B, 144)
        ens_logit_var = ens_logits.var(dim=1, unbiased=False)
        action_epistemic_uncertainty = torch.sqrt(
            ens_logit_var + 10.0 * evid_out["dirichlet_action_var"] + 1e-8
        )
        if legal_mask is not None:
            action_epistemic_uncertainty = action_epistemic_uncertainty * legal_mask.float()
            action_aleatoric_uncertainty = action_aleatoric_uncertainty * legal_mask.float()

        # Global Mutual Information I(y; theta | s) = H(E_m[p_m]) - E_m[H(p_m)] + vacuity: (B, 1)
        predictive_entropy = -(mean_prob * torch.log(mean_prob.clamp(min=1e-8))).sum(dim=-1, keepdim=True)
        mutual_info = (predictive_entropy - aleatoric_entropy).clamp(min=0.0)
        epistemic_uncertainty = mutual_info + evid_out["epistemic_vacuity"]

        # Value & Win Probability aggregation + value epistemic std
        ens_values = torch.tanh(ens_val_raw)
        ens_win_probs = torch.sigmoid(ens_val_raw)
        value = ens_values.mean(dim=1)
        win_prob = ens_win_probs.mean(dim=1)
        value_epistemic_std = ens_values.std(dim=1, unbiased=False)

        return {
            "policy_logits": policy_logits,
            "value": value,
            "win_prob": win_prob,
            "aleatoric_uncertainty": action_aleatoric_uncertainty,
            "epistemic_uncertainty": action_epistemic_uncertainty,
            "global_aleatoric_entropy": aleatoric_entropy,
            "global_epistemic_uncertainty": epistemic_uncertainty,
            "epistemic_vacuity": evid_out["epistemic_vacuity"],
            "dirichlet_alpha": evid_out["dirichlet_alpha"],
            "ensemble_policy_logits": ens_logits,
            "ensemble_values": ens_values,
            "value_epistemic_std": value_epistemic_std,
        }

    @torch.no_grad()
    def ucb_exploration_action(
        self,
        state: torch.Tensor,
        legal_mask: Optional[torch.Tensor] = None,
        beta: float = 1.5,
        aleatoric_penalty: float = 0.25
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Selects an action using Upper Confidence Bound (UCB) over epistemic uncertainty:
            Score(s, a) = Q_norm(s, a) + beta * sigma_epistemic(s, a) - aleatoric_penalty * sigma_aleatoric(s, a)
        Allows the agent to escape deterministic failure loops on retry attempts while avoiding
        purely chaotic high-aleatoric traps.

        Args:
            state: (B, 20, 9, 9) or (20, 9, 9) canonical board tensor
            legal_mask: Optional (B, 144) or (144,) boolean/float mask of legal moves
            beta: Exploration bonus weight on epistemic uncertainty (default 1.5)
            aleatoric_penalty: Penalty weight on irreducible aleatoric noise (default 0.25)

        Returns:
            best_actions: (B,) tensor of selected action indices in [0, 143]
            diagnostics: Dictionary with `ucb_scores`, `epistemic_bonus`, and `aleatoric_uncertainty`
        """
        single_input = (state.dim() == 3)
        if single_input:
            state = state.unsqueeze(0)
            if legal_mask is not None and legal_mask.dim() == 1:
                legal_mask = legal_mask.unsqueeze(0)

        out = self.forward(state, legal_mask=legal_mask)
        probs = F.softmax(out["policy_logits"], dim=-1)
        epistemic_std = out["epistemic_uncertainty"]
        aleatoric_std = out["aleatoric_uncertainty"]

        # Normalize epistemic bonus to match probability scale
        epistemic_norm = epistemic_std / (epistemic_std.max(dim=-1, keepdim=True).values + 1e-6)
        aleatoric_norm = aleatoric_std / (aleatoric_std.max(dim=-1, keepdim=True).values + 1e-6)

        ucb_scores = probs + beta * epistemic_norm - aleatoric_penalty * aleatoric_norm
        if legal_mask is not None:
            ucb_scores = ucb_scores.masked_fill(~legal_mask.bool(), -1e9)

        best_actions = torch.argmax(ucb_scores, dim=-1)
        diagnostics = {
            "ucb_scores": ucb_scores,
            "epistemic_bonus": beta * epistemic_norm,
            "aleatoric_uncertainty": aleatoric_std,
            "global_epistemic_uncertainty": out["global_epistemic_uncertainty"],
        }
        return best_actions, diagnostics

    def evidential_loss(
        self,
        dirichlet_alpha: torch.Tensor,
        target_actions: torch.Tensor,
        kl_weight: float = 0.05
    ) -> torch.Tensor:
        """
        Type-II Maximum Likelihood Dirichlet Evidential Loss + KL Divergence regularization
        penalizing false evidence on non-target moves (Sensoy et al., Eq. 5 & Eq. 7).
        """
        target_one_hot = F.one_hot(target_actions, num_classes=self.num_actions).float()
        strength = dirichlet_alpha.sum(dim=-1, keepdim=True)

        # Expected sum-of-squares Bayes risk under Dirichlet prior
        prob = dirichlet_alpha / strength
        err = (target_one_hot - prob).pow(2)
        var = prob * (1.0 - prob) / (strength + 1.0)
        risk_loss = (err + var).sum(dim=-1).mean()

        # Remove non-misleading evidence on ground-truth target class before KL penalty
        alpha_tilde = target_one_hot + (1.0 - target_one_hot) * dirichlet_alpha
        ones = torch.ones_like(alpha_tilde)
        sum_alpha_tilde = alpha_tilde.sum(dim=-1, keepdim=True)

        kl_div = (
            torch.lgamma(sum_alpha_tilde)
            - torch.lgamma(torch.tensor(float(self.num_actions), device=dirichlet_alpha.device))
            - torch.lgamma(alpha_tilde).sum(dim=-1, keepdim=True)
            + ((alpha_tilde - ones) * (torch.digamma(alpha_tilde) - torch.digamma(sum_alpha_tilde))).sum(dim=-1, keepdim=True)
        ).mean()

        return risk_loss + kl_weight * kl_div

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def create_epistemic_uncertainty_candy_net(device: str = "cpu") -> EpistemicUncertaintyCandyNet:
    model = EpistemicUncertaintyCandyNet(
        in_channels=20,
        d_model=192,
        num_res_blocks=5,
        num_ensemble_heads=5,
        num_actions=144
    )
    return model.to(device)


if __name__ == "__main__":
    torch.manual_seed(42)
    net = create_epistemic_uncertainty_candy_net("cpu")
    params = net.count_parameters()
    print(f"[EpistemicUncertaintyCandyNet] Trainable parameters: {params:,} ({params / 1e6:.2f}M)")

    dummy_board = torch.randn(2, 20, 9, 9)
    dummy_mask = torch.ones(2, 144, dtype=torch.bool)
    dummy_mask[:, 120:] = False

    out = net(dummy_board, legal_mask=dummy_mask)
    actions, ucb_info = net.ucb_exploration_action(dummy_board, legal_mask=dummy_mask, beta=1.5)
    loss = net.evidential_loss(out["dirichlet_alpha"], torch.tensor([5, 12]))

    print(f"  policy_logits:          {out['policy_logits'].shape}")
    print(f"  value:                  {out['value'].shape}")
    print(f"  win_prob:               {out['win_prob'].shape}")
    print(f"  aleatoric_uncertainty:  {out['aleatoric_uncertainty'].shape}")
    print(f"  epistemic_uncertainty:  {out['epistemic_uncertainty'].shape}")
    print(f"  ensemble_policy_logits: {out['ensemble_policy_logits'].shape}")
    print(f"  ucb_exploration_action: {actions.tolist()} (shapes: {ucb_info['ucb_scores'].shape})")
    print(f"  evidential_loss:        {loss.item():.4f}")
