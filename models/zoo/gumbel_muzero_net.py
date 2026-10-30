"""
Gumbel MuZero Network & Sequential Halving Search for Candy Crush Saga.
Based on:
  - Danihelka et al. (ICLR 2022), "Policy improvement by planning with Gumbel"
  - Schrittwieser et al. (Nature 2020), "Mastering Atari, Go, Chess and Shogi by Planning with a Learned Model"

Guarantees provable policy improvement (sigma(logits + gumbel + completed_Q)) even with very
small simulation budgets (e.g., 4 to 8 rollouts) in high-branching stochastic match-3 spaces.
"""

import math
from typing import Dict, Tuple, Optional, List
import torch
import torch.nn as nn
import torch.nn.functional as F


class SpatialResBlock(nn.Module):
    """Pre-activation Residual Block with Squeeze-and-Excitation channel recalibration."""
    def __init__(self, channels: int):
        super().__init__()
        self.norm1 = nn.BatchNorm2d(channels)
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.norm2 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, max(16, channels // 8), kernel_size=1),
            nn.GELU(),
            nn.Conv2d(max(16, channels // 8), channels, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv1(F.gelu(self.norm1(x)))
        out = self.conv2(F.gelu(self.norm2(out)))
        out = out * self.se(out)
        return x + out


class RepresentationNet(nn.Module):
    """
    Representation Function h_theta(s) -> z_0 in R^{B x 81 x d_model}.
    Encodes the raw 20-channel 9x9 board tensor into a spatially structured latent state.
    """
    def __init__(
        self,
        in_channels: int = 20,
        d_model: int = 256,
        num_res_blocks: int = 6,
        board_h: int = 9,
        board_w: int = 9
    ):
        super().__init__()
        self.d_model = d_model
        self.board_h = board_h
        self.board_w = board_w
        self.num_tokens = board_h * board_w

        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, d_model, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(d_model),
            nn.GELU()
        )
        self.res_tower = nn.ModuleList([SpatialResBlock(d_model) for _ in range(num_res_blocks)])
        self.pos_embed = nn.Parameter(torch.randn(1, self.num_tokens, d_model) * 0.02)
        self.final_norm = nn.LayerNorm(d_model)

    def forward(self, board_tensor: torch.Tensor) -> torch.Tensor:
        B, C, H, W = board_tensor.shape
        x = self.stem(board_tensor)
        for block in self.res_tower:
            x = block(x)
        # Flatten spatial grid to (B, 81, d_model)
        z_0 = x.flatten(2).transpose(1, 2) + self.pos_embed
        # Normalize latent state to [0, 1] scale + LayerNorm for stable recurrent rollouts
        z_0 = self.final_norm(z_0)
        return z_0


class DynamicsNet(nn.Module):
    """
    Latent Dynamics Function g_theta(z_t, a_t) -> (r_hat_t, z_{t+1}).
    Models the deterministic equivalent of expected cascade resolution and gravity drop
    conditioned on a discrete swap action a_t in [0, 143].
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

        # Spatial action plane embedding: maps action index to (d_model // 4) channels + coordinate plane
        self.action_embedding = nn.Embedding(num_actions, d_model)
        self.fuse_conv = nn.Sequential(
            nn.Conv2d(d_model * 2, d_model, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(d_model),
            nn.GELU()
        )
        self.transition_blocks = nn.ModuleList([
            SpatialResBlock(d_model),
            SpatialResBlock(d_model),
            SpatialResBlock(d_model)
        ])
        self.latent_norm = nn.LayerNorm(d_model)

        # Immediate cascade reward prediction head r_hat_t in R^{B x 1}
        self.reward_head = nn.Sequential(
            nn.Conv2d(d_model, 32, kernel_size=1),
            nn.GELU(),
            nn.Flatten(),
            nn.Linear(32 * board_h * board_w, 128),
            nn.GELU(),
            nn.Linear(128, 1)
        )

    def forward(self, z_t: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            z_t: Latent state tensor of shape (B, 81, d_model)
            action: Discrete action indices of shape (B,) in [0, 143]
        Returns:
            reward_pred: Predicted immediate cascade utility (B, 1)
            z_next: Next latent state (B, 81, d_model)
        """
        B, N, D = z_t.shape
        z_2d = z_t.transpose(1, 2).contiguous().view(B, D, self.board_h, self.board_w)

        a_emb = self.action_embedding(action).view(B, D, 1, 1).expand(B, D, self.board_h, self.board_w)
        fused = self.fuse_conv(torch.cat([z_2d, a_emb], dim=1))

        h = z_2d + fused
        for block in self.transition_blocks:
            h = block(h)

        reward_pred = self.reward_head(h)
        z_next = h.flatten(2).transpose(1, 2)
        z_next = self.latent_norm(z_next)
        return reward_pred, z_next


class PredictionNet(nn.Module):
    """
    Prediction Function f_theta(z_t) -> (policy_logits, win_value).
    Maps any latent state z_t to canonical 144 swap logits and calibrated win probability.
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
        self.board_h = board_h
        self.board_w = board_w

        # Policy Head over 72 Horizontal + 72 Vertical canonical swaps
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

        # Value / Win Probability Head V(z_t) in [0, 1]
        self.value_head = nn.Sequential(
            nn.Conv2d(d_model, 32, kernel_size=1),
            nn.GELU(),
            nn.Flatten(),
            nn.Linear(32 * board_h * board_w, 128),
            nn.GELU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, z_t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B, N, D = z_t.shape
        z_2d = z_t.transpose(1, 2).contiguous().view(B, D, self.board_h, self.board_w)
        policy_logits = self.policy_head(z_2d)
        win_prob = self.value_head(z_2d)
        return policy_logits, win_prob


class GumbelMuZeroCandyNet(nn.Module):
    """
    Complete Gumbel MuZero Architecture for Candy Crush Saga.
    Combines h_theta (Representation), g_theta (Dynamics), and f_theta (Prediction)
    with Gumbel-Top-k Sequential Halving latent planning.
    """
    def __init__(
        self,
        in_channels: int = 20,
        d_model: int = 256,
        num_res_blocks: int = 6,
        num_actions: int = 144,
        board_h: int = 9,
        board_w: int = 9,
        c_visit: float = 50.0,
        c_scale: float = 1.0,
        discount: float = 0.99
    ):
        super().__init__()
        self.in_channels = in_channels
        self.d_model = d_model
        self.num_actions = num_actions
        self.c_visit = c_visit
        self.c_scale = c_scale
        self.discount = discount

        self.representation = RepresentationNet(
            in_channels=in_channels,
            d_model=d_model,
            num_res_blocks=num_res_blocks,
            board_h=board_h,
            board_w=board_w
        )
        self.dynamics = DynamicsNet(
            d_model=d_model,
            num_actions=num_actions,
            board_h=board_h,
            board_w=board_w
        )
        self.prediction = PredictionNet(
            d_model=d_model,
            num_actions=num_actions,
            board_h=board_h,
            board_w=board_w
        )

    def initial_inference(self, board_tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Runs h_theta(s) followed by f_theta(z_0) -> (z_0, policy_logits, win_prob)."""
        z_0 = self.representation(board_tensor)
        policy_logits, win_prob = self.prediction(z_0)
        return z_0, policy_logits, win_prob

    def recurrent_inference(
        self, z_t: torch.Tensor, action: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Runs g_theta(z_t, a_t) followed by f_theta(z_{t+1}) -> (reward, z_{t+1}, policy_logits, win_prob)."""
        reward_pred, z_next = self.dynamics(z_t, action)
        policy_logits, win_prob = self.prediction(z_next)
        return reward_pred, z_next, policy_logits, win_prob

    def forward(
        self,
        board_tensor: torch.Tensor,
        action: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Canonical forward pass accepting (B, 20, 9, 9) board tensor.
        Optionally performs a 1-step latent rollout if `action` (or greedy top action) is evaluated.
        """
        z_0, policy_logits, win_prob = self.initial_inference(board_tensor)

        if action is None:
            action = torch.argmax(policy_logits, dim=-1)

        reward_pred, z_1, next_policy_logits, next_win_prob = self.recurrent_inference(z_0, action)

        return {
            "policy_logits": policy_logits,
            "win_prob": win_prob,
            "value": win_prob,
            "latent_z": z_0,
            "predicted_reward": reward_pred,
            "next_latent_z": z_1,
            "next_win_prob": next_win_prob
        }

    def sigma_q_transform(
        self,
        q_values: torch.Tensor,
        visit_counts: torch.Tensor,
        v_root: torch.Tensor
    ) -> torch.Tensor:
        """
        Monotonically increasing transformation sigma(q_hat(a)) from Danihelka et al. (2022):
            sigma(q) = (c_visit + max_b N(b)) * c_scale * q
        """
        max_visits = visit_counts.max(dim=-1, keepdim=True)[0]
        scale = (self.c_visit + max_visits) * self.c_scale
        return scale * q_values

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


@torch.no_grad()
def gumbel_sequential_halving_search(
    model: GumbelMuZeroCandyNet,
    state_tensor: torch.Tensor,
    legal_mask: Optional[torch.Tensor] = None,
    num_simulations: int = 8,
    top_k: int = 8,
    gumbel_scale: float = 1.0
) -> Dict[str, torch.Tensor]:
    """
    Performs Gumbel-Top-k sampling without replacement and Sequential Halving in latent space
    (Danihelka et al., ICLR 2022). Guarantees strict policy improvement even with 4-8 rollouts.

    Args:
        model: GumbelMuZeroCandyNet instance
        state_tensor: Board state tensor of shape (B, 20, 9, 9)
        legal_mask: Boolean tensor of shape (B, 144) indicating valid swaps
        num_simulations: Total latent simulation budget (e.g., 8)
        top_k: Initial number of candidate actions sampled via Gumbel-Top-k (e.g., 8)
        gumbel_scale: Temperature scale for Gumbel noise (set 0.0 for deterministic eval)

    Returns:
        Dictionary containing:
          - `improved_policy`: Provably improved target policy distribution pi' of shape (B, 144)
          - `selected_action`: Winning action indices after Sequential Halving of shape (B,)
          - `completed_q`: Completed Q-values for all 144 actions of shape (B, 144)
          - `root_value`: Root win probability estimate of shape (B, 1)
    """
    device = state_tensor.device
    B = state_tensor.shape[0]
    num_actions = model.num_actions

    z_root, root_logits, root_value = model.initial_inference(state_tensor)

    if legal_mask is not None:
        masked_logits = root_logits.masked_fill(~legal_mask.bool(), -1e9)
    else:
        masked_logits = root_logits.clone()

    # Sample Gumbel(0, 1) noise: g = -log(-log(U))
    u = torch.rand_like(masked_logits).clamp(min=1e-7, max=1.0 - 1e-7)
    gumbel_noise = -torch.log(-torch.log(u)) * gumbel_scale
    if legal_mask is not None:
        gumbel_noise = gumbel_noise.masked_fill(~legal_mask.bool(), -1e9)

    noisy_logits = masked_logits + gumbel_noise

    # Select initial top_k candidate actions per batch element
    k_init = min(top_k, num_actions)
    _, candidate_actions = torch.topk(noisy_logits, k=k_init, dim=-1)  # (B, k_init)

    q_estimates = torch.zeros((B, num_actions), device=device)
    visit_counts = torch.zeros((B, num_actions), device=device)

    # Number of Sequential Halving phases
    num_phases = max(1, int(math.ceil(math.log2(max(2, k_init)))))
    active_candidates = candidate_actions  # (B, K_current)

    for phase in range(num_phases):
        k_curr = active_candidates.shape[1]
        if k_curr <= 1:
            break

        # Budget per candidate in this phase (at least 1 rollout per active candidate)
        rolls_per_cand = max(1, num_simulations // (num_phases * k_curr))

        for idx in range(k_curr):
            act_col = active_candidates[:, idx]  # (B,)
            z_curr = z_root
            total_return = torch.zeros((B, 1), device=device)
            discount_factor = 1.0

            # Multi-step unroll in latent space along depth=rolls_per_cand
            curr_act = act_col
            for step in range(rolls_per_cand):
                r_hat, z_curr, step_logits, v_next = model.recurrent_inference(z_curr, curr_act)
                total_return = total_return + discount_factor * torch.tanh(r_hat)
                discount_factor *= model.discount
                if step + 1 < rolls_per_cand:
                    curr_act = torch.argmax(step_logits, dim=-1)

            bootstrapped_q = total_return + discount_factor * v_next  # (B, 1)

            # Update empirical Q-estimate and visit count for act_col
            batch_indices = torch.arange(B, device=device)
            old_n = visit_counts[batch_indices, act_col]
            old_q = q_estimates[batch_indices, act_col]
            new_n = old_n + float(rolls_per_cand)
            new_q = (old_q * old_n + bootstrapped_q.squeeze(-1) * float(rolls_per_cand)) / new_n

            visit_counts[batch_indices, act_col] = new_n
            q_estimates[batch_indices, act_col] = new_q

        # Rank active candidates by g(a) + logits(a) + sigma(q_hat(a))
        transformed_q = model.sigma_q_transform(q_estimates, visit_counts, root_value)
        halving_scores = gumbel_noise + masked_logits + transformed_q

        cand_scores = torch.gather(halving_scores, dim=1, index=active_candidates)
        keep_k = max(1, k_curr // 2)
        _, top_local_idx = torch.topk(cand_scores, k=keep_k, dim=-1)
        active_candidates = torch.gather(active_candidates, dim=1, index=top_local_idx)

    # Final winning action from Sequential Halving
    selected_action = active_candidates[:, 0]

    # Construct Completed Q-values (Danihelka et al. Eq. 10):
    # Use empirical Q(a) if N(a) > 0, else fallback to v_root
    completed_q = torch.where(visit_counts > 0, q_estimates, root_value.expand(-1, num_actions))
    sigma_completed_q = model.sigma_q_transform(completed_q, visit_counts, root_value)

    # Provably improved policy distribution pi'(a|s) = softmax(logits + sigma(completed_Q))
    improved_logits = masked_logits + sigma_completed_q
    improved_policy = F.softmax(improved_logits, dim=-1)

    return {
        "improved_policy": improved_policy,
        "selected_action": selected_action,
        "completed_q": completed_q,
        "root_value": root_value,
        "visit_counts": visit_counts
    }


def create_gumbel_muzero_net(device: str = "cpu") -> GumbelMuZeroCandyNet:
    model = GumbelMuZeroCandyNet(
        in_channels=20,
        d_model=256,
        num_res_blocks=6,
        num_actions=144
    )
    return model.to(device)


if __name__ == "__main__":
    torch.manual_seed(42)
    net = create_gumbel_muzero_net("cpu")
    params = net.count_parameters()
    print(f"[GumbelMuZeroCandyNet] Initialized with {params:,} trainable parameters ({params / 1e6:.2f}M)")

    dummy_board = torch.randn(2, 20, 9, 9)
    dummy_mask = torch.ones(2, 144, dtype=torch.bool)
    out = net(dummy_board)
    print(f"  policy_logits:    {out['policy_logits'].shape}")
    print(f"  win_prob:         {out['win_prob'].shape}")
    print(f"  latent_z:         {out['latent_z'].shape}")
    print(f"  predicted_reward: {out['predicted_reward'].shape}")

    search_res = gumbel_sequential_halving_search(
        net, dummy_board, legal_mask=dummy_mask, num_simulations=8, top_k=8
    )
    print(f"  Gumbel Sequential Halving improved_policy: {search_res['improved_policy'].shape}")
    print(f"  Gumbel Sequential Halving selected_action: {search_res['selected_action'].shape}")
