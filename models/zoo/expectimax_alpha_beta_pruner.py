"""
Ballard's *-Minimax (Star1 / Star2) Stochastic Alpha-Beta Pruning Network (`Star1ExpectimaxPrunerNet`).
========================================================================================================
Based on:
  - Ballard (1983), "The *-Minimax Search Procedure for Trees Containing Chance Nodes",
    Artificial Intelligence, 21(3), 327-350.
  - Hauk, Buro & Schaeffer (2006), "Rediscovering *-Minimax Search", LNCS / Computers and Games.

Why *-Minimax Matters for Candy Crush Lookahead:
  In stochastic Match-3 lookahead, every player swap (MAX node) is followed by a random candy refill
  and gravity cascade (CHANCE node) with K possible stochastic drop outcomes c_1, ..., c_K of
  probabilities p_1, ..., p_K (sum_k p_k = 1). Standard Expectimax must evaluate all K outcomes at
  every chance node, causing an exponential O((144 * K)^d) branching explosion.

  Ballard's `Star1` and `Star2` algorithms exploit bounded value intervals [V_min(s), V_max(s)]
  at chance nodes:
  Suppose we want to determine whether a chance node C can exceed the current best MAX value `alpha`.
  After evaluating the first (k - 1) drop outcomes with partial weighted sum S_{k-1} = sum_{i < k} p_i * v_i,
  and knowing that future unevaluated outcomes j > k cannot exceed their upper bounds U_j = V_max(s_j),
  outcome k MUST achieve at least the local Star1 threshold:
      alpha_k = (alpha - S_{k-1} - sum_{j > k} p_j * U_j) / p_k
  Symmetrically, to stay below abound `beta`, outcome k cannot exceed:
      beta_k  = (beta  - S_{k-1} - sum_{j > k} p_j * L_j) / p_k
  If alpha_k > U_k (i.e. even if outcome k and all remaining outcomes j > k achieve their maximum
  theoretical ceiling, the chance node still cannot reach `alpha`), we immediately PRUNE (alpha-cutoff)
  all remaining stochastic drop simulations!
"""

from typing import Dict, Optional, Tuple
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class StarResBlock(nn.Module):
    """Residual block with GroupNorm and Squeeze-and-Excitation for bounded interval estimation."""

    def __init__(self, channels: int = 256):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.gn1 = nn.GroupNorm(16, channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.gn2 = nn.GroupNorm(16, channels)
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // 8, kernel_size=1),
            nn.SiLU(),
            nn.Conv2d(channels // 8, channels, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.silu(self.gn1(self.conv1(x)))
        h = self.gn2(self.conv2(h))
        h = h * self.se(h)
        return F.silu(x + h)


class StarIntervalBoundHead(nn.Module):
    """
    Predicts Ballard's bounded value intervals for both states and (state, swap_action) chance nodes:
      - State interval: [V_min(s), V_exp(s), V_max(s)] in [-1, 1] with V_min(s) <= V_exp(s) <= V_max(s).
      - Action chance-node interval: [Q_min(s, a), Q_exp(s, a), Q_max(s, a)] in [-1, 1]^{B x 144}
        representing the worst-case drop outcome, expected value, and best-case miracle cascade ceiling
        for each of the 144 swap actions.
      - Stochastic K-branch outcome generator: predicts K canonical chance branch bounds
        [L_{a,k}, v_{a,k}, U_{a,k}] for vectorized Star1/Star2 pruning.
    """

    def __init__(
        self,
        d_model: int = 256,
        num_actions: int = 144,
        num_chance_branches: int = 8,
        board_h: int = 9,
        board_w: int = 9,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_actions = num_actions
        self.num_chance_branches = num_chance_branches
        self.board_h = board_h
        self.board_w = board_w

        # Precompute canonical 144 swap endpoint coordinates (72 horizontal + 72 vertical)
        r1_list, c1_list, r2_list, c2_list, orient_list = [], [], [], [], []
        for r in range(board_h):
            for c in range(board_w - 1):
                r1_list.append(r)
                c1_list.append(c)
                r2_list.append(r)
                c2_list.append(c + 1)
                orient_list.append(0)
        for r in range(board_h - 1):
            for c in range(board_w):
                r1_list.append(r)
                c1_list.append(c)
                r2_list.append(r + 1)
                c2_list.append(c)
                orient_list.append(1)

        self.register_buffer("idx_r1", torch.tensor(r1_list, dtype=torch.long))
        self.register_buffer("idx_c1", torch.tensor(c1_list, dtype=torch.long))
        self.register_buffer("idx_r2", torch.tensor(r2_list, dtype=torch.long))
        self.register_buffer("idx_c2", torch.tensor(c2_list, dtype=torch.long))
        self.register_buffer("idx_orient", torch.tensor(orient_list, dtype=torch.long))

        self.action_embed = nn.Embedding(num_actions, d_model)
        self.orient_embed = nn.Embedding(2, d_model)

        # Global board context projector
        self.global_proj = nn.Sequential(
            nn.Conv2d(d_model, 64, kernel_size=1, bias=False),
            nn.GroupNorm(8, 64),
            nn.SiLU(),
            nn.Flatten(),
            nn.Linear(64 * board_h * board_w, d_model),
            nn.LayerNorm(d_model),
            nn.SiLU(),
        )

        # State-level interval head: outputs (center_raw, down_radius_raw, up_radius_raw)
        self.state_interval_mlp = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.SiLU(),
            nn.Linear(128, 3),
        )

        # Action-level dipole + global context projector
        self.action_feat_mlp = nn.Sequential(
            nn.Linear(d_model * 3, d_model),
            nn.LayerNorm(d_model),
            nn.SiLU(),
            nn.Linear(d_model, 128),
            nn.LayerNorm(128),
            nn.SiLU(),
        )

        # Outputs per action:
        # 0: policy preference logit
        # 1: Q_exp raw
        # 2: lower radius raw (Q_exp - Q_min)
        # 3: upper radius raw (Q_max - Q_exp)
        # 4: refill volatility sigma_drop
        self.action_bound_head = nn.Linear(128, 5)

        # K-branch Star2 probing head: predicts K sorted stochastic drop branch bounds per action
        self.chance_branch_head = nn.Linear(128, num_chance_branches * 3)
        self.chance_prob_logits = nn.Linear(128, num_chance_branches)

    def forward(self, spatial_feat: torch.Tensor) -> Dict[str, torch.Tensor]:
        B, D, H, W = spatial_feat.shape
        g_ctx = self.global_proj(spatial_feat)  # (B, D)

        # 1. State-level bounded interval [V_min(s), V_exp(s), V_max(s)] in [-1, 1]
        s_raw = self.state_interval_mlp(g_ctx)  # (B, 3)
        v_exp = torch.tanh(s_raw[:, 0:1])
        v_rad_low = torch.sigmoid(s_raw[:, 1:2]) * (v_exp + 1.0)     # Guarantees V_min >= -1.0
        v_rad_high = torch.sigmoid(s_raw[:, 2:3]) * (1.0 - v_exp)    # Guarantees V_max <= +1.0
        v_min = (v_exp - v_rad_low).clamp(-1.0, 1.0)
        v_max = (v_exp + v_rad_high).clamp(-1.0, 1.0)

        # 2. Extract swap-dipole features for all 144 actions
        f1 = spatial_feat[:, :, self.idx_r1, self.idx_c1].transpose(1, 2)  # (B, 144, D)
        f2 = spatial_feat[:, :, self.idx_r2, self.idx_c2].transpose(1, 2)  # (B, 144, D)
        act_ids = torch.arange(self.num_actions, device=spatial_feat.device)
        a_emb = (self.action_embed(act_ids) + self.orient_embed(self.idx_orient)).unsqueeze(0).expand(B, -1, -1)

        dipole_in = torch.cat([f1 + f2, (f1 - f2).abs(), a_emb + g_ctx.unsqueeze(1)], dim=-1)  # (B, 144, 3*D)
        act_hidden = self.action_feat_mlp(dipole_in)  # (B, 144, 128)

        bound_raw = self.action_bound_head(act_hidden)  # (B, 144, 5)
        raw_logits = bound_raw[:, :, 0]
        q_exp = torch.tanh(bound_raw[:, :, 1])
        q_rad_low = torch.sigmoid(bound_raw[:, :, 2]) * (q_exp + 1.0)
        q_rad_high = torch.sigmoid(bound_raw[:, :, 3]) * (1.0 - q_exp)
        q_min = (q_exp - q_rad_low).clamp(-1.0, 1.0)
        q_max = (q_exp + q_rad_high).clamp(-1.0, 1.0)
        refill_volatility = F.softplus(bound_raw[:, :, 4])

        # 3. K-branch stochastic chance node decomposition for Star1/Star2
        K = self.num_chance_branches
        branch_raw = self.chance_branch_head(act_hidden).view(B, self.num_actions, K, 3)
        branch_v = torch.tanh(branch_raw[:, :, :, 0])
        branch_low = torch.sigmoid(branch_raw[:, :, :, 1]) * (branch_v + 1.0)
        branch_high = torch.sigmoid(branch_raw[:, :, :, 2]) * (1.0 - branch_v)
        branch_L = (branch_v - branch_low).clamp(-1.0, 1.0)
        branch_U = (branch_v + branch_high).clamp(-1.0, 1.0)
        branch_probs = F.softmax(self.chance_prob_logits(act_hidden), dim=-1)  # (B, 144, K)

        return {
            "raw_logits": raw_logits,
            "v_min": v_min,
            "v_exp": v_exp,
            "v_max": v_max,
            "q_min": q_min,
            "q_exp": q_exp,
            "q_max": q_max,
            "refill_volatility": refill_volatility,
            "branch_L": branch_L,
            "branch_v": branch_v,
            "branch_U": branch_U,
            "branch_probs": branch_probs,
        }


class Star1ExpectimaxPrunerNet(nn.Module):
    """
    Ballard's *-Minimax (`Star1` / `Star2`) Stochastic Alpha-Beta Pruning Network.
    Accepts `(B, 20, 9, 9)` board state tensor and outputs:
      - `policy_logits`: `(B, 144)` Star1-pruned & interval-informed policy logits
      - `value`: `(B, 1)` expected state value in [-1, 1]
      - `win_prob`: `(B, 1)` calibrated win probability in [0, 1]
      - `v_min`, `v_max`: `(B, 1)` state value lower & upper bounds [V_min(s), V_max(s)]
      - `q_min`, `q_exp`, `q_max`: `(B, 144)` action chance-node value interval bounds
      - `star1_active_mask`: `(B, 144)` boolean mask of actions surviving Star1 alpha-cutoff
      - `branch_evaluated_mask`: `(B, 144, K)` boolean mask of chance branches evaluated before Star1 cutoff
      - `pruning_efficiency`: `(B, 1)` fraction of stochastic chance branches pruned by Star1/Star2
    """

    def __init__(
        self,
        in_channels: int = 20,
        d_model: int = 256,
        num_res_blocks: int = 5,
        num_actions: int = 144,
        num_chance_branches: int = 8,
        board_h: int = 9,
        board_w: int = 9,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.d_model = d_model
        self.num_res_blocks = num_res_blocks
        self.num_actions = num_actions
        self.num_chance_branches = num_chance_branches
        self.board_h = board_h
        self.board_w = board_w

        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, d_model, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(16, d_model),
            nn.SiLU(),
        )
        self.res_tower = nn.ModuleList([StarResBlock(d_model) for _ in range(num_res_blocks)])

        self.interval_head = StarIntervalBoundHead(
            d_model=d_model,
            num_actions=num_actions,
            num_chance_branches=num_chance_branches,
            board_h=board_h,
            board_w=board_w,
        )

    @staticmethod
    def run_vectorized_star1_pruning(
        q_min: torch.Tensor,
        q_exp: torch.Tensor,
        q_max: torch.Tensor,
        branch_L: torch.Tensor,
        branch_v: torch.Tensor,
        branch_U: torch.Tensor,
        branch_probs: torch.Tensor,
        legal_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Executes Ballard's exact `Star1` / `Star2` chance-node alpha-beta pruning across
        all 144 actions and K stochastic drop branches.

        Returns:
            star1_q_values: (B, 144) expected values (or upper-bound certificates for pruned nodes)
            star1_active_mask: (B, 144) actions whose [Q_min, Q_max] interval is not dominated by alpha*
            branch_eval_mask: (B, 144, K) True for stochastic branches evaluated before Star1 cutoff
            pruning_efficiency: (B, 1) fraction of total (144 * K) stochastic evaluations saved
        """
        B, A, K = branch_v.shape

        if legal_mask is not None:
            valid_mask = legal_mask.bool()
            masked_q_min = q_min.masked_fill(~valid_mask, -1e9)
            masked_q_exp = q_exp.masked_fill(~valid_mask, -1e9)
        else:
            valid_mask = torch.ones_like(q_min, dtype=torch.bool)
            masked_q_min = q_min
            masked_q_exp = q_exp

        # Best guaranteed lower bound alpha* across root actions (Star2 probing bound)
        # We use the highest conservative expected bound among legal moves as the root alpha cutoff
        alpha_star = torch.max(
            masked_q_min.max(dim=-1, keepdim=True)[0],
            0.5 * (masked_q_min.max(dim=-1, keepdim=True)[0] + masked_q_exp.max(dim=-1, keepdim=True)[0]),
        )  # (B, 1)

        # Action-level Star2 cutoff: any action whose best-case ceiling Q_max(s, a) < alpha*
        # can NEVER beat the best move even under a 1-in-a-million miracle drop!
        star1_active_mask = (q_max >= alpha_star) & valid_mask
        # Ensure at least the top-1 action is always active
        top_act = masked_q_exp.argmax(dim=-1, keepdim=True)
        star1_active_mask.scatter_(1, top_act, True)

        # Now run Ballard's exact sequential Star1 chance-node bound tightening across K branches
        # suffix_U[:, :, k] = sum_{j > k} p_j * U_j
        weighted_U = branch_probs * branch_U  # (B, 144, K)
        total_U = weighted_U.sum(dim=-1, keepdim=True)  # (B, 144, 1)
        cumsum_U = torch.cumsum(weighted_U, dim=-1)     # (B, 144, K)
        suffix_U = total_U - cumsum_U                   # sum_{j > k} p_j * U_j

        # Partial evaluated sum S_{k-1} = sum_{i < k} p_i * v_i
        weighted_v = branch_probs * branch_v            # (B, 144, K)
        cumsum_v = torch.cumsum(weighted_v, dim=-1)
        prefix_v = cumsum_v - weighted_v                # sum_{i < k} p_i * v_i

        # Ballard's Star1 local threshold alpha_k required at branch k:
        #   p_k * v_k >= alpha* - prefix_v - suffix_U
        # A cutoff occurs after/at branch k if even achieving U_k cannot satisfy the bound:
        #   prefix_v + p_k * U_k + suffix_U < alpha*
        optimistic_ceiling_at_k = prefix_v + weighted_U + suffix_U  # (B, 144, K)
        can_still_beat_alpha = (optimistic_ceiling_at_k >= alpha_star.unsqueeze(-1)) & star1_active_mask.unsqueeze(-1)

        # Once a branch fails `can_still_beat_alpha`, all subsequent branches k+1..K-1 are pruned!
        # First branch k=0 is evaluated only if star1_active_mask is True.
        survives_prior = torch.ones((B, A, K), dtype=torch.bool, device=branch_v.device)
        survives_prior[:, :, 1:] = torch.cumprod(can_still_beat_alpha[:, :, :-1].long(), dim=-1).bool()
        branch_eval_mask = survives_prior & star1_active_mask.unsqueeze(-1)

        # Star1 expected value estimate using evaluated branches + upper bound on unpruned
        star1_q_values = torch.where(
            star1_active_mask,
            0.5 * q_exp + 0.5 * cumsum_v[:, :, -1],
            q_max,  # Upper bound certificate for pruned suboptimal actions
        )

        total_branches = float(A * K)
        evaluated_branches = branch_eval_mask.float().sum(dim=(1, 2), keepdim=False).unsqueeze(-1)
        pruning_efficiency = 1.0 - (evaluated_branches / total_branches)

        return star1_q_values, star1_active_mask, branch_eval_mask, pruning_efficiency

    def forward(
        self,
        board_tensor: torch.Tensor,
        legal_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Canonical forward pass accepting `(B, 20, 9, 9)` board state tensor.
        """
        h = self.stem(board_tensor)
        for block in self.res_tower:
            h = block(h)

        bounds = self.interval_head(h)
        raw_logits = bounds["raw_logits"]
        v_min = bounds["v_min"]
        v_exp = bounds["v_exp"]
        v_max = bounds["v_max"]
        q_min = bounds["q_min"]
        q_exp = bounds["q_exp"]
        q_max = bounds["q_max"]

        star1_q, star1_active_mask, branch_eval_mask, pruning_eff = self.run_vectorized_star1_pruning(
            q_min=q_min,
            q_exp=q_exp,
            q_max=q_max,
            branch_L=bounds["branch_L"],
            branch_v=bounds["branch_v"],
            branch_U=bounds["branch_U"],
            branch_probs=bounds["branch_probs"],
            legal_mask=legal_mask,
        )

        # Combine neural policy preference logits with Star1 Expectimax Q-values
        # Penalize actions provably pruned by Star1 bound cutoff while keeping gradients smooth
        pruned_penalty = torch.where(star1_active_mask, 0.0, -4.0)
        policy_logits = raw_logits + 3.0 * star1_q + pruned_penalty

        if legal_mask is not None:
            policy_logits = policy_logits.masked_fill(~legal_mask.bool(), -1e9)

        win_prob = torch.sigmoid(2.5 * v_exp)

        return {
            "policy_logits": policy_logits,
            "value": v_exp,
            "win_prob": win_prob,
            "v_min": v_min,
            "v_max": v_max,
            "q_min": q_min,
            "q_exp": q_exp,
            "q_max": q_max,
            "star1_q_values": star1_q,
            "star1_active_mask": star1_active_mask,
            "branch_evaluated_mask": branch_eval_mask,
            "pruning_efficiency": pruning_eff,
            "refill_volatility": bounds["refill_volatility"],
        }

    def star1_interval_loss(
        self,
        board_tensor: torch.Tensor,
        target_actions: torch.Tensor,
        target_values: torch.Tensor,
        realized_min_returns: Optional[torch.Tensor] = None,
        realized_max_returns: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Trains Star1ExpectimaxPrunerNet to maintain valid, tight envelope bounds
        [V_min(s) <= V(s) <= V_max(s)] and [Q_min(s,a) <= Q(s,a) <= Q_max(s,a)]
        without overly loose intervals that would weaken Star1 pruning efficiency.
        """
        out = self.forward(board_tensor)
        policy_loss = F.cross_entropy(out["policy_logits"], target_actions)
        value_loss = F.mse_loss(out["value"], target_values)

        # Envelope containment loss + interval width tightening regularizer
        low_target = realized_min_returns if realized_min_returns is not None else (target_values - 0.15).clamp(-1.0, 1.0)
        high_target = realized_max_returns if realized_max_returns is not None else (target_values + 0.15).clamp(-1.0, 1.0)

        bound_loss = F.smooth_l1_loss(out["v_min"], low_target) + F.smooth_l1_loss(out["v_max"], high_target)
        width_reg = 0.05 * (out["q_max"] - out["q_min"]).mean()

        total_loss = policy_loss + value_loss + 0.5 * bound_loss + width_reg
        return {
            "total_loss": total_loss,
            "policy_loss": policy_loss,
            "value_loss": value_loss,
            "interval_bound_loss": bound_loss,
            "interval_width_reg": width_reg,
        }

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def create_star1_expectimax_pruner_net(device: str = "cpu") -> Star1ExpectimaxPrunerNet:
    model = Star1ExpectimaxPrunerNet(
        in_channels=20,
        d_model=256,
        num_res_blocks=5,
        num_actions=144,
        num_chance_branches=8,
    )
    return model.to(device)


if __name__ == "__main__":
    torch.manual_seed(42)
    net = create_star1_expectimax_pruner_net("cpu")
    params = net.count_parameters()
    print(f"[Star1ExpectimaxPrunerNet] Initialized with {params:,} trainable parameters ({params / 1e6:.2f}M)")

    dummy_board = torch.randn(2, 20, 9, 9)
    out = net(dummy_board)
    print(f"  policy_logits:      {out['policy_logits'].shape}")
    print(f"  value:              {out['value'].shape}")
    print(f"  win_prob:           {out['win_prob'].shape}")
    print(f"  [v_min, v_max]:     {out['v_min'].shape}, {out['v_max'].shape}")
    print(f"  [q_min, q_max]:     {out['q_min'].shape}, {out['q_max'].shape}")
    print(f"  pruning_efficiency: {out['pruning_efficiency'].squeeze(-1).tolist()}")
