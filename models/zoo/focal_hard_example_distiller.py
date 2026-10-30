"""
AlphaCandy Model Zoo — Hard-Negative Focal Distiller & Failure-Memory Tabu Network
(`HardNegativeFocalDistillerNet`).

Based on:
  - Lin et al. (ICCV 2017), "Focal Loss for Dense Object Detection"
  - Glover (1989), "Tabu Search — Part I" (adapted to Differentiable Neural Tabu Memory)
  - Hinton et al. (2015), "Distilling the Knowledge in a Neural Network"

Tailored specifically for hard-negative mining and distillation on the 349 stubborn bottleneck
levels where standard cross-entropy policies repeatedly fall into identical dead-end move traps:
  1. Focal Policy Cross-Entropy Loss (`focal_policy_loss`):
       L_focal = -(1 - p_t)^gamma * log(p_t)   with gamma = 2.5
     Down-weights easy 3-match moves by >95% so gradient updates focus almost exclusively on
     subtle, low-probability breakthrough moves on stubborn levels.
  2. Differentiable Failure-Memory Tabu Head (`FailureMemoryTabuHead`):
     Maintains an associative cosine-similarity Failure Memory Bank (and accepts explicit per-state
     `failed_actions_mask` / counts from prior failed attempts on the same level) to compute a
     learned state-dependent Tabu Repulsion Penalty R_tabu(s, a) >= 0 that repels the policy away
     from repeating dead-end trajectories.
"""

from typing import Dict, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F


class BottleneckResidualBlock(nn.Module):
    """Spatial residual block with dilated 3x3 context branch for detecting corner/choke-point traps."""

    def __init__(self, d_model: int = 192):
        super().__init__()
        half = d_model // 2
        self.local_conv = nn.Sequential(
            nn.Conv2d(d_model, half, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(half),
            nn.SiLU()
        )
        # Dilated receptive field captures long-range blocker choke points & isolated corners
        self.dilated_conv = nn.Sequential(
            nn.Conv2d(d_model, half, kernel_size=3, padding=2, dilation=2, bias=False),
            nn.BatchNorm2d(half),
            nn.SiLU()
        )
        self.fuse = nn.Sequential(
            nn.Conv2d(d_model, d_model, kernel_size=1, bias=False),
            nn.BatchNorm2d(d_model)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        branch = torch.cat([self.local_conv(x), self.dilated_conv(x)], dim=1)
        return F.silu(x + self.fuse(branch))


class FailureMemoryTabuHead(nn.Module):
    """
    Neural Failure-Memory Tabu Repulsion Head.
    Combines:
      1. Explicit per-sample `failed_actions_mask` (B, 144) passed from previous failed attempts on
         the current level/board state.
      2. A persistent Associative Failure Ring Buffer (`memory_keys` in R^(M x 128),
         `memory_failed_actions` in R^(M x 144)) that retrieves historical failed moves on
         structurally similar board states via cosine similarity retrieval.
    Outputs a non-negative Tabu Repulsion Penalty R_tabu(s, a) in R_+^(B x 144) that suppresses
    dead-end actions.
    """

    def __init__(
        self,
        d_model: int = 192,
        key_dim: int = 128,
        num_actions: int = 144,
        memory_capacity: int = 256,
        similarity_threshold: float = 0.80
    ):
        super().__init__()
        self.d_model = d_model
        self.key_dim = key_dim
        self.num_actions = num_actions
        self.memory_capacity = memory_capacity
        self.similarity_threshold = similarity_threshold

        # Projects global board embedding -> L2-normalized state lookup key
        self.key_proj = nn.Linear(d_model, key_dim, bias=False)

        # Context-aware repulsion modulator: decides how strongly to penalize each failed move
        # given the board's current latent state and failure history vector
        self.tabu_gate_mlp = nn.Sequential(
            nn.Linear(d_model + num_actions, 256),
            nn.LayerNorm(256),
            nn.SiLU(),
            nn.Linear(256, num_actions),
            nn.Softplus()
        )
        self.base_tabu_strength = nn.Parameter(torch.tensor(2.5))

        # Persistent associative ring buffer for failed (state_key, action) pairs
        self.register_buffer("memory_keys", torch.zeros(memory_capacity, key_dim))
        self.register_buffer("memory_failed_counts", torch.zeros(memory_capacity, num_actions))
        self.register_buffer("memory_valid_mask", torch.zeros(memory_capacity, dtype=torch.bool))
        self.register_buffer("write_ptr", torch.tensor(0, dtype=torch.long))

    @torch.no_grad()
    def record_failed_attempt(
        self,
        state_embedding: torch.Tensor,
        failed_actions: torch.Tensor
    ) -> None:
        """
        Records failed actions from a dead-end attempt into the Associative Tabu Ring Buffer.
        Args:
            state_embedding: (B, d_model) global state representation
            failed_actions: (B,) action indices in [0, 143] or (B, 144) multi-hot failure counts
        """
        keys = F.normalize(self.key_proj(state_embedding), p=2, dim=-1)  # (B, key_dim)
        B = keys.shape[0]

        if failed_actions.dim() == 1:
            fail_vec = F.one_hot(failed_actions.long(), num_classes=self.num_actions).float()
        else:
            fail_vec = failed_actions.float()

        for b in range(B):
            idx = int(self.write_ptr.item()) % self.memory_capacity
            self.memory_keys[idx] = keys[b]
            self.memory_failed_counts[idx] = fail_vec[b]
            self.memory_valid_mask[idx] = True
            self.write_ptr.add_(1)

    def retrieve_memory_failures(self, state_embedding: torch.Tensor) -> torch.Tensor:
        """
        Queries the associative ring buffer for historical failed actions on similar states.
        Returns retrieved failure profile of shape (B, 144).
        """
        B = state_embedding.shape[0]
        if not self.memory_valid_mask.any():
            return torch.zeros(B, self.num_actions, device=state_embedding.device, dtype=state_embedding.dtype)

        query_keys = F.normalize(self.key_proj(state_embedding), p=2, dim=-1)  # (B, key_dim)
        valid_keys = self.memory_keys[self.memory_valid_mask]                  # (M_valid, key_dim)
        valid_counts = self.memory_failed_counts[self.memory_valid_mask]       # (M_valid, 144)

        cos_sim = torch.matmul(query_keys, valid_keys.T)                       # (B, M_valid)
        sim_mask = (cos_sim >= self.similarity_threshold).float()
        weights = F.softmax(cos_sim * 10.0, dim=-1) * sim_mask                 # (B, M_valid)
        return torch.matmul(weights, valid_counts)                             # (B, 144)

    def forward(
        self,
        state_embedding: torch.Tensor,
        failed_actions_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Computes the non-negative Tabu Repulsion Penalty R_tabu(s, a) in R_+^(B x 144).
        """
        B = state_embedding.shape[0]
        retrieved_failures = self.retrieve_memory_failures(state_embedding)

        if failed_actions_mask is not None:
            if failed_actions_mask.dim() == 1:
                explicit_fail = F.one_hot(failed_actions_mask.long(), num_classes=self.num_actions).float()
            else:
                explicit_fail = failed_actions_mask.float()
            combined_failures = explicit_fail + retrieved_failures
        else:
            combined_failures = retrieved_failures

        # Learned state-conditioned modulation + direct proportional tabu repulsion
        gate_scale = self.tabu_gate_mlp(torch.cat([state_embedding, combined_failures], dim=-1))
        tabu_penalty = combined_failures * (F.softplus(self.base_tabu_strength) + gate_scale)
        return tabu_penalty, combined_failures


class HardNegativeFocalDistillerNet(nn.Module):
    """
    Hard-Negative Mining Focal Policy Distiller & Failure-Memory Tabu Network
    (`HardNegativeFocalDistillerNet`).

    Designed to conquer the 349 stubborn Candy Crush levels by combining:
      1. Dilated Choke-Point Spatial Backbone (`BottleneckResidualBlock`)
      2. Focal Policy Cross-Entropy Loss L_focal = -(1 - p_t)^gamma * log(p_t) (gamma = 2.5)
      3. Differentiable Failure-Memory Tabu Head that repels the policy away from failed moves
      4. Board Hardness / Trap Difficulty Estimator Head
    """

    def __init__(
        self,
        in_channels: int = 20,
        d_model: int = 192,
        num_blocks: int = 6,
        num_actions: int = 144,
        focal_gamma: float = 2.5,
        board_h: int = 9,
        board_w: int = 9
    ):
        super().__init__()
        self.in_channels = in_channels
        self.d_model = d_model
        self.num_blocks = num_blocks
        self.num_actions = num_actions
        self.focal_gamma = focal_gamma
        self.board_h = board_h
        self.board_w = board_w

        # 1. Input Stem + Dilated Bottleneck Residual Tower
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, d_model, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(d_model),
            nn.SiLU()
        )
        self.blocks = nn.ModuleList([
            BottleneckResidualBlock(d_model=d_model) for _ in range(num_blocks)
        ])

        # Global Self-Attention for non-local trap/blocker correlation
        self.attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=6, batch_first=True)
        self.attn_norm = nn.LayerNorm(d_model)

        # 2. Raw Policy Head
        self.policy_conv = nn.Sequential(
            nn.Conv2d(d_model, 48, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(48),
            nn.SiLU()
        )
        self.policy_mlp = nn.Sequential(
            nn.Linear(48 * board_h * board_w, 256),
            nn.LayerNorm(256),
            nn.SiLU(),
            nn.Linear(256, num_actions)
        )

        # 3. Failure-Memory Tabu Repulsion Head
        self.tabu_head = FailureMemoryTabuHead(
            d_model=d_model,
            key_dim=128,
            num_actions=num_actions,
            memory_capacity=256
        )

        # 4. Value, Win Probability, and State Hardness / Bottleneck Difficulty Heads
        self.value_head = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.SiLU(),
            nn.Linear(128, 1)
        )
        self.hardness_head = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.SiLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def encode_board(self, board_tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B, C, H, W = board_tensor.shape
        x = self.stem(board_tensor)
        for block in self.blocks:
            x = block(x)

        tokens = x.flatten(2).transpose(1, 2)
        attn_out, _ = self.attn(tokens, tokens, tokens, need_weights=False)
        tokens = self.attn_norm(tokens + attn_out)

        feat_2d = tokens.transpose(1, 2).contiguous().view(B, self.d_model, H, W)
        global_feat = tokens.mean(dim=1)
        return feat_2d, global_feat

    def forward(
        self,
        board_tensor: torch.Tensor,
        legal_mask: Optional[torch.Tensor] = None,
        failed_actions_mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            board_tensor: (B, 20, 9, 9) canonical board state tensor
            legal_mask: Optional (B, 144) mask of legal swaps
            failed_actions_mask: Optional (B, 144) or (B,) tensor of failed actions from prior
                                 attempts on the same level to repel away from
        Returns:
            Dictionary with canonical `policy_logits` (B, 144), `value` (B, 1), `win_prob` (B, 1),
            `raw_policy_logits` (B, 144), `tabu_repulsion_penalty` (B, 144), and `hardness_score` (B, 1).
        """
        feat_2d, global_feat = self.encode_board(board_tensor)

        raw_policy_logits = self.policy_mlp(self.policy_conv(feat_2d).flatten(1))
        tabu_penalty, active_failures = self.tabu_head(global_feat, failed_actions_mask=failed_actions_mask)

        # Subtract Tabu Repulsion Penalty to repel policy away from dead-end failed moves
        policy_logits = raw_policy_logits - tabu_penalty

        if legal_mask is not None:
            policy_logits = policy_logits.masked_fill(~legal_mask.bool(), -1e9)

        val_raw = self.value_head(global_feat)
        value = torch.tanh(val_raw)
        win_prob = torch.sigmoid(val_raw)
        hardness_score = self.hardness_head(global_feat)

        return {
            "policy_logits": policy_logits,
            "raw_policy_logits": raw_policy_logits,
            "tabu_repulsion_penalty": tabu_penalty,
            "active_failure_memory": active_failures,
            "value": value,
            "win_prob": win_prob,
            "hardness_score": hardness_score,
            "state_embedding": global_feat,
        }

    def focal_policy_loss(
        self,
        policy_logits: torch.Tensor,
        target_actions: torch.Tensor,
        gamma: Optional[float] = None,
        sample_hardness_weights: Optional[torch.Tensor] = None,
        failed_actions_mask: Optional[torch.Tensor] = None,
        tabu_margin_weight: float = 0.25
    ) -> Dict[str, torch.Tensor]:
        """
        Computes Focal Policy Cross-Entropy Loss:
            L_focal = -(1 - p_t)^gamma * log(p_t)   (default gamma = 2.5)
        plus an optional Tabu Margin Repulsion Loss that penalizes probability mass assigned
        to known failed actions from dead-end trajectories.
        """
        g = self.focal_gamma if gamma is None else gamma
        log_probs = F.log_softmax(policy_logits, dim=-1)
        probs = torch.exp(log_probs)

        # Gather p_t and log(p_t) for ground-truth winning/teacher actions
        log_pt = log_probs.gather(1, target_actions.unsqueeze(-1)).squeeze(-1)
        pt = probs.gather(1, target_actions.unsqueeze(-1)).squeeze(-1)

        # Focal modulating factor (1 - p_t)^gamma
        focal_weight = (1.0 - pt).clamp(min=1e-6).pow(g)
        per_sample_focal = -focal_weight * log_pt

        if sample_hardness_weights is not None:
            per_sample_focal = per_sample_focal * sample_hardness_weights.view(-1)

        focal_ce = per_sample_focal.mean()

        # Optional unlikelihood / repulsion loss on known failed actions: -log(1 - p_fail)
        tabu_loss = torch.tensor(0.0, device=policy_logits.device)
        if failed_actions_mask is not None:
            if failed_actions_mask.dim() == 1:
                fail_indicator = F.one_hot(failed_actions_mask.long(), num_classes=self.num_actions).float()
            else:
                fail_indicator = (failed_actions_mask > 0).float()
            # Ensure target action is not penalized
            target_oh = F.one_hot(target_actions.long(), num_classes=self.num_actions).float()
            fail_indicator = fail_indicator * (1.0 - target_oh)
            fail_probs = (probs * fail_indicator).clamp(max=1.0 - 1e-6)
            tabu_loss = -torch.log(1.0 - fail_probs).sum(dim=-1).mean()

        total_loss = focal_ce + tabu_margin_weight * tabu_loss
        return {
            "total_loss": total_loss,
            "focal_ce_loss": focal_ce,
            "tabu_repulsion_loss": tabu_loss,
            "mean_focal_factor": focal_weight.mean(),
        }

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def create_hard_negative_focal_distiller_net(device: str = "cpu") -> HardNegativeFocalDistillerNet:
    model = HardNegativeFocalDistillerNet(
        in_channels=20,
        d_model=192,
        num_blocks=6,
        num_actions=144,
        focal_gamma=2.5
    )
    return model.to(device)


if __name__ == "__main__":
    torch.manual_seed(42)
    net = create_hard_negative_focal_distiller_net("cpu")
    params = net.count_parameters()
    print(f"[HardNegativeFocalDistillerNet] Trainable parameters: {params:,} ({params / 1e6:.2f}M)")

    dummy_board = torch.randn(2, 20, 9, 9)
    failed_mask = torch.zeros(2, 144)
    failed_mask[0, 15] = 2.0  # Action 15 failed twice on sample 0
    failed_mask[1, 88] = 1.0  # Action 88 failed once on sample 1

    out = net(dummy_board, failed_actions_mask=failed_mask)
    # Also test recording into the associative ring buffer
    net.tabu_head.record_failed_attempt(out["state_embedding"], torch.tensor([15, 88]))
    out_after_mem = net(dummy_board)

    loss_dict = net.focal_policy_loss(
        out["policy_logits"],
        target_actions=torch.tensor([42, 101]),
        failed_actions_mask=failed_mask
    )

    print(f"  policy_logits:          {out['policy_logits'].shape}")
    print(f"  value:                  {out['value'].shape}")
    print(f"  win_prob:               {out['win_prob'].shape}")
    print(f"  tabu_repulsion_penalty: {out_after_mem['tabu_repulsion_penalty'].shape}")
    print(f"  hardness_score:         {out['hardness_score'].shape}")
    print(f"  focal_loss (gamma=2.5): {loss_dict['total_loss'].item():.4f} (focal_ce={loss_dict['focal_ce_loss'].item():.4f})")
