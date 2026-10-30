"""
CandyDecisionTransformer: Causal Trajectory & Target-Conditioned ("Upside-Down RL") Transformer.
Based on:
  - Chen et al. (NeurIPS 2021), "Decision Transformer: Reinforcement Learning via Sequence Modeling"
  - Schmidhuber (2019), "Reinforcement Learning Upside Down"

Models 5-modality trajectory sequences over a context window of K=8 turns:
  (Return-To-Go R_t, Objectives-Remaining-Ratio O_t, Moves-Remaining M_t, Board-State s_t, Action a_t)
Allows prompting the policy at inference time with desired victory targets
(`target_objectives_remaining = 0.0`, `target_return_to_go = 1.0`) to pace special-candy combinations
against remaining move budgets, while supporting a canonical single-step `(B, 20, 9, 9)` fallback mode.
"""

import math
from typing import Dict, Tuple, Optional, Union
import torch
import torch.nn as nn
import torch.nn.functional as F


class DecisionTransformerOutput(tuple):
    """
    Hybrid output container supporting BOTH tuple unpacking:
        policy_logits, win_prob = model(board_tensor)
    AND dictionary key lookup:
        out = model(board_tensor); out["policy_logits"], out["win_prob"], out["trajectory_logits"]
    """
    _KEYS = ("policy_logits", "win_prob", "trajectory_logits", "predicted_obj_delta")

    def __new__(
        cls,
        policy_logits: torch.Tensor,
        win_prob: torch.Tensor,
        trajectory_logits: Optional[torch.Tensor] = None,
        predicted_obj_delta: Optional[torch.Tensor] = None
    ):
        instance = super().__new__(cls, (policy_logits, win_prob))
        instance._extras = {
            "policy_logits": policy_logits,
            "win_prob": win_prob,
            "value": win_prob,
            "trajectory_logits": trajectory_logits if trajectory_logits is not None else policy_logits.unsqueeze(1),
            "predicted_obj_delta": predicted_obj_delta
        }
        return instance

    def __getitem__(self, key: Union[int, slice, str]):
        if isinstance(key, str):
            if key in self._extras:
                return self._extras[key]
            raise KeyError(f"Unknown key '{key}'. Available keys: {list(self._extras.keys())}")
        return super().__getitem__(key)

    def keys(self):
        return self._extras.keys()


class SpatialStateEncoder(nn.Module):
    """Encodes a (N, 20, 9, 9) board state into a d_model state token with spatial attention pooling."""
    def __init__(self, in_channels: int = 20, d_model: int = 256):
        super().__init__()
        self.conv_net = nn.Sequential(
            nn.Conv2d(in_channels, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.Conv2d(128, d_model, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(d_model),
            nn.GELU(),
            nn.Conv2d(d_model, d_model, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(d_model),
            nn.GELU()
        )
        # Keep both global pooled and spatial bottleneck features for accurate 144-swap decoding
        self.spatial_bottleneck = nn.Sequential(
            nn.Conv2d(d_model, 32, kernel_size=1),
            nn.GELU(),
            nn.Flatten(),
            nn.Linear(32 * 9 * 9, d_model),
            nn.LayerNorm(d_model)
        )

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        feat = self.conv_net(states)
        pooled = feat.mean(dim=(2, 3))
        spatial_tok = self.spatial_bottleneck(feat)
        return pooled + spatial_tok


class CausalTransformerBlock(nn.Module):
    """Pre-LayerNorm Causal Self-Attention Block for autoregressive trajectory modeling."""
    def __init__(self, d_model: int = 256, num_heads: int = 8, dropout: float = 0.05):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout)
        )

    def forward(self, x: torch.Tensor, causal_mask: torch.Tensor) -> torch.Tensor:
        x_norm = self.norm1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm, attn_mask=causal_mask, need_weights=False)
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x


class CandyDecisionTransformer(nn.Module):
    """
    Causal Trajectory Decision Transformer for Candy Crush Saga.
    Conditions action generation on a K=8 window of:
      1. Return-To-Go (RTG)
      2. Objectives-Remaining-Ratio (0.0 = Victory Target)
      3. Moves-Remaining-Ratio
      4. 20-Channel Spatial Board State
      5. Previous Canonical Swap Action
    """
    MODALITIES_PER_STEP = 5  # (RTG, Obj_Ratio, Moves_Ratio, State, Action)

    def __init__(
        self,
        in_channels: int = 20,
        d_model: int = 256,
        num_layers: int = 6,
        num_heads: int = 8,
        context_len: int = 8,
        num_actions: int = 144
    ):
        super().__init__()
        self.in_channels = in_channels
        self.d_model = d_model
        self.context_len = context_len
        self.num_actions = num_actions
        self.max_seq_tokens = context_len * self.MODALITIES_PER_STEP

        # 1. Modality Embeddings
        self.rtg_embed = nn.Sequential(
            nn.Linear(1, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, d_model)
        )
        self.obj_embed = nn.Sequential(
            nn.Linear(1, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, d_model)
        )
        self.moves_embed = nn.Sequential(
            nn.Linear(1, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, d_model)
        )
        self.state_encoder = SpatialStateEncoder(in_channels=in_channels, d_model=d_model)
        # Action vocabulary includes 0..143 canonical swaps + index 144 as <PAD/QUERY> token
        self.action_embed = nn.Embedding(num_actions + 1, d_model)

        # Timestep & Modality-Type Positional Embeddings
        self.timestep_embed = nn.Embedding(context_len, d_model)
        self.modality_type_embed = nn.Embedding(self.MODALITIES_PER_STEP, d_model)
        self.embed_norm = nn.LayerNorm(d_model)

        # Causal Transformer Backbone
        self.blocks = nn.ModuleList([
            CausalTransformerBlock(d_model=d_model, num_heads=num_heads)
            for _ in range(num_layers)
        ])
        self.final_norm = nn.LayerNorm(d_model)

        # Output Heads from the conditioned State Token representation
        self.policy_head = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Linear(d_model, num_actions)
        )
        self.win_head = nn.Sequential(
            nn.Linear(d_model * 2, 128),
            nn.GELU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        self.obj_delta_head = nn.Sequential(
            nn.Linear(d_model * 2, 64),
            nn.GELU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward_trajectory(
        self,
        states: torch.Tensor,
        actions: Optional[torch.Tensor] = None,
        returns_to_go: Optional[torch.Tensor] = None,
        objectives_remaining: Optional[torch.Tensor] = None,
        moves_remaining: Optional[torch.Tensor] = None
    ) -> DecisionTransformerOutput:
        """
        Full trajectory forward pass over K <= 8 turns.
        Args:
            states: (B, K, 20, 9, 9) sequence of board states
            actions: (B, K) sequence of past actions in [0, 144] (defaults to 144 query token)
            returns_to_go: (B, K, 1) target return-to-go sequence (defaults to 1.0 victory return)
            objectives_remaining: (B, K, 1) target objectives ratio (defaults to 0.0 victory prompt)
            moves_remaining: (B, K, 1) normalized moves remaining (extracted from channel 19 if omitted)
        """
        B, K, C, H, W = states.shape
        device = states.device
        if K > self.context_len:
            states = states[:, -self.context_len:]
            K = self.context_len

        if returns_to_go is None:
            returns_to_go = torch.ones((B, K, 1), device=device, dtype=states.dtype)
        if objectives_remaining is None:
            objectives_remaining = torch.zeros((B, K, 1), device=device, dtype=states.dtype)
        if moves_remaining is None:
            # Extract normalized moves from channel 19 mean
            moves_remaining = states[:, :, 19, :, :].mean(dim=(-2, -1), keepdim=True).squeeze(-1)
        if actions is None:
            actions = torch.full((B, K), self.num_actions, device=device, dtype=torch.long)

        # Encode all 5 modalities to (B, K, d_model)
        t_idx = torch.arange(K, device=device).unsqueeze(0)  # (1, K)
        t_emb = self.timestep_embed(t_idx)

        rtg_tok = self.rtg_embed(returns_to_go) + t_emb + self.modality_type_embed.weight[0]
        obj_tok = self.obj_embed(objectives_remaining) + t_emb + self.modality_type_embed.weight[1]
        mov_tok = self.moves_embed(moves_remaining) + t_emb + self.modality_type_embed.weight[2]

        flat_states = states.reshape(B * K, C, H, W)
        state_raw = self.state_encoder(flat_states).view(B, K, self.d_model)
        sta_tok = state_raw + t_emb + self.modality_type_embed.weight[3]

        act_tok = self.action_embed(actions.clamp(0, self.num_actions)) + t_emb + self.modality_type_embed.weight[4]

        # Interleave tokens into (B, 5 * K, d_model): [R_0, O_0, M_0, s_0, a_0, R_1, ...]
        stacked = torch.stack([rtg_tok, obj_tok, mov_tok, sta_tok, act_tok], dim=2)  # (B, K, 5, d_model)
        token_seq = self.embed_norm(stacked.view(B, K * self.MODALITIES_PER_STEP, self.d_model))

        # Strict lower-triangular causal attention mask
        seq_len = K * self.MODALITIES_PER_STEP
        causal_mask = torch.triu(
            torch.full((seq_len, seq_len), float("-inf"), device=device), diagonal=1
        )

        h = token_seq
        for block in self.blocks:
            h = block(h, causal_mask=causal_mask)
        h = self.final_norm(h)

        # Reshape back to (B, K, 5, d_model) and extract State token (modality index 3) + Obj token (index 1)
        h_steps = h.view(B, K, self.MODALITIES_PER_STEP, self.d_model)
        state_ctx = torch.cat([h_steps[:, :, 3, :], h_steps[:, :, 1, :]], dim=-1)  # (B, K, 2*d_model)

        traj_logits = self.policy_head(state_ctx)       # (B, K, 144)
        last_logits = traj_logits[:, -1, :]             # (B, 144)
        win_prob = self.win_head(state_ctx[:, -1, :])   # (B, 1)
        obj_delta = self.obj_delta_head(state_ctx[:, -1, :])  # (B, 1)

        return DecisionTransformerOutput(
            policy_logits=last_logits,
            win_prob=win_prob,
            trajectory_logits=traj_logits,
            predicted_obj_delta=obj_delta
        )

    def forward(
        self,
        board_or_trajectory: torch.Tensor,
        target_objectives_remaining: float = 0.0,
        target_return_to_go: float = 1.0,
        moves_left_ratio: Optional[torch.Tensor] = None
    ) -> DecisionTransformerOutput:
        """
        Polymorphic forward pass:
          - If `board_or_trajectory` has shape `(B, 20, 9, 9)`: runs single-step fallback mode
            prompted with upside-down RL victory targets (`target_objectives_remaining = 0.0`).
          - If `board_or_trajectory` has shape `(B, K, 20, 9, 9)`: runs full K-step trajectory mode.
        """
        if board_or_trajectory.dim() == 4:
            B = board_or_trajectory.shape[0]
            device = board_or_trajectory.device
            dtype = board_or_trajectory.dtype
            states = board_or_trajectory.unsqueeze(1)  # (B, 1, 20, 9, 9)
            rtg = torch.full((B, 1, 1), target_return_to_go, device=device, dtype=dtype)
            obj = torch.full((B, 1, 1), target_objectives_remaining, device=device, dtype=dtype)
            mov = moves_left_ratio.view(B, 1, 1) if moves_left_ratio is not None else None
            return self.forward_trajectory(
                states=states,
                returns_to_go=rtg,
                objectives_remaining=obj,
                moves_remaining=mov
            )
        elif board_or_trajectory.dim() == 5:
            return self.forward_trajectory(states=board_or_trajectory)
        else:
            raise ValueError(f"Expected 4D (B,20,9,9) or 5D (B,K,20,9,9) input, got {board_or_trajectory.shape}")

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def create_candy_decision_transformer(device: str = "cpu") -> CandyDecisionTransformer:
    model = CandyDecisionTransformer(
        in_channels=20,
        d_model=256,
        num_layers=6,
        num_heads=8,
        context_len=8,
        num_actions=144
    )
    return model.to(device)


if __name__ == "__main__":
    torch.manual_seed(42)
    net = create_candy_decision_transformer("cpu")
    params = net.count_parameters()
    print(f"[CandyDecisionTransformer] Initialized with {params:,} trainable parameters ({params / 1e6:.2f}M)")

    # Test 1: Canonical Single-Step Fallback Mode (B, 20, 9, 9)
    dummy_single = torch.randn(2, 20, 9, 9)
    policy_logits, win_prob = net(dummy_single, target_objectives_remaining=0.0)
    print(f"  [Single-Step Mode] policy_logits: {policy_logits.shape} | win_prob: {win_prob.shape}")

    # Test 2: Full K=8 Trajectory Mode (B, 8, 20, 9, 9)
    dummy_traj = torch.randn(2, 8, 20, 9, 9)
    dummy_actions = torch.randint(0, 144, (2, 8))
    dummy_rtg = torch.linspace(1.0, 0.2, 8).unsqueeze(0).unsqueeze(-1).expand(2, 8, 1)
    dummy_obj = torch.linspace(0.8, 0.0, 8).unsqueeze(0).unsqueeze(-1).expand(2, 8, 1)
    dummy_mov = torch.linspace(1.0, 0.1, 8).unsqueeze(0).unsqueeze(-1).expand(2, 8, 1)

    traj_out = net.forward_trajectory(
        states=dummy_traj,
        actions=dummy_actions,
        returns_to_go=dummy_rtg,
        objectives_remaining=dummy_obj,
        moves_remaining=dummy_mov
    )
    print(f"  [K=8 Trajectory Mode] trajectory_logits: {traj_out['trajectory_logits'].shape} | final policy: {traj_out['policy_logits'].shape}")
