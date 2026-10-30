"""Cross-Attempt Episodic KV-Memory Transformer (Transformer-XL Style).

Allows Attempt k on a hard Candy Crush level to cross-attend to compressed
episodic memory vectors from failed Attempts 1..k-1, avoiding bottleneck traps
and dead-end board configurations discovered in earlier tries.
"""

from typing import Dict, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossAttemptMemoryLayer(nn.Module):
    """Self-attention over current board tokens + cross-attention to past attempts."""

    def __init__(self, d_model: int = 128, nhead: int = 4, dropout: float = 0.1) -> None:
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Linear(d_model * 2, d_model),
        )
        self.norm3 = nn.LayerNorm(d_model)
        self.mem_gate = nn.Parameter(torch.tensor(0.5))

    def forward(
        self,
        tokens: torch.Tensor,
        attempt_memory_kv: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        attn_out, _ = self.self_attn(tokens, tokens, tokens, need_weights=False)
        tokens = self.norm1(tokens + attn_out)

        if attempt_memory_kv is not None and attempt_memory_kv.size(1) > 0:
            cross_out, _ = self.cross_attn(
                query=tokens,
                key=attempt_memory_kv,
                value=attempt_memory_kv,
                need_weights=False,
            )
            tokens = self.norm2(tokens + torch.tanh(self.mem_gate) * cross_out)
        else:
            tokens = self.norm2(tokens)

        tokens = self.norm3(tokens + self.ffn(tokens))
        return tokens


class EpisodicMemoryTransformerCandyNet(nn.Module):
    """Transformer-XL inspired cross-attempt episodic memory network.

    Args:
        in_channels: Input board channels (20).
        d_model: Token and episodic memory embedding dimension.
        num_layers: Number of cross-attempt transformer blocks.
        num_actions: Action space size (144 directional swaps).
    """

    def __init__(
        self,
        in_channels: int = 20,
        d_model: int = 128,
        num_layers: int = 3,
        nhead: int = 4,
        num_actions: int = 144,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.num_actions = num_actions

        self.patch_embed = nn.Sequential(
            nn.Conv2d(in_channels, d_model, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(d_model),
            nn.GELU(),
        )
        self.pos_embed = nn.Parameter(torch.randn(1, 81, d_model) * 0.02)
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

        self.layers = nn.ModuleList(
            [CrossAttemptMemoryLayer(d_model=d_model, nhead=nhead) for _ in range(num_layers)]
        )

        # Learned default memory slot when attempt_memory_kv is None (Attempt 1)
        self.null_memory = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

        # Compresses current board + trajectory state into a new episodic memory vector
        self.memory_writer = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.LayerNorm(d_model),
            nn.Tanh(),
        )

        self.policy_head = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Linear(d_model, num_actions),
        )
        self.value_head = nn.Sequential(
            nn.Linear(d_model * 2, 64),
            nn.GELU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        board_tensor: torch.Tensor,
        attempt_memory_kv: Optional[torch.Tensor] = None,
        legal_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        b = board_tensor.size(0)
        spatial = self.patch_embed(board_tensor).flatten(2).transpose(1, 2)  # (B, 81, D)
        spatial = spatial + self.pos_embed
        cls = self.cls_token.expand(b, -1, -1)
        tokens = torch.cat([cls, spatial], dim=1)  # (B, 82, D)

        mem_kv = (
            attempt_memory_kv
            if attempt_memory_kv is not None
            else self.null_memory.expand(b, 1, self.d_model)
        )

        for layer in self.layers:
            tokens = layer(tokens, mem_kv)

        cls_out = tokens[:, 0]
        board_mean = tokens[:, 1:].mean(dim=1)
        combined = torch.cat([cls_out, board_mean], dim=-1)

        policy_logits = self.policy_head(combined)
        if legal_mask is not None:
            policy_logits = policy_logits.masked_fill(~legal_mask.bool(), -1e9)

        win_prob = self.value_head(combined)
        new_memory_vector = self.memory_writer(combined)

        return {
            "policy_logits": policy_logits,
            "win_prob": win_prob,
            "new_memory_vector": new_memory_vector,
        }
