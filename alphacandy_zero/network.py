"""
AlphaCandyZeroNet - the policy/value network inside the AlphaCandy-Zero search loop.

Why this shape (and why NOT 79M / 300M parameters):
  * The network is called inside search. On Kaggle's 4 vCPUs every extra millisecond per call
    is paid ~32-64x per move. KataGo and AlphaZero reached superhuman Go with 6-40 block ResNets;
    our 9x9 match-3 board is far smaller than 19x19 Go. Compute should go into SEARCH and DATA.
  * Only components that passed a real test are kept:
      - S6 colour-invariant stem: measured 0.0000% logit drift under colour permutations
        (analysis/ARCHITECTURAL_ABLATIONS_REPORT.md, ablation 2) vs 58.7% for a plain conv stem.
      - Bilinear symmetric edge policy head: 100% swap-direction consistency with ~80x fewer
        head parameters (ablation 1).
      - Separate moves / objectives / pace conditioning via FiLM (fixes legacy channel-19 bug).
  * KataGo-style auxiliary targets give dense gradient: per-cell "will this jelly be cleared"
    map (ownership analogue) and a distribution over final objectives remaining.

Inputs : planes (B, 24, 9, 9), globals (B, 16), optional legal mask (B, 144)
Outputs: policy_logits (B,144), win_logit (B,1), final_bins (B,8), jelly_logits (B,81)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from alphacandy_zero.encoding import (
    ACTION_U,
    ACTION_V,
    NUM_ACTIONS,
    NUM_COLOR_PLANES,
    NUM_GLOBALS,
    NUM_PLANES,
)

NUM_FINAL_BINS = 8  # final_remaining / current_remaining in {0, ≤.1, ≤.25, ≤.4, ≤.6, ≤.8, ≤1, >1}
FINAL_BIN_EDGES = (0.0, 0.1, 0.25, 0.4, 0.6, 0.8, 1.0)


@dataclass
class ZeroConfig:
    channels: int = 96
    blocks: int = 6
    attn_after: List[int] = field(default_factory=lambda: [3])
    color_channels: int = 32
    edge_dim: int = 64
    heads: int = 4


PRESETS: Dict[str, ZeroConfig] = {
    "tiny": ZeroConfig(channels=32, blocks=2, attn_after=[], color_channels=16, edge_dim=32, heads=2),
    "cpu": ZeroConfig(channels=96, blocks=6, attn_after=[3], color_channels=32, edge_dim=64, heads=4),
    "base": ZeroConfig(channels=192, blocks=10, attn_after=[4, 8], color_channels=48, edge_dim=128, heads=8),
    "large": ZeroConfig(channels=320, blocks=16, attn_after=[5, 10, 15], color_channels=64, edge_dim=192, heads=8),
}


class ColorInvariantStem(nn.Module):
    """Shared per-colour encoder + symmetric (mean, max) pooling => exact S6 invariance."""

    def __init__(self, cfg: ZeroConfig):
        super().__init__()
        aux = NUM_PLANES - NUM_COLOR_PLANES
        dc = cfg.color_channels
        self.per_color = nn.Sequential(
            nn.Conv2d(1 + aux, dc, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(dc, dc, 3, padding=1), nn.ReLU(inplace=True),
        )
        self.aux_conv = nn.Conv2d(aux, cfg.channels, 3, padding=1)
        self.fuse = nn.Conv2d(2 * dc + cfg.channels, cfg.channels, 1)
        self.bn = nn.BatchNorm2d(cfg.channels)
        self.pos = nn.Parameter(torch.zeros(1, cfg.channels, 9, 9))

    def forward(self, planes: torch.Tensor) -> torch.Tensor:
        B = planes.size(0)
        col = planes[:, :NUM_COLOR_PLANES]  # (B,6,9,9)
        aux = planes[:, NUM_COLOR_PLANES:]  # (B,18,9,9)
        per = torch.cat(
            [col.reshape(B * NUM_COLOR_PLANES, 1, 9, 9),
             aux.unsqueeze(1).expand(-1, NUM_COLOR_PLANES, -1, -1, -1).reshape(B * NUM_COLOR_PLANES, -1, 9, 9)],
            dim=1,
        )
        h = self.per_color(per).view(B, NUM_COLOR_PLANES, -1, 9, 9)
        pooled = torch.cat([h.mean(1), h.amax(1)], dim=1)
        x = self.fuse(torch.cat([pooled, self.aux_conv(aux)], dim=1))
        return F.relu(self.bn(x) + self.pos)


class FiLMResBlock(nn.Module):
    def __init__(self, C: int, gpool: bool):
        super().__init__()
        self.c1 = nn.Conv2d(C, C, 3, padding=1, bias=False)
        self.b1 = nn.BatchNorm2d(C)
        self.c2 = nn.Conv2d(C, C, 3, padding=1, bias=False)
        self.b2 = nn.BatchNorm2d(C)
        self.film = nn.Linear(NUM_GLOBALS, 2 * C)
        nn.init.zeros_(self.film.weight)
        nn.init.zeros_(self.film.bias)
        self.gpool = nn.Linear(2 * C, C) if gpool else None

    def forward(self, x: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.b1(self.c1(x)))
        if self.gpool is not None:  # KataGo global-pooling bias: whole-board context each block
            pooled = torch.cat([h.mean((2, 3)), h.amax((2, 3))], dim=1)
            h = h + self.gpool(pooled)[:, :, None, None]
        h = self.b2(self.c2(h))
        gamma, beta = self.film(g).chunk(2, dim=1)
        h = h * (1.0 + gamma[:, :, None, None]) + beta[:, :, None, None]
        return F.relu(x + h)


class AxialAttention(nn.Module):
    """Row attention then column attention: one hop covers a full striped-candy laser line."""

    def __init__(self, C: int, heads: int):
        super().__init__()
        self.n1 = nn.LayerNorm(C)
        self.row = nn.MultiheadAttention(C, heads, batch_first=True)
        self.n2 = nn.LayerNorm(C)
        self.col = nn.MultiheadAttention(C, heads, batch_first=True)
        self.n3 = nn.LayerNorm(C)
        self.ffn = nn.Sequential(nn.Linear(C, 2 * C), nn.GELU(), nn.Linear(2 * C, C))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, Hh, Ww = x.shape
        t = x.permute(0, 2, 3, 1)  # B,H,W,C
        r = self.n1(t).reshape(B * Hh, Ww, C)
        t = t + self.row(r, r, r, need_weights=False)[0].reshape(B, Hh, Ww, C)
        c = self.n2(t).permute(0, 2, 1, 3).reshape(B * Ww, Hh, C)
        t = t + self.col(c, c, c, need_weights=False)[0].reshape(B, Ww, Hh, C).permute(0, 2, 1, 3)
        t = t + self.ffn(self.n3(t))
        return t.permute(0, 3, 1, 2).contiguous()


class AlphaCandyZeroNet(nn.Module):
    def __init__(self, preset: str = "cpu", cfg: Optional[ZeroConfig] = None):
        super().__init__()
        self.preset = preset
        self.cfg = cfg or PRESETS[preset]
        C = self.cfg.channels
        self.stem = ColorInvariantStem(self.cfg)
        self.blocks = nn.ModuleList(
            [FiLMResBlock(C, gpool=(i % 3 == 2)) for i in range(self.cfg.blocks)]
        )
        self.attn = nn.ModuleDict(
            {str(i): AxialAttention(C, self.cfg.heads) for i in self.cfg.attn_after if i < self.cfg.blocks}
        )
        E = self.cfg.edge_dim
        # Policy: bilinear symmetric edge head over the 144 adjacent swaps
        self.edge_proj = nn.Conv2d(C, E, 1)
        self.edge_mlp = nn.Sequential(nn.Linear(3 * E, E), nn.ReLU(inplace=True), nn.Linear(E, 1))
        self.action_bias = nn.Parameter(torch.zeros(NUM_ACTIONS))
        self.register_buffer("U", torch.as_tensor(ACTION_U), persistent=False)
        self.register_buffer("V", torch.as_tensor(ACTION_V), persistent=False)
        # Value + auxiliary heads
        self.value_mlp = nn.Sequential(
            nn.Linear(2 * C + NUM_GLOBALS, 256), nn.ReLU(inplace=True), nn.Linear(256, 1 + NUM_FINAL_BINS)
        )
        self.jelly_head = nn.Conv2d(C, 1, 1)

    def forward(
        self, planes: torch.Tensor, globals_: torch.Tensor, legal_mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        x = self.stem(planes)
        for i, blk in enumerate(self.blocks):
            x = blk(x, globals_)
            if str(i) in self.attn:
                x = self.attn[str(i)](x)
        B = x.size(0)
        z = self.edge_proj(x).flatten(2).transpose(1, 2)  # B,81,E
        zu, zv = z[:, self.U], z[:, self.V]
        edge = torch.cat([zu + zv, (zu - zv).abs(), zu * zv], dim=-1)
        logits = self.edge_mlp(edge).squeeze(-1) + self.action_bias
        if legal_mask is not None:
            logits = logits.masked_fill(~legal_mask, -1e9)
        pooled = torch.cat([x.mean((2, 3)), x.amax((2, 3)), globals_], dim=1)
        v = self.value_mlp(pooled)
        return {
            "policy_logits": logits,
            "win_logit": v[:, :1],
            "final_bins": v[:, 1:],
            "jelly_logits": self.jelly_head(x).flatten(1),
        }


def count_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())


if __name__ == "__main__":
    import time

    for name in PRESETS:
        net = AlphaCandyZeroNet(name).eval()
        p = torch.randn(4, NUM_PLANES, 9, 9).clamp(min=0)
        g = torch.rand(4, NUM_GLOBALS)
        with torch.no_grad():
            t = time.perf_counter()
            out = net(p, g)
            dt = (time.perf_counter() - t) * 1000
            # S6 invariance check: permute colour planes
            perm = torch.randperm(6)
            p2 = p.clone()
            p2[:, :6] = p[:, perm]
            d = (net(p2, g)["policy_logits"] - out["policy_logits"]).abs().max().item()
        print(f"{name:6s} params={count_params(net):>11,}  fwd(B=4)={dt:6.1f}ms  "
              f"policy={tuple(out['policy_logits'].shape)}  S6-drift={d:.2e}")
