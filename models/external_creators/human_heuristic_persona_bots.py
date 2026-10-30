"""Human Cognitive & Behavioral Persona Heuristic Bots for Candy Crush.

Implements three differentiable / fast PyTorch persona bots capturing distinct
human playstyles for behavioral benchmarking and population diversity:
1. CasualHumanPersonaBot: Top-to-bottom visual scanning bias (runs counter to
   optimal bottom-of-board cascade play) with perceptual noise.
2. ColorBombHoarderPersonaBot: Reluctant to detonate existing specials/Color Bombs,
   strongly prioritizing new special creation while hoarding existing specials.
3. ObjectiveTunnelVisionBot: Hyper-fixates on swaps adjacent to jelly, frosting,
   or blockers while ignoring cascade potential elsewhere on the board.
"""

from typing import Dict, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class _BasePersonaSwapScorer(nn.Module):
    """Shared lightweight swap feature extractor mapping (B, 20, 9, 9) -> 144 swap scores."""

    def __init__(self, in_channels: int = 20, hidden_dim: int = 48) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden_dim, 2, kernel_size=1),
        )
        self.value_fc = nn.Sequential(
            nn.Linear(2 * 81, 1),
            nn.Sigmoid(),
        )
        # Precompute (row, col) coordinates for all 144 actions (72 H-swaps + 72 V-swaps)
        h_rows = torch.arange(9).unsqueeze(1).expand(9, 8).reshape(-1).float()
        v_rows = torch.arange(8).unsqueeze(1).expand(8, 9).reshape(-1).float()
        self.register_buffer("action_rows", torch.cat([h_rows, v_rows], dim=0))  # (144,)

        h_idx_a = (torch.arange(9).unsqueeze(1) * 9 + torch.arange(8).unsqueeze(0)).reshape(-1)
        h_idx_b = h_idx_a + 1
        v_idx_a = (torch.arange(8).unsqueeze(1) * 9 + torch.arange(9).unsqueeze(0)).reshape(-1)
        v_idx_b = v_idx_a + 9
        self.register_buffer("cell_a", torch.cat([h_idx_a, v_idx_a], dim=0))
        self.register_buffer("cell_b", torch.cat([h_idx_b, v_idx_b], dim=0))

    def _pair_sum(self, cell_map: torch.Tensor) -> torch.Tensor:
        flat = cell_map.flatten(1)  # (B, 81)
        return flat[:, self.cell_a] + flat[:, self.cell_b]


class CasualHumanPersonaBot(_BasePersonaSwapScorer):
    """Models a casual human player who scans the board top-to-bottom (row 0 first)."""

    def __init__(self, in_channels: int = 20, top_row_bias: float = 1.25) -> None:
        super().__init__(in_channels=in_channels)
        self.top_row_bias = top_row_bias

    def forward(
        self,
        board_tensor: torch.Tensor,
        legal_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        feat = self.conv(board_tensor)
        base_logits = self._pair_sum(feat[:, 0])
        # Top rows (row 0..2) receive a strong positive visual attention bias
        reading_bias = self.top_row_bias * (1.0 - self.action_rows / 8.0).unsqueeze(0)
        policy_logits = base_logits + reading_bias
        if legal_mask is not None:
            policy_logits = policy_logits.masked_fill(~legal_mask.bool(), -1e9)
        win_prob = self.value_fc(feat.flatten(1))
        return {"policy_logits": policy_logits, "win_prob": win_prob}


class ColorBombHoarderPersonaBot(_BasePersonaSwapScorer):
    """Hoards existing specials/Color Bombs (channels 6..9) and avoids spending them."""

    def __init__(self, in_channels: int = 20, hoard_penalty: float = 3.5) -> None:
        super().__init__(in_channels=in_channels)
        self.hoard_penalty = hoard_penalty

    def forward(
        self,
        board_tensor: torch.Tensor,
        legal_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        feat = self.conv(board_tensor)
        base_logits = self._pair_sum(feat[:, 0])
        # Channels 6..9 encode Striped H/V, Wrapped, and Color Bomb specials
        special_presence = board_tensor[:, 6:10].sum(dim=1)  # (B, 9, 9)
        touches_special = self._pair_sum(special_presence)
        policy_logits = base_logits - self.hoard_penalty * touches_special
        if legal_mask is not None:
            policy_logits = policy_logits.masked_fill(~legal_mask.bool(), -1e9)
        win_prob = self.value_fc(feat.flatten(1))
        return {"policy_logits": policy_logits, "win_prob": win_prob}


class ObjectiveTunnelVisionBot(_BasePersonaSwapScorer):
    """Hyper-fixates on swaps touching jelly/frosting/blockers (channels 10..15)."""

    def __init__(self, in_channels: int = 20, tunnel_bonus: float = 4.0) -> None:
        super().__init__(in_channels=in_channels)
        self.tunnel_bonus = tunnel_bonus

    def forward(
        self,
        board_tensor: torch.Tensor,
        legal_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        feat = self.conv(board_tensor)
        base_logits = self._pair_sum(feat[:, 0])
        # Channels 10..15 represent jelly, frosting, cages, and blocker layers
        objective_map = F.max_pool2d(
            board_tensor[:, 10:16].sum(dim=1, keepdim=True),
            kernel_size=3,
            stride=1,
            padding=1,
        ).squeeze(1)
        objective_proximity = self._pair_sum(objective_map)
        policy_logits = base_logits + self.tunnel_bonus * objective_proximity
        if legal_mask is not None:
            policy_logits = policy_logits.masked_fill(~legal_mask.bool(), -1e9)
        win_prob = self.value_fc(feat.flatten(1))
        return {"policy_logits": policy_logits, "win_prob": win_prob}
