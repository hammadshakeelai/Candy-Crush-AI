"""
AlphaCandy Model Zoo — Part 2.2: Feudal Hierarchical Option-Critic Architecture.
================================================================================
Two-Level Hierarchical Reinforcement Learning (`HierarchicalSubgoalCandyNet`) combining
Feudal Networks (Vezhnevets et al., 2017) with the Option-Critic Architecture (Bacon et al., 2017).

1. High-Level Strategic Manager Network:
   Evaluates the global 20-channel board state and selects probability weights over
   6 Strategic Options (Subgoals) ω ∈ {0..5}:
     0: MANUFACTURE_SPECIAL      — Create Striped / Wrapped / Color Bomb
     1: ALIGN_AND_DETONATE_COMBO — Pair two adjacent specials for board-wiping combos
     2: SUPPRESS_CHOCOLATE_MIXER — Neutralize expanding chocolate/mixer hazards immediately
     3: UNCAP_DISPENSER_PORTAL   — Clear top-row / portal-choking blockers to restore flow
     4: SNIPE_CORNER_EDGE_JELLY  — Clear low-degree-of-freedom corner & boundary jellies
     5: BOTTOM_CASCADE_SHAKEUP   — Trigger high-entropy bottom-row matches for cascades

   The Manager also predicts:
     - Option Q-values Q_Ω(s, ω) ∈ R^(B, 6)
     - Option Termination Probabilities β_ω(s) ∈ [0, 1]^(B, 6)
     - A 9x9 Spatial Subgoal Attention Heatmap G(r, c) ∈ [0, 1]^(B, 9, 9) highlighting the
       exact board coordinates where the Worker should focus its tactical swap.

2. Low-Level Tactical Worker Network:
   Conditioned on the Manager's Option Embedding e_ω via FiLM (Feature-wise Linear Modulation)
   and gated by the spatial subgoal heatmap G(r, c), produces both:
     - Option-conditional policies π_W(a | s, ω, G) ∈ R^(B, 6, 144)
     - Marginalized / selected canonical 144-action swap logits π(a | s) ∈ R^(B, 144)
"""

from enum import IntEnum
from typing import Dict, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class StrategicOption(IntEnum):
    MANUFACTURE_SPECIAL = 0
    ALIGN_AND_DETONATE_COMBO = 1
    SUPPRESS_CHOCOLATE_MIXER = 2
    UNCAP_DISPENSER_PORTAL = 3
    SNIPE_CORNER_EDGE_JELLY = 4
    BOTTOM_CASCADE_SHAKEUP = 5


OPTION_NAMES: Tuple[str, ...] = (
    "MANUFACTURE_SPECIAL",
    "ALIGN_AND_DETONATE_COMBO",
    "SUPPRESS_CHOCOLATE_MIXER",
    "UNCAP_DISPENSER_PORTAL",
    "SNIPE_CORNER_EDGE_JELLY",
    "BOTTOM_CASCADE_SHAKEUP",
)


class ResidualSpatialBlock(nn.Module):
    """Residual convolutional block with Squeeze-and-Excitation channel recalibration."""

    def __init__(self, channels: int = 160):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, channels // 4),
            nn.GELU(),
            nn.Linear(channels // 4, channels),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.gelu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        scale = self.se(out).unsqueeze(-1).unsqueeze(-1)
        return F.gelu(x + out * scale)


class HighLevelManagerNet(nn.Module):
    """
    Feudal Manager Network.
    Evaluates macro-board topology and outputs:
      - option_logits / option_probs: (B, 6) distribution over the 6 Strategic Options
      - option_q_values: (B, 6) expected macro-return per option
      - option_terminations: (B, 6) termination probabilities β_ω(s) ∈ [0, 1]
      - per_option_heatmaps: (B, 6, 9, 9) spatial subgoal masks in [0, 1]
      - subgoal_heatmap: (B, 9, 9) synthesized spatial subgoal attention G(r, c) ∈ [0, 1]
      - option_context: (B, d_option) continuous goal embedding passed to Worker
    """

    def __init__(
        self,
        d_Backbone: int = 160,
        num_options: int = 6,
        d_option: int = 128,
        board_h: int = 9,
        board_w: int = 9
    ):
        super().__init__()
        self.num_options = num_options
        self.d_option = d_option
        self.board_h = board_h
        self.board_w = board_w

        # Learnable codebook of 6 Strategic Option embeddings
        self.option_codebook = nn.Parameter(torch.randn(num_options, d_option) * 0.05)

        # Global macro-state compressor
        self.macro_pool = nn.Sequential(
            nn.Conv2d(d_Backbone, d_Backbone, kernel_size=3, padding=1),
            nn.BatchNorm2d(d_Backbone),
            nn.GELU()
        )

        # Domain-aware inductive prior extractor from raw 20-channel board
        # Summarizes special count, combo adjacency, chocolate presence, top-row blockers, corner jellies, bottom entropy
        self.inductive_prior_proj = nn.Sequential(
            nn.Linear(6, 32),
            nn.GELU(),
            nn.Linear(32, num_options)
        )

        # Option Policy, Option Q-Value, and Termination β heads
        self.option_policy_head = nn.Sequential(
            nn.Linear(d_Backbone * 2, 128),
            nn.GELU(),
            nn.Linear(128, num_options)
        )
        self.option_q_head = nn.Sequential(
            nn.Linear(d_Backbone * 2, 128),
            nn.GELU(),
            nn.Linear(128, num_options)
        )
        self.termination_head = nn.Sequential(
            nn.Linear(d_Backbone * 2, 64),
            nn.GELU(),
            nn.Linear(64, num_options),
            nn.Sigmoid()
        )

        # Spatial Subgoal Heatmap Generator: outputs 6 option-specific 9x9 heatmaps in [0, 1]
        self.spatial_subgoal_conv = nn.Sequential(
            nn.Conv2d(d_Backbone, 96, kernel_size=3, padding=1),
            nn.BatchNorm2d(96),
            nn.GELU(),
            nn.Conv2d(96, num_options, kernel_size=1),
            nn.Sigmoid()
        )

    def _compute_inductive_priors(self, board_tensor: torch.Tensor) -> torch.Tensor:
        """
        Extracts 6 domain-grounded features aligned with the 6 Strategic Options to give
        the Manager strong inductive bias even before RL fine-tuning:
          0: Regular color density (potential to manufacture specials)
          1: Existing Special Candy count (channels 6..9) for combos
          2: Chocolate / hazard density (channel 10 & 13)
          3: Top-3 row blocker / portal congestion (channels 10..14 in rows 0..2)
          4: Corner & outer-rim jelly density (channels 15..16 on border cells)
          5: Bottom-3 row active candy density (rows 6..8)
        """
        B = board_tensor.shape[0]
        color_density = board_tensor[:, 0:6].sum(dim=(1, 2, 3)) / 81.0
        special_count = board_tensor[:, 6:10].sum(dim=(1, 2, 3)) / 10.0
        chocolate_hazard = (board_tensor[:, 10] + board_tensor[:, 13]).sum(dim=(1, 2)) / 10.0
        top_blockers = board_tensor[:, 10:15, :3, :].sum(dim=(1, 2, 3)) / 15.0

        jellies = board_tensor[:, 15] + board_tensor[:, 16]
        border_mask = torch.ones_like(jellies)
        border_mask[:, 1:-1, 1:-1] = 0.0
        corner_edge_jellies = (jellies * border_mask).sum(dim=(1, 2)) / 20.0

        bottom_active = board_tensor[:, 18, 6:, :].sum(dim=(1, 2)) / 27.0

        feats = torch.stack([
            color_density,
            special_count,
            chocolate_hazard,
            top_blockers,
            corner_edge_jellies,
            bottom_active
        ], dim=-1)  # (B, 6)
        return self.inductive_prior_proj(feats)

    def forward(
        self,
        backbone_feat: torch.Tensor,
        board_tensor: torch.Tensor,
        forced_option: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        B, C, H, W = backbone_feat.shape
        macro_feat = self.macro_pool(backbone_feat)

        # Global mean + max spatial pooling
        g_mean = macro_feat.mean(dim=(2, 3))
        g_max = macro_feat.flatten(2).max(dim=-1).values
        g_vec = torch.cat([g_mean, g_max], dim=-1)  # (B, 2*d_Backbone)

        prior_bias = self._compute_inductive_priors(board_tensor)
        option_logits = self.option_policy_head(g_vec) + 0.25 * prior_bias  # (B, 6)
        option_probs = F.softmax(option_logits, dim=-1)                     # (B, 6)
        option_q_values = self.option_q_head(g_vec)                         # (B, 6)
        option_terminations = self.termination_head(g_vec)                  # (B, 6)

        # Per-option spatial subgoal heatmaps: (B, 6, 9, 9) in [0, 1]
        per_option_heatmaps = self.spatial_subgoal_conv(macro_feat)

        if forced_option is not None:
            # Hard conditioning on a specific option index (B,)
            weights = F.one_hot(forced_option.long(), num_classes=self.num_options).float()
        else:
            # Differentiable soft expectation over all 6 strategic options
            weights = option_probs

        # Synthesize composite 9x9 spatial subgoal heatmap G(r, c) ∈ [0, 1]^(B, 9, 9)
        subgoal_heatmap = torch.einsum("bo,bohw->bhw", weights, per_option_heatmaps).clamp(0.0, 1.0)

        # Synthesize Option Context Vector e_ω ∈ R^(B, d_option)
        option_context = torch.matmul(weights, self.option_codebook)  # (B, d_option)

        return {
            "option_logits": option_logits,
            "option_probs": option_probs,
            "option_q_values": option_q_values,
            "option_terminations": option_terminations,
            "per_option_heatmaps": per_option_heatmaps,
            "subgoal_heatmap": subgoal_heatmap,
            "option_context": option_context,
        }


class LowLevelWorkerNet(nn.Module):
    """
    Low-Level Tactical Worker Network.
    Conditioned on the Manager's Option Embedding `option_context` via FiLM
    (Feature-wise Linear Modulation) and spatial subgoal attention heatmap `G(r, c) ∈ [0, 1]^(9x9)`,
    computes both per-option policies π(a | s, ω) and the marginalized 144-action swap logits.
    """

    def __init__(
        self,
        d_backbone: int = 160,
        d_option: int = 128,
        num_options: int = 6,
        num_actions: int = 144,
        board_h: int = 9,
        board_w: int = 9
    ):
        super().__init__()
        self.d_backbone = d_backbone
        self.d_option = d_option
        self.num_options = num_options
        self.num_actions = num_actions
        self.board_h = board_h
        self.board_w = board_w

        # FiLM generator: maps option embedding e_ω -> (gamma, beta) channel modulation
        self.film_generator = nn.Linear(d_option, d_backbone * 2)

        # Spatial Subgoal Heatmap encoder (fuses G(r, c) with modulated features)
        self.heatmap_gate_conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(32, d_backbone, kernel_size=1),
            nn.Sigmoid()
        )

        self.worker_refine = nn.Sequential(
            ResidualSpatialBlock(d_backbone),
            ResidualSpatialBlock(d_backbone),
            nn.Conv2d(d_backbone, 96, kernel_size=1),
            nn.GELU()
        )

        # Action heads:
        # 1) Direct conditioned Worker head -> (B, 144)
        self.conditioned_action_head = nn.Sequential(
            nn.Linear(96 * board_h * board_w + d_option, 384),
            nn.GELU(),
            nn.Linear(384, num_actions)
        )

        # 2) Multi-Option ensemble head -> (B, 6, 144) so we can inspect every option's tactical preference
        self.multi_option_action_head = nn.Linear(96 * board_h * board_w, num_options * num_actions)

    def forward(
        self,
        backbone_feat: torch.Tensor,
        option_context: torch.Tensor,
        subgoal_heatmap: torch.Tensor,
        option_probs: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            backbone_feat: (B, d_backbone, 9, 9)
            option_context: (B, d_option)
            subgoal_heatmap: (B, 9, 9) in [0, 1]
            option_probs: (B, 6)
        Returns:
            policy_logits: (B, 144) canonical swap logits
            per_option_policy_logits: (B, 6, 144) swap logits per strategic option
        """
        B, C, H, W = backbone_feat.shape

        # 1. FiLM Modulation from Manager's Option Embedding
        gamma_beta = self.film_generator(option_context)  # (B, 2*C)
        gamma, beta = torch.chunk(gamma_beta, 2, dim=-1)
        gamma = 1.0 + torch.tanh(gamma).unsqueeze(-1).unsqueeze(-1)
        beta = beta.unsqueeze(-1).unsqueeze(-1)
        modulated = gamma * backbone_feat + beta

        # 2. Spatial Subgoal Attention Gating via G(r, c)
        g_map = subgoal_heatmap.unsqueeze(1)  # (B, 1, 9, 9)
        spatial_gate = self.heatmap_gate_conv(g_map)  # (B, C, 9, 9)
        # Residual amplification of subgoal target coordinates
        worker_in = modulated * (1.0 + spatial_gate * g_map)

        refined = self.worker_refine(worker_in).flatten(1)  # (B, 96*81)

        # Per-option policy bank: (B, 6, 144)
        per_option_logits = self.multi_option_action_head(refined).view(B, self.num_options, self.num_actions)

        # Conditioned worker logits + expectation over option bank
        cond_logits = self.conditioned_action_head(torch.cat([refined, option_context], dim=-1))
        marginalized_logits = torch.einsum("bo,boa->ba", option_probs, per_option_logits)
        policy_logits = 0.5 * cond_logits + 0.5 * marginalized_logits

        return policy_logits, per_option_logits


class HierarchicalSubgoalCandyNet(nn.Module):
    """
    Full Feudal / Option-Critic Two-Level Hierarchical Architecture for Candy Crush.

    Inputs:
      - board_tensor: (B, 20, 9, 9) canonical 20-channel board state
      - legal_mask: Optional (B, 144) boolean mask of legal moves
      - forced_option: Optional (B,) integer tensor forcing a specific StrategicOption (0..5)

    Outputs (Dict):
      - policy_logits: (B, 144) canonical swap logits
      - value: (B, 1) expected state value V(s) = Σ_ω π_M(ω|s) Q_Ω(s, ω)
      - win_prob: (B, 1) calibrated win probability in [0, 1]
      - option_probs: (B, 6) Manager probability distribution over the 6 Strategic Options
      - option_logits: (B, 6) unnormalized Manager option logits
      - option_q_values: (B, 6) Option-Critic Q_Ω(s, ω)
      - option_terminations: (B, 6) Option termination probabilities β_ω(s) ∈ [0, 1]
      - subgoal_heatmap: (B, 9, 9) spatial subgoal attention map G(r, c) ∈ [0, 1]^(9x9)
      - per_option_heatmaps: (B, 6, 9, 9) spatial heatmaps per strategic option
      - per_option_policy_logits: (B, 6, 144) Worker action logits conditioned on each option
    """

    def __init__(
        self,
        in_channels: int = 20,
        d_backbone: int = 160,
        num_backbone_blocks: int = 4,
        num_options: int = 6,
        d_option: int = 128,
        num_actions: int = 144
    ):
        super().__init__()
        self.in_channels = in_channels
        self.d_backbone = d_backbone
        self.num_options = num_options
        self.num_actions = num_actions

        # Shared Perceptual Spatial Backbone
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, d_backbone, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(d_backbone),
            nn.GELU()
        )
        self.backbone = nn.Sequential(*[
            ResidualSpatialBlock(d_backbone) for _ in range(num_backbone_blocks)
        ])

        # Level 1: High-Level Strategic Manager
        self.manager = HighLevelManagerNet(
            d_Backbone=d_backbone,
            num_options=num_options,
            d_option=d_option
        )

        # Level 2: Low-Level Tactical Worker
        self.worker = LowLevelWorkerNet(
            d_backbone=d_backbone,
            d_option=d_option,
            num_options=num_options,
            num_actions=num_actions
        )

        # Calibrated Win Probability Head
        self.win_head = nn.Sequential(
            nn.Linear(num_options + d_option, 64),
            nn.GELU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(
        self,
        board_tensor: torch.Tensor,
        legal_mask: Optional[torch.Tensor] = None,
        forced_option: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        # 1. Shared spatial representation
        feat = self.backbone(self.stem(board_tensor))  # (B, d_backbone, 9, 9)

        # 2. High-Level Manager selects Option weights & 9x9 Subgoal Heatmap G(r, c)
        mgr_out = self.manager(feat, board_tensor, forced_option=forced_option)

        # 3. Low-Level Worker conditioned on Manager's Option & Subgoal Heatmap outputs 144 swap logits
        policy_logits, per_option_policy_logits = self.worker(
            backbone_feat=feat,
            option_context=mgr_out["option_context"],
            subgoal_heatmap=mgr_out["subgoal_heatmap"],
            option_probs=mgr_out["option_probs"]
        )

        if legal_mask is not None:
            policy_logits = policy_logits.masked_fill(~legal_mask.bool(), -1e9)

        # 4. Option-Critic State Value V(s) = <π_M(·|s), Q_Ω(s, ·)> squashed to [-1, 1]
        expected_q = (mgr_out["option_probs"] * mgr_out["option_q_values"]).sum(dim=-1, keepdim=True)
        value = torch.tanh(expected_q)
        win_prob = self.win_head(torch.cat([mgr_out["option_q_values"], mgr_out["option_context"]], dim=-1))

        return {
            "policy_logits": policy_logits,
            "value": value,
            "win_prob": win_prob,
            "option_probs": mgr_out["option_probs"],
            "option_logits": mgr_out["option_logits"],
            "option_q_values": mgr_out["option_q_values"],
            "option_terminations": mgr_out["option_terminations"],
            "subgoal_heatmap": mgr_out["subgoal_heatmap"],
            "per_option_heatmaps": mgr_out["per_option_heatmaps"],
            "per_option_policy_logits": per_option_policy_logits,
        }

    def select_hierarchical_move(
        self,
        board_tensor: torch.Tensor,
        legal_mask: torch.Tensor
    ) -> Dict[str, object]:
        """
        Interpretable hierarchical inference: returns the dominant StrategicOption name,
        its probability, peak (r, c) subgoal focus coordinate, and selected action index.
        """
        with torch.no_grad():
            out = self.forward(board_tensor, legal_mask=legal_mask)
            opt_probs = out["option_probs"][0]
            best_opt_idx = int(torch.argmax(opt_probs).item())
            heatmap = out["subgoal_heatmap"][0]
            flat_peak = int(torch.argmax(heatmap).item())
            peak_r, peak_c = flat_peak // 9, flat_peak % 9
            best_action = int(torch.argmax(out["policy_logits"][0]).item())
            return {
                "selected_option_id": best_opt_idx,
                "selected_option_name": OPTION_NAMES[best_opt_idx],
                "option_confidence": float(opt_probs[best_opt_idx].item()),
                "subgoal_peak_coord": (peak_r, peak_c),
                "action": best_action,
                "win_prob": float(out["win_prob"][0].item()),
            }

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


if __name__ == "__main__":
    torch.manual_seed(42)
    net = HierarchicalSubgoalCandyNet()
    print(f"[HierarchicalSubgoalCandyNet] Trainable Parameters: {net.count_parameters():,}")
    dummy = torch.randn(2, 20, 9, 9)
    out = net(dummy)
    print("  policy_logits:", out["policy_logits"].shape)
    print("  value:", out["value"].shape)
    print("  option_probs:", out["option_probs"].shape)
    print("  subgoal_heatmap:", out["subgoal_heatmap"].shape,
          f"range=[{out['subgoal_heatmap'].min().item():.3f}, {out['subgoal_heatmap'].max().item():.3f}]")
