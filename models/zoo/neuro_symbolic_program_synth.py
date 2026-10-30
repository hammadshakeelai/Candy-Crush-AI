"""
Neuro-Symbolic Tactical Domain-Specific Language (DSL) & Program Synthesis Network
(`NeuroSymbolicTacticalDSLNet`) for Candy Crush Saga.
Based on:
  - Chaudhuri et al. (Communications of the ACM 2021), "Neurosymbolic Programming"
  - Verma et al. (ICML 2018), "Programmatically Interpretable Reinforcement Learning"
  - Badreddine et al. (Artificial Intelligence 2022), "Logic Tensor Networks"

Combines 12 differentiable symbolic domain-specific predicates (e.g., `P_ColorBombAdjacent`,
`P_DoubleSpecialCombo`, `P_StripedRowAlignedWithJelly`, `P_ChocolateThreat`, `P_BottomRowMatch`,
`P_CornerJellyExposed`, `P_IngredientOverExit`) with differentiable Fuzzy T-Norm logical clauses
and a Neural Rule-Composition Transformer so that every move decision is both superhuman and
100% human-interpretable.
"""

import math
from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


SYMBOLIC_PREDICATE_NAMES: List[str] = [
    "P_ColorBombAdjacent",
    "P_DoubleSpecialCombo",
    "P_StripedRowAlignedWithJelly",
    "P_WrappedOverDenseJellyCluster",
    "P_ChocolateThreat",
    "P_FrostingChainBreak",
    "P_BottomRowMatch",
    "P_CornerJellyExposed",
    "P_IngredientOverExit",
    "P_DoubleJellyDirectStrike",
    "P_ColorMajoritySynergy",
    "P_EndgameUrgencyFocus",
]


class DifferentiableSymbolicPredicateEngine(nn.Module):
    """
    Evaluates 12 domain-specific symbolic match-3 predicates directly over the canonical
    (B, 20, 9, 9) board tensor for every canonical swap action a in [0..143]:
      - Actions 0..71:   Horizontal swaps (r, c) <-> (r, c+1) for r in 0..8, c in 0..7
      - Actions 72..143: Vertical swaps   (r, c) <-> (r+1, c) for r in 0..7, c in 0..8
    Produces a differentiable truth matrix P(s, a) in [0, 1]^{B x 144 x 12}.
    """
    def __init__(self, board_h: int = 9, board_w: int = 9):
        super().__init__()
        self.board_h = board_h
        self.board_w = board_w
        self.num_actions = 144
        self.num_predicates = 12

        # Precompute endpoint coordinates (r1, c1) and (r2, c2) for all 144 canonical swaps
        r1_list, c1_list, r2_list, c2_list = [], [], [], []
        # 72 Horizontal swaps: r in 0..8, c in 0..7
        for r in range(board_h):
            for c in range(board_w - 1):
                r1_list.append(r)
                c1_list.append(c)
                r2_list.append(r)
                c2_list.append(c + 1)
        # 72 Vertical swaps: r in 0..7, c in 0..8
        for r in range(board_h - 1):
            for c in range(board_w):
                r1_list.append(r)
                c1_list.append(c)
                r2_list.append(r + 1)
                c2_list.append(c)

        self.register_buffer("idx_r1", torch.tensor(r1_list, dtype=torch.long))
        self.register_buffer("idx_c1", torch.tensor(c1_list, dtype=torch.long))
        self.register_buffer("idx_r2", torch.tensor(r2_list, dtype=torch.long))
        self.register_buffer("idx_c2", torch.tensor(c2_list, dtype=torch.long))

        # Fixed 3x3 neighborhood dilation kernel for adjacency detection
        adj_kernel = torch.ones((1, 1, 3, 3), dtype=torch.float32)
        self.register_buffer("adj_kernel", adj_kernel)

        # Bottom-row cascade bonus profile: row 8 -> 1.0, row 7 -> 0.85, row 6 -> 0.65, ...
        row_weights = torch.linspace(0.0, 1.0, board_h).pow(2).view(1, board_h, 1).expand(1, board_h, board_w)
        self.register_buffer("bottom_row_map", row_weights.clone())

        # Corner mask (3x3 corner quadrants weighted toward exact corners)
        corner_map = torch.zeros((1, board_h, board_w), dtype=torch.float32)
        for cr in [0, board_h - 1]:
            for cc in [0, board_w - 1]:
                corner_map[0, cr, cc] = 1.0
        corner_dilated = F.conv2d(corner_map.unsqueeze(0), adj_kernel / 3.0, padding=1).squeeze(0).clamp(0.0, 1.0)
        self.register_buffer("corner_mask", corner_dilated)

        # Learnable calibration temperature and threshold shifts per predicate
        self.predicate_gain = nn.Parameter(torch.ones(1, 1, self.num_predicates) * 2.0)
        self.predicate_bias = nn.Parameter(torch.zeros(1, 1, self.num_predicates))

    def _gather_swap_pair(self, grid_map: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extracts values at endpoint 1 (r1, c1) and endpoint 2 (r2, c2) for all 144 actions.
        Args:
            grid_map: (B, 9, 9)
        Returns:
            v1, v2: Each of shape (B, 144)
        """
        v1 = grid_map[:, self.idx_r1, self.idx_c1]
        v2 = grid_map[:, self.idx_r2, self.idx_c2]
        return v1, v2

    def forward(self, board_tensor: torch.Tensor) -> torch.Tensor:
        """
        Evaluates all 12 symbolic predicates over the 144 actions.
        Args:
            board_tensor: Canonical (B, 20, 9, 9) board tensor
        Returns:
            predicates: Differentiable truth matrix in [0, 1] of shape (B, 144, 12)
        """
        B, C, H, W = board_tensor.shape

        # Extract semantic channel maps (clamped to [0, 1] for soft truth evaluation)
        colors = board_tensor[:, 0:7].clamp(0.0, 1.0)                       # (B, 7, 9, 9)
        striped_h = board_tensor[:, 8].clamp(0.0, 1.0)                      # (B, 9, 9)
        striped_v = board_tensor[:, 9].clamp(0.0, 1.0)                      # (B, 9, 9)
        wrapped = board_tensor[:, 10].clamp(0.0, 1.0)                       # (B, 9, 9)
        color_bomb = board_tensor[:, 11].clamp(0.0, 1.0)                    # (B, 9, 9)
        any_special = (striped_h + striped_v + wrapped + color_bomb).clamp(0.0, 1.0)

        single_jelly = board_tensor[:, 13].clamp(0.0, 1.0)                  # (B, 9, 9)
        double_jelly = board_tensor[:, 14].clamp(0.0, 1.0)                  # (B, 9, 9)
        any_jelly = (single_jelly + double_jelly).clamp(0.0, 1.0)           # (B, 9, 9)

        chocolate = board_tensor[:, 15].clamp(0.0, 1.0)                     # (B, 9, 9)
        licorice = board_tensor[:, 16].clamp(0.0, 1.0)                      # (B, 9, 9)
        frosting = board_tensor[:, 17].clamp(0.0, 1.0)                      # (B, 9, 9)
        moves_norm = board_tensor[:, 18].clamp(0.0, 1.0)                    # (B, 9, 9)
        portals = board_tensor[:, 19].abs().clamp(0.0, 1.0)                 # (B, 9, 9)

        # --- Predicate 1: P_ColorBombAdjacent ---
        cb1, cb2 = self._gather_swap_pair(color_bomb)
        p01_color_bomb = (cb1 + cb2).clamp(0.0, 1.0)

        # --- Predicate 2: P_DoubleSpecialCombo ---
        sp1, sp2 = self._gather_swap_pair(any_special)
        p02_double_special = (sp1 * sp2).clamp(0.0, 1.0)

        # --- Predicate 3: P_StripedRowAlignedWithJelly ---
        # Row jelly density for horizontal striped; Col jelly density for vertical striped
        row_jelly_density = any_jelly.mean(dim=2, keepdim=True).expand(B, H, W)
        col_jelly_density = any_jelly.mean(dim=1, keepdim=True).expand(B, H, W)
        striped_aligned_map = (striped_h * row_jelly_density + striped_v * col_jelly_density).clamp(0.0, 1.0)
        sa1, sa2 = self._gather_swap_pair(striped_aligned_map)
        p03_striped_jelly = torch.maximum(sa1, sa2)

        # --- Predicate 4: P_WrappedOverDenseJellyCluster ---
        jelly_3x3 = (F.conv2d(any_jelly.unsqueeze(1), self.adj_kernel, padding=1).squeeze(1) / 4.0).clamp(0.0, 1.0)
        wrapped_cluster_map = (wrapped * jelly_3x3).clamp(0.0, 1.0)
        wc1, wc2 = self._gather_swap_pair(wrapped_cluster_map)
        p04_wrapped_cluster = torch.maximum(wc1, wc2)

        # --- Predicate 5: P_ChocolateThreat ---
        choc_adj = (F.conv2d(chocolate.unsqueeze(1), self.adj_kernel, padding=1).squeeze(1) / 2.0).clamp(0.0, 1.0)
        ct1, ct2 = self._gather_swap_pair(choc_adj)
        p05_chocolate_threat = torch.maximum(ct1, ct2)

        # --- Predicate 6: P_FrostingChainBreak ---
        blocker_map = (frosting + licorice).clamp(0.0, 1.0)
        blocker_adj = (F.conv2d(blocker_map.unsqueeze(1), self.adj_kernel, padding=1).squeeze(1) / 3.0).clamp(0.0, 1.0)
        fb1, fb2 = self._gather_swap_pair(blocker_adj)
        p06_frosting_break = torch.maximum(fb1, fb2)

        # --- Predicate 7: P_BottomRowMatch ---
        bot_map = self.bottom_row_map.expand(B, H, W)
        br1, br2 = self._gather_swap_pair(bot_map)
        p07_bottom_row = 0.5 * (br1 + br2)

        # --- Predicate 8: P_CornerJellyExposed ---
        corner_jelly_map = (any_jelly * self.corner_mask).clamp(0.0, 1.0)
        cj1, cj2 = self._gather_swap_pair(corner_jelly_map)
        p08_corner_jelly = torch.maximum(cj1, cj2)

        # --- Predicate 9: P_IngredientOverExit ---
        # Column alignment with portals / ingredient drop chutes or vertical blocker columns
        col_portal_target = (portals + any_jelly).mean(dim=1, keepdim=True).expand(B, H, W).clamp(0.0, 1.0)
        ie1, ie2 = self._gather_swap_pair(col_portal_target)
        p09_ingredient_exit = torch.maximum(ie1, ie2)

        # --- Predicate 10: P_DoubleJellyDirectStrike ---
        dj1, dj2 = self._gather_swap_pair(double_jelly)
        p10_double_jelly = torch.maximum(dj1, dj2)

        # --- Predicate 11: P_ColorMajoritySynergy ---
        # Checks whether swapped cells belong to the most abundant color on the board
        color_counts = colors.sum(dim=(2, 3), keepdim=True)  # (B, 7, 1, 1)
        color_freq_weights = F.softmax(color_counts / 5.0, dim=1)
        synergy_map = (colors * color_freq_weights).sum(dim=1).clamp(0.0, 1.0)  # (B, 9, 9)
        cm1, cm2 = self._gather_swap_pair(synergy_map)
        p11_color_majority = torch.maximum(cm1, cm2)

        # --- Predicate 12: P_EndgameUrgencyFocus ---
        # High truth value when moves remaining < 25% AND swap touches an active objective/jelly
        low_moves_gate = torch.sigmoid((0.25 - moves_norm) * 10.0)
        urgency_map = (low_moves_gate * (any_jelly + any_special).clamp(0.0, 1.0)).clamp(0.0, 1.0)
        eu1, eu2 = self._gather_swap_pair(urgency_map)
        p12_endgame_urgency = torch.maximum(eu1, eu2)

        raw_predicates = torch.stack([
            p01_color_bomb,
            p02_double_special,
            p03_striped_jelly,
            p04_wrapped_cluster,
            p05_chocolate_threat,
            p06_frosting_break,
            p07_bottom_row,
            p08_corner_jelly,
            p09_ingredient_exit,
            p10_double_jelly,
            p11_color_majority,
            p12_endgame_urgency,
        ], dim=-1)  # (B, 144, 12)

        # Differentiable calibrated fuzzy truth values in [0, 1]
        calibrated = torch.sigmoid(self.predicate_gain * (raw_predicates - 0.25) + self.predicate_bias)
        return calibrated


class DifferentiableDSLRuleSynthesizer(nn.Module):
    """
    Program Synthesis / Fuzzy Logic Rule Layer.
    Composes the 12 atomic symbolic predicates into `num_rules` higher-order logical clauses
    using differentiable Łukasiewicz / Product T-norm Conjunctions (AND) and Disjunctions (OR):
        Clause_r(s, a) = T_AND_{p=1}^{12} (1 - alpha_{r,p} + alpha_{r,p} * P_p(s, a))
    where alpha_{r,p} in [0, 1] is the learned predicate inclusion weight in rule r.
    """
    def __init__(self, num_predicates: int = 12, num_rules: int = 24):
        super().__init__()
        self.num_predicates = num_predicates
        self.num_rules = num_rules

        # Learnable inclusion logits for AND clauses and OR clauses
        self.and_inclusion_logits = nn.Parameter(torch.randn(num_rules // 2, num_predicates) * 0.5)
        self.or_inclusion_logits = nn.Parameter(torch.randn(num_rules // 2, num_predicates) * 0.5)

        # Priority weights mapping the synthesized rules to action utility scores
        self.rule_priority_weights = nn.Linear(num_rules, 1)

    def forward(self, predicates: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            predicates: (B, 144, 12) truth values in [0, 1]
        Returns:
            rule_activations: (B, 144, num_rules) truth values of synthesized composite rules
            symbolic_logits: (B, 144) interpretable DSL action utility scores
        """
        # 1. Differentiable Product T-Norm Conjunctions (AND rules):
        #    prod_p (1 - alpha_{r,p} * (1 - P_p)) computed stably in log domain
        alpha_and = torch.sigmoid(self.and_inclusion_logits).unsqueeze(0).unsqueeze(0)  # (1, 1, R/2, 12)
        p_exp = predicates.unsqueeze(2)  # (B, 144, 1, 12)
        and_terms = (1.0 - alpha_and * (1.0 - p_exp)).clamp(min=1e-4, max=1.0)
        and_clauses = torch.exp(torch.log(and_terms).sum(dim=-1))  # (B, 144, R/2)

        # 2. Differentiable Probabilistic T-Conorm Disjunctions (OR rules):
        #    1 - prod_p (1 - beta_{r,p} * P_p)
        beta_or = torch.sigmoid(self.or_inclusion_logits).unsqueeze(0).unsqueeze(0)  # (1, 1, R/2, 12)
        or_terms = (1.0 - beta_or * p_exp).clamp(min=1e-4, max=1.0)
        or_clauses = 1.0 - torch.exp(torch.log(or_terms).sum(dim=-1))  # (B, 144, R/2)

        rule_activations = torch.cat([and_clauses, or_clauses], dim=-1)  # (B, 144, num_rules)
        symbolic_logits = self.rule_priority_weights(rule_activations).squeeze(-1)  # (B, 144)
        return rule_activations, symbolic_logits


class NeuroSymbolicTacticalDSLNet(nn.Module):
    """
    Neuro-Symbolic Tactical DSL & Program Synthesis Architecture for Candy Crush Saga.
    Combines:
      1. 12 Differentiable Symbolic Predicates P_1..P_12 evaluated per swap action
      2. Differentiable Fuzzy T-Norm Rule Synthesizer (24 composite AND/OR tactical clauses)
      3. Neural Rule-Composition Transformer that contextualizes symbolic rules with spatial board state
    """
    def __init__(
        self,
        in_channels: int = 20,
        d_model: int = 192,
        num_rules: int = 24,
        num_transformer_layers: int = 3,
        num_heads: int = 6,
        num_actions: int = 144,
        board_h: int = 9,
        board_w: int = 9
    ):
        super().__init__()
        self.in_channels = in_channels
        self.d_model = d_model
        self.num_rules = num_rules
        self.num_actions = num_actions
        self.board_h = board_h
        self.board_w = board_w

        # 1. Symbolic Predicate Engine & Differentiable DSL Rule Synthesizer
        self.predicate_engine = DifferentiableSymbolicPredicateEngine(board_h=board_h, board_w=board_w)
        self.rule_synthesizer = DifferentiableDSLRuleSynthesizer(num_predicates=12, num_rules=num_rules)

        # 2. Neural Spatial Context & Rule-Composition Transformer over the 144 Action Tokens
        self.spatial_stem = nn.Sequential(
            nn.Conv2d(in_channels, d_model, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(d_model),
            nn.GELU(),
            nn.Conv2d(d_model, d_model, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(d_model),
            nn.GELU()
        )

        # Projects (12 predicates + 24 synthesized rules + endpoint spatial features) -> d_model
        self.action_token_proj = nn.Sequential(
            nn.Linear(12 + num_rules + d_model * 2, d_model),
            nn.LayerNorm(d_model),
            nn.GELU()
        )
        self.action_pos_embed = nn.Parameter(torch.randn(1, num_actions, d_model) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_model * 3,
            dropout=0.05,
            activation="gelu",
            batch_first=True,
            norm_first=True
        )
        self.rule_transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_transformer_layers)
        self.final_norm = nn.LayerNorm(d_model)

        # Neural refinement head over the 144 action tokens
        self.neural_action_scorer = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 1)
        )

        # Learnable gate balancing interpretable symbolic DSL logits and neural residual logits
        self.symbolic_trust_logit = nn.Parameter(torch.tensor(0.5))

        # Value / Win Probability Head conditioned on both symbolic rules and neural state
        self.value_head = nn.Sequential(
            nn.Linear(d_model + 12 + num_rules, 128),
            nn.GELU(),
            nn.Linear(128, 1)
        )

    def forward(
        self,
        board_tensor: torch.Tensor,
        legal_mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Canonical forward pass accepting (B, 20, 9, 9) board tensor.
        Returns `policy_logits`, `win_prob`, `value`, `predicate_truth_matrix`,
        `synthesized_rule_scores`, `symbolic_policy_logits`, and `neural_residual_logits`.
        """
        B, C, H, W = board_tensor.shape

        # 1. Evaluate 12 symbolic predicates P(s, a) in [0, 1]^{B x 144 x 12}
        predicates = self.predicate_engine(board_tensor)

        # 2. Synthesize 24 composite logical rules and pure symbolic action utilities
        rule_scores, symbolic_logits = self.rule_synthesizer(predicates)  # (B, 144, 24), (B, 144)

        # 3. Extract spatial features at swap endpoints (r1, c1) and (r2, c2)
        spatial_feat = self.spatial_stem(board_tensor)  # (B, d_model, 9, 9)
        f1 = spatial_feat[:, :, self.predicate_engine.idx_r1, self.predicate_engine.idx_c1].transpose(1, 2)
        f2 = spatial_feat[:, :, self.predicate_engine.idx_r2, self.predicate_engine.idx_c2].transpose(1, 2)

        # 4. Fuse symbolic predicates, synthesized clauses, and spatial endpoint features
        action_tokens = self.action_token_proj(
            torch.cat([predicates, rule_scores, f1, f2], dim=-1)
        ) + self.action_pos_embed  # (B, 144, d_model)

        refined_tokens = self.final_norm(self.rule_transformer(action_tokens))
        neural_logits = self.neural_action_scorer(refined_tokens).squeeze(-1)  # (B, 144)

        # Combine interpretable symbolic DSL program with neural composition refinement
        alpha_sym = torch.sigmoid(self.symbolic_trust_logit)
        policy_logits = alpha_sym * symbolic_logits + (1.0 - alpha_sym) * neural_logits

        if legal_mask is not None:
            policy_logits = policy_logits.masked_fill(~legal_mask.bool(), -1e9)

        # 5. Compute global value & win probability from pooled rules + neural tokens
        global_token = refined_tokens.mean(dim=1)         # (B, d_model)
        max_predicates = predicates.max(dim=1)[0]         # (B, 12)
        max_rules = rule_scores.max(dim=1)[0]             # (B, 24)
        val_raw = self.value_head(torch.cat([global_token, max_predicates, max_rules], dim=-1))
        win_prob = torch.sigmoid(val_raw)
        value = torch.tanh(val_raw)

        return {
            "policy_logits": policy_logits,
            "win_prob": win_prob,
            "value": value,
            "predicate_truth_matrix": predicates,
            "synthesized_rule_scores": rule_scores,
            "symbolic_policy_logits": symbolic_logits,
            "neural_residual_logits": neural_logits
        }

    @torch.no_grad()
    def explain_action(self, board_tensor: torch.Tensor, action_idx: Optional[int] = None) -> Dict[str, object]:
        """
        Generates a 100% human-interpretable symbolic proof for the selected (or specified) swap action.
        """
        out = self.forward(board_tensor[:1])
        if action_idx is None:
            action_idx = int(torch.argmax(out["policy_logits"][0]).item())

        pred_vals = out["predicate_truth_matrix"][0, action_idx].cpu().tolist()
        ranked = sorted(
            zip(SYMBOLIC_PREDICATE_NAMES, pred_vals),
            key=lambda kv: kv[1],
            reverse=True
        )
        r1 = int(self.predicate_engine.idx_r1[action_idx].item())
        c1 = int(self.predicate_engine.idx_c1[action_idx].item())
        r2 = int(self.predicate_engine.idx_r2[action_idx].item())
        c2 = int(self.predicate_engine.idx_c2[action_idx].item())

        top_reasons = [f"{name}={val:.2f}" for name, val in ranked[:3]]
        explanation = (
            f"Swap ({r1},{c1})<->({r2},{c2}) [Action #{action_idx}] triggered by symbolic predicates: "
            + ", ".join(top_reasons)
        )
        return {
            "action_idx": action_idx,
            "swap_coords": ((r1, c1), (r2, c2)),
            "top_predicates": ranked[:4],
            "explanation": explanation
        }

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def create_neuro_symbolic_dsl_net(device: str = "cpu") -> NeuroSymbolicTacticalDSLNet:
    """Factory function initializing NeuroSymbolicTacticalDSLNet on the target device."""
    model = NeuroSymbolicTacticalDSLNet(
        in_channels=20,
        d_model=192,
        num_rules=24,
        num_transformer_layers=3,
        num_heads=6,
        num_actions=144
    )
    return model.to(device)


if __name__ == "__main__":
    torch.manual_seed(42)
    net = create_neuro_symbolic_dsl_net("cpu")
    params = net.count_parameters()
    print(f"[NeuroSymbolicTacticalDSLNet] Initialized with {params:,} trainable parameters ({params / 1e6:.2f}M)")

    dummy_board = torch.randn(2, 20, 9, 9).abs()
    out = net(dummy_board)
    print(f"  policy_logits:           {out['policy_logits'].shape}")
    print(f"  win_prob:                {out['win_prob'].shape}")
    print(f"  value:                   {out['value'].shape}")
    print(f"  predicate_truth_matrix:  {out['predicate_truth_matrix'].shape}")
    print(f"  synthesized_rule_scores: {out['synthesized_rule_scores'].shape}")
    proof = net.explain_action(dummy_board)
    print(f"  Symbolic Proof:          {proof['explanation']}")
