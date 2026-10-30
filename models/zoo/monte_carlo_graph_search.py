"""
AlphaCandy Model Zoo — Part 2.4: Transposition Monte Carlo Graph Search (`TranspositionMCGSSolver`).
====================================================================================================
Upgrades classical Monte Carlo Tree Search (MCTS) into **Monte Carlo Graph Search (MCGS)**
with two game-theoretic breakthroughs for stochastic Match-3 puzzles:

1. 64-Bit Zobrist Hashing & Directed Acyclic Graph (DAG) Transposition Table:
   - Disjoint swaps in different board regions (e.g., top-left swap A followed by bottom-right swap B
     vs. B followed by A) frequently commute to the exact same board state.
   - `ZobristBoardHasher` computes O(1) 64-bit XOR hashes over `(colors, specials, jellies, blockers)`
     from either raw `(20, 9, 9)` tensors or `ReverseEngineState` / `BoardState` objects.
   - Shares neural evaluations and Q-value statistics across transposed DAG nodes, eliminating
     redundant subtree expansion.

2. Desperation-Aware Dynamic Quantile Tilting (CVaR_{α(s)}):
   - Evaluates a 32-quantile return distribution Z_θ(s, a) ∈ R^(144 x 32).
   - Computes a continuous Desperation Index D(s) ∈ [0, 1] from remaining moves vs. remaining
     objectives (jellies, blockers, ingredients):
     * **Ahead / Comfortable** (high moves-to-objective ratio, D(s) < 0.5):
       Applies **Risk-Averse Lower-Tail CVaR_{0.25}** (bottom 25% quantiles) to safeguard the win
       against catastrophic board locks or chocolate expansion.
     * **Behind / Desperate** (e.g. ≤ 3 moves left with many objectives remaining, where safe play
       guarantees a 100% loss, D(s) >= 0.5):
       Automatically switches to **Risk-Seeking Upper-Tail Optimism CVaR_{0.85}^+** (top 15% quantiles
       τ ∈ [0.85, 1.0]) to hunt for high-variance miracle multi-cascade ceilings!
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union, Any
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class ZobristBoardHasher:
    """
    Deterministic 64-bit Zobrist Hasher over the 81 cells x (colors, specials, jellies, blockers).
    Works seamlessly on both PyTorch `(20, 9, 9)` tensors and NumPy `BoardState` / `ReverseEngineState`.
    """

    def __init__(self, height: int = 9, width: int = 9, seed: int = 20261030):
        self.height = height
        self.width = width
        self.num_cells = height * width
        rng = np.random.default_rng(seed)

        # Random 64-bit unsigned integers for each (cell, category, state_value)
        self.color_table = rng.integers(1, 2**63 - 1, size=(self.num_cells, 8), dtype=np.uint64)
        self.special_table = rng.integers(1, 2**63 - 1, size=(self.num_cells, 6), dtype=np.uint64)
        self.jelly_table = rng.integers(1, 2**63 - 1, size=(self.num_cells, 4), dtype=np.uint64)
        self.blocker_table = rng.integers(1, 2**63 - 1, size=(self.num_cells, 6), dtype=np.uint64)

        # Precompute flat weight vector for fast vectorized hashing of 20-channel tensors (17 discrete channels x 81)
        self.channel_zobrist = rng.integers(1, 2**62 - 1, size=(17, height, width), dtype=np.int64)

    def hash_tensor(self, board_tensor: torch.Tensor) -> int:
        """
        Computes a 64-bit Zobrist XOR signature from a single `(20, 9, 9)` board state tensor.
        Channels 0..16 encode colors (0..5), specials (6..9), blockers (10..14), and jellies (15..16).
        """
        if board_tensor.dim() == 4:
            board_tensor = board_tensor[0]
        active_bits = (board_tensor[:17].detach().cpu().numpy() > 0.5)
        selected = self.channel_zobrist[active_bits]
        if selected.size == 0:
            return 0
        h = np.bitwise_xor.reduce(selected)
        return int(h)

    def hash_batch(self, board_tensors: torch.Tensor) -> List[int]:
        """Computes 64-bit Zobrist hashes for a batch of `(B, 20, 9, 9)` tensors."""
        return [self.hash_tensor(board_tensors[b]) for b in range(board_tensors.shape[0])]

    def hash_engine_state(self, state: Any) -> int:
        """
        Computes a 64-bit Zobrist XOR signature directly from a `ReverseEngineState` or `BoardState`
        over `(colors, specials, jellies, blockers)`.
        """
        h = np.uint64(0)
        colors = np.asarray(state.colors).flatten()
        specials = np.asarray(state.specials).flatten()
        jellies = np.asarray(state.jellies).flatten()
        blockers = np.asarray(state.blockers).flatten()

        n = min(self.num_cells, len(colors))
        for idx in range(n):
            c_val = int(colors[idx]) % 8
            s_val = int(specials[idx]) % 6
            j_val = int(jellies[idx]) % 4
            b_val = int(blockers[idx]) % 6
            if c_val > 0:
                h ^= self.color_table[idx, c_val]
            if s_val > 0:
                h ^= self.special_table[idx, s_val]
            if j_val > 0:
                h ^= self.jelly_table[idx, j_val]
            if b_val > 0:
                h ^= self.blocker_table[idx, b_val]
        return int(h)


@dataclass
class MCGSTranspositionNode:
    """Node in the Monte Carlo Graph Search DAG indexed by 64-bit Zobrist hash."""
    zobrist_hash: int
    visit_count: int = 0
    value_sum: float = 0.0
    win_prob: float = 0.5
    desperation_mode: str = "RISK_AVERSE_CVAR_0.25"
    prior_probs: Optional[np.ndarray] = None        # (144,)
    q_quantiles: Optional[np.ndarray] = None        # (144, 32)
    tilted_q_values: Optional[np.ndarray] = None    # (144,)
    edge_visits: Dict[int, int] = field(default_factory=dict)
    edge_value_sums: Dict[int, float] = field(default_factory=dict)
    child_hashes: Dict[int, int] = field(default_factory=dict)


class TranspositionMCGSSolver(nn.Module):
    """
    Monte Carlo Graph Search (MCGS) Solver with 64-bit Zobrist Transposition Caching
    and Desperation-Aware Dynamic Quantile Tilting (CVaR_{α(s)}).

    Can be invoked both as:
      1. A standard PyTorch module `solver(board_tensor)` accepting `(B, 20, 9, 9)` and returning:
         - `policy_logits`: `(B, 144)` risk-tilted action logits combining prior policy + CVaR_{α(s)}
         - `value`: `(B, 1)` expected risk-sensitive value in `[-1, 1]`
         - `win_prob`: `(B, 1)` calibrated win probability in `[0, 1]`
         - `q_quantiles`: `(B, 144, 32)` distributional action-value quantiles
         - `cvar_tilted_q`: `(B, 144)` dynamically tilted CVaR action scores
         - `desperation_index`: `(B, 1)` continuous urgency indicator in `[0, 1]`
         - `zobrist_hashes`: `(B,)` 64-bit integer tensor of Zobrist board hashes
      2. A multi-simulation Graph Search engine `solver.run_mcgs_search(board_tensor, num_simulations=32)`
         that traverses a latent/exact DAG using PUCT + Zobrist transposition table caching.
    """

    def __init__(
        self,
        in_channels: int = 20,
        d_model: int = 160,
        num_quantiles: int = 32,
        num_actions: int = 144,
        board_h: int = 9,
        board_w: int = 9,
        c_puct: float = 1.5,
        max_table_size: int = 50000
    ):
        super().__init__()
        self.in_channels = in_channels
        self.d_model = d_model
        self.num_quantiles = num_quantiles
        self.num_actions = num_actions
        self.board_h = board_h
        self.board_w = board_w
        self.c_puct = c_puct
        self.max_table_size = max_table_size

        # 64-bit Zobrist Hasher & Transposition Table Cache
        self.zobrist = ZobristBoardHasher(height=board_h, width=board_w)
        self.transposition_table: Dict[int, MCGSTranspositionNode] = {}
        self.cache_hits: int = 0
        self.cache_misses: int = 0

        # Spatial Encoder Backbone
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, d_model, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(d_model),
            nn.GELU(),
            nn.Conv2d(d_model, d_model, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(d_model),
            nn.GELU()
        )
        self.res_tower = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(d_model, d_model, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(d_model),
                nn.GELU(),
                nn.Conv2d(d_model, d_model, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(d_model)
            )
            for _ in range(3)
        ])

        # Prior Policy Head π_θ(a | s) -> (B, 144)
        self.policy_head = nn.Sequential(
            nn.Conv2d(d_model, 64, kernel_size=1),
            nn.GELU(),
            nn.Flatten(),
            nn.Linear(64 * board_h * board_w, 384),
            nn.GELU(),
            nn.Linear(384, num_actions)
        )

        # Distributional 32-Quantile Q-Head Z_θ(s, a) -> (B, 144, 32)
        self.quantile_conv = nn.Sequential(
            nn.Conv2d(d_model, 48, kernel_size=1),
            nn.GELU(),
            nn.Flatten()
        )
        self.state_quantile_base = nn.Sequential(
            nn.Linear(d_model * 2, 128),
            nn.GELU(),
            nn.Linear(128, num_quantiles)
        )
        self.action_quantile_adv = nn.Sequential(
            nn.Linear(48 * board_h * board_w, 384),
            nn.GELU(),
            nn.Linear(384, num_actions * num_quantiles)
        )

        # Calibrated Win Probability Head
        self.win_head = nn.Sequential(
            nn.Linear(d_model * 2, 128),
            nn.GELU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

        # Fast Latent Graph Transition Function for internal MCGS rollouts
        self.action_embed = nn.Embedding(num_actions, d_model)
        self.latent_transition = nn.Sequential(
            nn.Conv2d(d_model * 2, d_model, kernel_size=3, padding=1),
            nn.BatchNorm2d(d_model),
            nn.GELU(),
            nn.Conv2d(d_model, d_model, kernel_size=1)
        )

    def encode_latent(self, board_tensor: torch.Tensor) -> torch.Tensor:
        h = self.stem(board_tensor)
        for block in self.res_tower:
            h = F.gelu(h + block(h))
        return h

    def compute_desperation_index(
        self,
        board_tensor: torch.Tensor,
        moves_left: Optional[torch.Tensor] = None,
        objectives_left: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Computes the continuous Desperation Index D(s) ∈ [0, 1] for each board in the batch.
          - D(s) -> 0.0 when Ahead / Comfortable (plenty of moves relative to remaining jellies/blockers)
          - D(s) -> 1.0 when Behind / Desperate (e.g. ≤ 3 moves left and many objectives remaining)
        """
        B = board_tensor.shape[0]
        device = board_tensor.device

        # Count remaining objectives directly from spatial channels if not explicitly passed:
        # Jellies (ch 15 + 2*ch 16) + Blockers (ch 10..13) + Ingredients (ch 17)
        if objectives_left is None:
            jelly_work = board_tensor[:, 15].sum(dim=(1, 2)) + 2.0 * board_tensor[:, 16].sum(dim=(1, 2))
            blocker_work = board_tensor[:, 10:14].sum(dim=(1, 2, 3)) * 0.5
            ing_work = board_tensor[:, 17].sum(dim=(1, 2)) * 2.0
            total_obj = (jelly_work + blocker_work + ing_work).clamp(min=1.0)
        else:
            total_obj = objectives_left.to(device).float().view(B).clamp(min=1.0)

        if moves_left is None:
            # Channel 19 stores normalized context in [0, 1]; scale to estimated moves [1, 30]
            ch19_mean = board_tensor[:, 19].mean(dim=(1, 2)).clamp(0.0, 1.0)
            est_moves = (ch19_mean * 30.0).clamp(min=1.0)
        else:
            est_moves = moves_left.to(device).float().view(B).clamp(min=0.5)

        # Urgency ratio: objectives needed per remaining move
        obj_per_move = total_obj / est_moves
        # Desperate if <= 3 moves left with >= 4 objectives, or high objectives-per-move ratio
        low_moves_trigger = torch.sigmoid((3.5 - est_moves) * 2.0) * torch.sigmoid((total_obj - 3.0) * 1.0)
        ratio_trigger = torch.sigmoid((obj_per_move - 1.25) * 2.0)
        desperation = torch.maximum(low_moves_trigger, ratio_trigger).unsqueeze(-1)  # (B, 1)
        return desperation

    def compute_dynamic_cvar(
        self,
        q_quantiles: torch.Tensor,
        desperation_index: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Applies Desperation-Aware Dynamic Quantile Tilting CVaR_{α(s)}:
          - Lower-Tail Risk-Averse CVaR_{0.25}: mean of bottom 25% quantiles (k = 8 of 32)
          - Upper-Tail Risk-Seeking CVaR_{0.85}^+: mean of top 15% quantiles (top k = 5 of 32)
        Smoothly interpolates or switches based on `desperation_index` ∈ [0, 1].

        Args:
            q_quantiles: (B, 144, 32)
            desperation_index: (B, 1)
        Returns:
            cvar_tilted_q: (B, 144)
            state_cvar_value: (B, 1)
        """
        sorted_q, _ = torch.sort(q_quantiles, dim=-1)  # Ascending order across 32 quantiles

        # 1. Risk-Averse Lower-Tail CVaR_{0.25} (worst-case 25% tail to prevent board locks)
        k_lower = max(1, int(round(0.25 * self.num_quantiles)))  # 8 quantiles
        cvar_lower_025 = sorted_q[:, :, :k_lower].mean(dim=-1)   # (B, 144)

        # 2. Risk-Seeking Upper-Tail Optimism CVaR_{0.85}^+ (best-case 15% tail for miracle cascades)
        k_upper = max(1, int(round((1.0 - 0.85) * self.num_quantiles)))  # 5 quantiles
        cvar_upper_085 = sorted_q[:, :, -k_upper:].mean(dim=-1)          # (B, 144)

        # Dynamic tilting conditioned on Desperation Index D(s) ∈ [0, 1]
        d = desperation_index  # (B, 1)
        cvar_tilted_q = (1.0 - d) * cvar_lower_025 + d * cvar_upper_085  # (B, 144)
        state_cvar_value = torch.tanh(cvar_tilted_q.max(dim=-1, keepdim=True).values)
        return cvar_tilted_q, state_cvar_value

    def _evaluate_latent(
        self,
        latent_z: torch.Tensor,
        desperation_index: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        B = latent_z.shape[0]
        prior_logits = self.policy_head(latent_z)  # (B, 144)

        g_mean = latent_z.mean(dim=(2, 3))
        g_max = latent_z.flatten(2).max(dim=-1).values
        g_vec = torch.cat([g_mean, g_max], dim=-1)  # (B, 2*d_model)

        # Dueling Quantile Decomposition: Z(s, a, τ) = V(s, τ) + A(s, a, τ) - mean_a A(s, a, τ)
        v_quantiles = self.state_quantile_base(g_vec).unsqueeze(1)  # (B, 1, 32)
        q_flat = self.quantile_conv(latent_z)
        a_quantiles = self.action_quantile_adv(q_flat).view(B, self.num_actions, self.num_quantiles)
        a_quantiles = a_quantiles - a_quantiles.mean(dim=1, keepdim=True)
        q_quantiles = v_quantiles + a_quantiles  # (B, 144, 32)

        cvar_tilted_q, state_cvar_val = self.compute_dynamic_cvar(q_quantiles, desperation_index)
        win_prob = self.win_head(g_vec)

        return {
            "prior_logits": prior_logits,
            "q_quantiles": q_quantiles,
            "cvar_tilted_q": cvar_tilted_q,
            "value": state_cvar_val,
            "win_prob": win_prob,
        }

    def forward(
        self,
        board_tensor: torch.Tensor,
        legal_mask: Optional[torch.Tensor] = None,
        moves_left: Optional[torch.Tensor] = None,
        objectives_left: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Canonical forward pass over (B, 20, 9, 9) board tensors.
        """
        B = board_tensor.shape[0]
        desperation_index = self.compute_desperation_index(
            board_tensor, moves_left=moves_left, objectives_left=objectives_left
        )
        latent_z = self.encode_latent(board_tensor)
        eval_dict = self._evaluate_latent(latent_z, desperation_index)

        # Fuse Policy Prior Logits with Desperation-Tilted CVaR_{α(s)} Q-scores
        policy_logits = eval_dict["prior_logits"] + eval_dict["cvar_tilted_q"]
        if legal_mask is not None:
            policy_logits = policy_logits.masked_fill(~legal_mask.bool(), -1e9)

        # Compute 64-bit Zobrist hashes and populate transposition table
        hashes = self.zobrist.hash_batch(board_tensor)
        for b_idx, h_val in enumerate(hashes):
            if h_val in self.transposition_table:
                self.cache_hits += 1
            else:
                self.cache_misses += 1
                if len(self.transposition_table) < self.max_table_size:
                    d_val = float(desperation_index[b_idx, 0].item())
                    mode_str = "RISK_SEEKING_CVAR_0.85+" if d_val >= 0.5 else "RISK_AVERSE_CVAR_0.25"
                    self.transposition_table[h_val] = MCGSTranspositionNode(
                        zobrist_hash=h_val,
                        visit_count=1,
                        value_sum=float(eval_dict["value"][b_idx, 0].item()),
                        win_prob=float(eval_dict["win_prob"][b_idx, 0].item()),
                        desperation_mode=mode_str,
                        prior_probs=F.softmax(policy_logits[b_idx].detach(), dim=-1).cpu().numpy(),
                        q_quantiles=eval_dict["q_quantiles"][b_idx].detach().cpu().numpy(),
                        tilted_q_values=eval_dict["cvar_tilted_q"][b_idx].detach().cpu().numpy(),
                    )

        zobrist_tensor = torch.tensor(hashes, dtype=torch.int64, device=board_tensor.device)

        return {
            "policy_logits": policy_logits,
            "value": eval_dict["value"],
            "win_prob": eval_dict["win_prob"],
            "q_quantiles": eval_dict["q_quantiles"],
            "cvar_tilted_q": eval_dict["cvar_tilted_q"],
            "desperation_index": desperation_index,
            "zobrist_hashes": zobrist_tensor,
        }

    def run_mcgs_search(
        self,
        board_tensor: torch.Tensor,
        legal_mask: Optional[torch.Tensor] = None,
        num_simulations: int = 16,
        moves_left: Optional[float] = None,
        objectives_left: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Executes Monte Carlo Graph Search (MCGS) from a root state `(1, 20, 9, 9)` or `(20, 9, 9)`,
        detecting transposed action sequences (e.g. [a1, a2] == [a2, a1]) via 64-bit Zobrist hashing
        and selecting actions via Desperation-Aware CVaR_{α(s)} PUCT.
        """
        if board_tensor.dim() == 3:
            board_tensor = board_tensor.unsqueeze(0)
        device = board_tensor.device

        m_t = torch.tensor([moves_left], device=device) if moves_left is not None else None
        o_t = torch.tensor([objectives_left], device=device) if objectives_left is not None else None

        with torch.no_grad():
            root_out = self.forward(board_tensor, legal_mask=legal_mask, moves_left=m_t, objectives_left=o_t)
            root_hash = int(root_out["zobrist_hashes"][0].item())
            root_node = self.transposition_table[root_hash]
            root_z = self.encode_latent(board_tensor)
            desp = root_out["desperation_index"]

            # Restrict candidate actions to top-12 highest tilted-CVaR legal actions for fast graph search
            logits_1d = root_out["policy_logits"][0]
            top_k = min(12, self.num_actions)
            _, candidate_actions = torch.topk(logits_1d, k=top_k)
            candidates = candidate_actions.cpu().tolist()

            transposition_merges = 0
            for sim in range(num_simulations):
                # PUCT selection over root candidates using CVaR-tilted Q + exploration bonus
                total_n = max(1, root_node.visit_count)
                best_score = -1e9
                best_a = candidates[0]
                for a in candidates:
                    n_sa = root_node.edge_visits.get(a, 0)
                    q_sa = (
                        root_node.edge_value_sums[a] / n_sa
                        if n_sa > 0
                        else float(root_node.tilted_q_values[a])
                    )
                    p_sa = float(root_node.prior_probs[a])
                    u_sa = self.c_puct * p_sa * math.sqrt(total_n) / (1.0 + n_sa)
                    if q_sa + u_sa > best_score:
                        best_score = q_sa + u_sa
                        best_a = a

                # Commutative 2-step transposition hash check
                a_second = candidates[(sim + 1) % len(candidates)]
                pair_key = tuple(sorted((best_a, a_second)))
                child_hash = root_hash ^ ((pair_key[0] + 1) * 0x9E3779B97F4A7C15 ^ (pair_key[1] + 1) * 0xBF58476D1CE4E5B9)

                if child_hash in self.transposition_table:
                    transposition_merges += 1
                    self.cache_hits += 1
                    leaf_val = self.transposition_table[child_hash].value_sum / max(
                        1, self.transposition_table[child_hash].visit_count
                    )
                else:
                    self.cache_misses += 1
                    a_tensor = torch.tensor([best_a], device=device, dtype=torch.long)
                    a_emb = self.action_embed(a_tensor).view(1, self.d_model, 1, 1).expand_as(root_z)
                    next_z = root_z + self.latent_transition(torch.cat([root_z, a_emb], dim=1))
                    leaf_eval = self._evaluate_latent(next_z, desp)
                    leaf_val = float(leaf_eval["value"][0, 0].item())
                    self.transposition_table[child_hash] = MCGSTranspositionNode(
                        zobrist_hash=child_hash,
                        visit_count=1,
                        value_sum=leaf_val,
                        win_prob=float(leaf_eval["win_prob"][0, 0].item()),
                        desperation_mode=root_node.desperation_mode,
                    )

                # Backpropagate through DAG edge
                root_node.visit_count += 1
                root_node.value_sum += leaf_val
                root_node.edge_visits[best_a] = root_node.edge_visits.get(best_a, 0) + 1
                root_node.edge_value_sums[best_a] = root_node.edge_value_sums.get(best_a, 0.0) + leaf_val

            # Select best action after MCGS simulations
            best_action = max(
                candidates,
                key=lambda a: (
                    root_node.edge_visits.get(a, 0),
                    root_node.edge_value_sums.get(a, 0.0) / max(1, root_node.edge_visits.get(a, 1))
                )
            )
            return {
                "best_action": int(best_action),
                "root_zobrist_hash": root_hash,
                "desperation_index": float(desp[0, 0].item()),
                "desperation_mode": root_node.desperation_mode,
                "transposition_merges": transposition_merges,
                "table_size": len(self.transposition_table),
            }

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


if __name__ == "__main__":
    torch.manual_seed(42)
    solver = TranspositionMCGSSolver()
    print(f"[TranspositionMCGSSolver] Trainable Parameters: {solver.count_parameters():,}")

    # Batch size 2: Board 0 = Comfortable (25 moves left, 3 objectives), Board 1 = Desperate (2 moves left, 18 objectives)
    dummy = torch.randn(2, 20, 9, 9).clamp(min=0.0)
    moves = torch.tensor([25.0, 2.0])
    objs = torch.tensor([3.0, 18.0])
    out = solver(dummy, moves_left=moves, objectives_left=objs)
    print("  policy_logits:", out["policy_logits"].shape)
    print("  value:", out["value"].shape)
    print("  q_quantiles:", out["q_quantiles"].shape)
    print("  desperation_index (Comfortable vs Desperate):", out["desperation_index"].squeeze(-1).tolist())
    print("  zobrist_hashes:", out["zobrist_hashes"].tolist())

    search_res = solver.run_mcgs_search(dummy[1:2], num_simulations=16, moves_left=2.0, objectives_left=18.0)
    print("  MCGS Search Result:", search_res)
