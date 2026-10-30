"""
AlphaCandy Model Zoo — Part 2.1: Wormhole Graph Message Passing Neural Network (MPNN).
======================================================================================
Explicit Heterogeneous Directed Multigraph Message Passing Network over the 81 board cells.

Unlike standard Euclidean CNNs or uniform Grid Transformers, `WormholeGraphNeuralNet`
dynamically constructs a 4-relation directed multigraph G = (V, E_adj, E_gravity, E_portal, E_special_blast)
directly from each 20-channel board tensor:
  1. `E_adj`           : 4-neighbor Euclidean orthogonal grid adjacency ( masked by active grid channel 18 )
  2. `E_gravity`       : Directed downward column drop & cascade flow edges (r -> r+1, r+2)
  3. `E_portal`        : Non-local teleporter wormhole edges connecting portal entrances to exits (channels 16 & 17)
  4. `E_special_blast` : Long-range row/column laser edges from Striped candies (channels 6 & 7)
                         and 3x3 blast radius edges from Wrapped candies (channel 8)

Executes 8 rounds of Edge-Typed Relational Graph Attention Message Passing (RGAT / MPNN),
allowing cells at the opposite end of a teleporter wormhole or a full-row striped laser
to exchange 1-hop messages in a single layer.
"""

import math
from typing import Dict, Optional, Tuple, List
import torch
import torch.nn as nn
import torch.nn.functional as F


class DynamicMultigraphBuilder(nn.Module):
    """
    Dynamically constructs a batched 4-channel relational adjacency tensor A ∈ {0, 1}^(B, 4, 81, 81)
    from the canonical (B, 20, 9, 9) board tensor and optional explicit portal coordinates.
    """

    def __init__(self, height: int = 9, width: int = 9):
        super().__init__()
        self.height = height
        self.width = width
        self.num_nodes = height * width

        # Precompute static base templates for E_adj and E_gravity on 9x9 grid
        e_adj = torch.zeros(self.num_nodes, self.num_nodes, dtype=torch.float32)
        e_grav = torch.zeros(self.num_nodes, self.num_nodes, dtype=torch.float32)

        for r in range(height):
            for c in range(width):
                u = r * width + c
                # 4-neighbor Euclidean adjacency
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < height and 0 <= nc < width:
                        v = nr * width + nc
                        e_adj[u, v] = 1.0

                # Directed downward gravity edges (1-step and 2-step column drop flow)
                for drop in [1, 2]:
                    nr = r + drop
                    if nr < height:
                        v = nr * width + c
                        e_grav[u, v] = 1.0 / float(drop)

                # Precomputed row/col/3x3 masks for fast vectorized special blast lookup
        row_mask = torch.zeros(self.num_nodes, self.num_nodes, dtype=torch.float32)
        col_mask = torch.zeros(self.num_nodes, self.num_nodes, dtype=torch.float32)
        box_mask = torch.zeros(self.num_nodes, self.num_nodes, dtype=torch.float32)

        for r1 in range(height):
            for c1 in range(width):
                u = r1 * width + c1
                for r2 in range(height):
                    for c2 in range(width):
                        v = r2 * width + c2
                        if u == v:
                            continue
                        if r1 == r2:
                            row_mask[u, v] = 1.0
                        if c1 == c2:
                            col_mask[u, v] = 1.0
                        if abs(r1 - r2) <= 1 and abs(c1 - c2) <= 1:
                            box_mask[u, v] = 1.0

        self.register_buffer("base_adj", e_adj)
        self.register_buffer("base_grav", e_grav)
        self.register_buffer("row_mask", row_mask)
        self.register_buffer("col_mask", col_mask)
        self.register_buffer("box_mask", box_mask)

    def forward(
        self,
        board_tensor: torch.Tensor,
        portal_pairs: Optional[List[List[Tuple[Tuple[int, int], Tuple[int, int]]]]] = None
    ) -> torch.Tensor:
        """
        Constructs the 4-type multigraph adjacency matrix of shape (B, 4, 81, 81).
          Edge Type 0: E_adj           (Euclidean 4-neighbor adjacency)
          Edge Type 1: E_gravity       (Directed downward column drop)
          Edge Type 2: E_portal        (Non-local teleporter wormholes from ch 16 & 17)
          Edge Type 3: E_special_blast (Striped H/V lasers from ch 6 & 7 + Wrapped 3x3 from ch 8)
        """
        B, C, H, W = board_tensor.shape
        N = H * W
        device = board_tensor.device

        # Active cell mask (channel 18); fallback to all ones if zero-initialized dummy tensor
        active = board_tensor[:, 18].reshape(B, N)
        has_active = (active.sum(dim=-1, keepdim=True) > 0).float()
        active_mask = active * has_active + (1.0 - has_active)  # (B, N)
        active_pair = (active_mask.unsqueeze(2) * active_mask.unsqueeze(1)).clamp(0.0, 1.0)  # (B, N, N)

        # 1. E_adj: 4-neighbor grid adjacency gated by active cells
        e_adj = self.base_adj.unsqueeze(0).expand(B, N, N) * active_pair

        # 2. E_gravity: Directed downward column drop edges gated by active cells
        e_gravity = self.base_grav.unsqueeze(0).expand(B, N, N) * active_pair

        # 3. E_portal: Non-local wormhole edges connecting portal entrance (ch 16) to exit (ch 17)
        # Also supports explicit portal_pairs if provided
        ch16 = (board_tensor[:, 16].reshape(B, N) > 0.25).float()
        ch17 = (board_tensor[:, 17].reshape(B, N) > 0.25).float()
        # Connect entrance nodes u (ch16) <-> exit nodes v (ch17), plus top-of-column to bottom-of-column wrap fallback
        e_portal = torch.bmm(ch16.unsqueeze(2), ch17.unsqueeze(1)) + torch.bmm(ch17.unsqueeze(2), ch16.unsqueeze(1))
        e_portal = e_portal.clamp(0.0, 1.0)

        if portal_pairs is not None:
            for b_idx, b_portals in enumerate(portal_pairs):
                if b_idx < B and b_portals:
                    for (sr, sc), (dr, dc) in b_portals:
                        if 0 <= sr < H and 0 <= sc < W and 0 <= dr < H and 0 <= dc < W:
                            u = sr * W + sc
                            v = dr * W + dc
                            e_portal[b_idx, u, v] = 1.0
                            e_portal[b_idx, v, u] = 1.0

        # 4. E_special_blast:
        #   - Striped Horizontal (channel 6): broadcasts along entire row
        #   - Striped Vertical (channel 7): broadcasts along entire column
        #   - Wrapped Candy (channel 8): broadcasts across 3x3 blast zone
        striped_h = (board_tensor[:, 6].reshape(B, N) > 0.25).float()  # (B, N)
        striped_v = (board_tensor[:, 7].reshape(B, N) > 0.25).float()  # (B, N)
        wrapped = (board_tensor[:, 8].reshape(B, N) > 0.25).float()    # (B, N)

        blast_h = striped_h.unsqueeze(2) * self.row_mask.unsqueeze(0)
        blast_v = striped_v.unsqueeze(2) * self.col_mask.unsqueeze(0)
        blast_w = wrapped.unsqueeze(2) * self.box_mask.unsqueeze(0)
        e_special_blast = (blast_h + blast_v + blast_w).clamp(0.0, 1.0)

        # Stack all 4 relational graph topologies: (B, 4, 81, 81)
        edge_tensors = torch.stack([e_adj, e_gravity, e_portal, e_special_blast], dim=1)
        return edge_tensors


class EdgeTypedGraphAttentionLayer(nn.Module):
    """
    Single round of Heterogeneous Edge-Typed Graph Attention Message Passing (MPNN).
    Computes relation-specific queries, keys, and messages across the 4 edge types:
      - E_adj (0), E_gravity (1), E_portal (2), E_special_blast (3)
    """

    def __init__(
        self,
        d_model: int = 192,
        num_heads: int = 6,
        num_edge_types: int = 4,
        dropout: float = 0.05
    ):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.num_edge_types = num_edge_types
        self.scale = 1.0 / math.sqrt(self.head_dim)

        self.norm1 = nn.LayerNorm(d_model)

        # Node query and key projections
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)

        # Edge-type specific message value projections (4 distinct relational message transforms)
        self.edge_v_projs = nn.ModuleList([
            nn.Linear(d_model, d_model, bias=False) for _ in range(num_edge_types)
        ])

        # Learnable per-head relation importance weights & attention biases
        self.edge_type_bias = nn.Parameter(torch.zeros(num_edge_types, num_heads, 1, 1))
        self.edge_gate = nn.Linear(d_model * 2, num_edge_types)

        self.out_proj = nn.Linear(d_model, d_model)

        # Gated Recurrent / Residual Node Update (GRU-style message fusion + FFN)
        self.update_gate = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.Sigmoid()
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 3),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 3, d_model),
            nn.Dropout(dropout)
        )

    def forward(
        self,
        h: torch.Tensor,
        edge_adj: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            h: Node embeddings (B, 81, d_model)
            edge_adj: Multigraph adjacency tensor (B, 4, 81, 81)
        Returns:
            h_next: Updated node embeddings (B, 81, d_model)
            rel_attn_summary: Mean attention routed across each edge type (B, 4)
        """
        B, N, D = h.shape
        H = self.num_heads
        Dh = self.head_dim

        h_norm = self.norm1(h)
        q = self.q_proj(h_norm).view(B, N, H, Dh).transpose(1, 2)  # (B, H, N, Dh)
        k = self.k_proj(h_norm).view(B, N, H, Dh).transpose(1, 2)  # (B, H, N, Dh)

        # Base node-to-node attention logits: (B, H, N, N)
        raw_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        # Self-loop identity to ensure isolated nodes retain valid softmax distribution
        eye = torch.eye(N, device=h.device, dtype=h.dtype).view(1, 1, N, N)

        agg_messages = torch.zeros_like(h)
        rel_strengths = []

        # Compute adaptive per-node relation gating
        global_ctx = h_norm.mean(dim=1, keepdim=True).expand(B, N, D)
        rel_gates = F.softmax(self.edge_gate(torch.cat([h_norm, global_ctx], dim=-1)), dim=-1)  # (B, N, 4)

        for r_idx in range(self.num_edge_types):
            # Adjacency mask for relation r_idx: (B, 1, N, N)
            adj_r = edge_adj[:, r_idx].unsqueeze(1)
            connectivity = (adj_r + eye).clamp(0.0, 1.0)

            # Mask non-edges with large negative value
            scores_r = raw_scores + self.edge_type_bias[r_idx]
            scores_r = scores_r.masked_fill(connectivity <= 1e-5, -1e9)
            attn_r = F.softmax(scores_r, dim=-1)  # (B, H, N, N)

            # Zero out self-loop contribution when measuring pure relational message flow
            off_diag_attn = (attn_r * adj_r).mean(dim=(1, 2, 3))  # (B,)
            rel_strengths.append(off_diag_attn)

            # Project source node messages with edge-type specific linear layer
            v_r = self.edge_v_projs[r_idx](h_norm).view(B, N, H, Dh).transpose(1, 2)  # (B, H, N, Dh)
            msg_r = torch.matmul(attn_r, v_r)  # (B, H, N, Dh)
            msg_r = msg_r.transpose(1, 2).contiguous().view(B, N, D)

            # Weight by node-level relation gate
            gate_r = rel_gates[:, :, r_idx].unsqueeze(-1)  # (B, N, 1)
            agg_messages = agg_messages + gate_r * msg_r

        projected_msg = self.out_proj(agg_messages)

        # Gated residual node update
        z_gate = self.update_gate(torch.cat([h, projected_msg], dim=-1))
        h = h + z_gate * projected_msg
        h = h + self.ffn(self.norm2(h))

        rel_attn_summary = torch.stack(rel_strengths, dim=-1)  # (B, 4)
        return h, rel_attn_summary


class WormholeGraphNeuralNet(nn.Module):
    """
    Explicit Heterogeneous Directed Multigraph Message Passing Network (MPNN)
    over the 81 Candy Crush board cells.

    Executes 8 rounds of Edge-Typed Graph Attention Message Passing over:
      - E_adj (4-neighbor grid adjacency)
      - E_gravity (directed downward column drop edges)
      - E_portal (non-local teleporter wormhole edges from channels 16 & 17)
      - E_special_blast (Striped row/col laser edges from channels 6 & 7 + Wrapped 3x3 from channel 8)

    Inputs:
      - board_tensor: (B, 20, 9, 9) canonical 20-channel spatial state
    Outputs (Dict):
      - policy_logits: (B, 144) canonical swap action logits
      - value: (B, 1) expected state value in [-1, 1]
      - win_prob: (B, 1) calibrated win probability in [0, 1]
      - node_embeddings: (B, 81, d_model) final 8-hop graph node representations
      - edge_adj: (B, 4, 81, 81) dynamically constructed 4-type multigraph adjacency
      - relation_attention: (B, 8, 4) message passing intensity per layer and edge type
    """

    def __init__(
        self,
        in_channels: int = 20,
        board_h: int = 9,
        board_w: int = 9,
        d_model: int = 192,
        num_mpnn_rounds: int = 8,
        num_heads: int = 6,
        num_actions: int = 144
    ):
        super().__init__()
        self.in_channels = in_channels
        self.board_h = board_h
        self.board_w = board_w
        self.num_nodes = board_h * board_w
        self.d_model = d_model
        self.num_mpnn_rounds = num_mpnn_rounds
        self.num_actions = num_actions

        # 1. Dynamic 4-Relation Multigraph Builder
        self.graph_builder = DynamicMultigraphBuilder(height=board_h, width=board_w)

        # 2. Initial Node Feature Encoder (local 1x1 + 3x3 context + 2D coordinate encoding)
        self.node_stem = nn.Sequential(
            nn.Conv2d(in_channels, d_model, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(d_model),
            nn.GELU(),
            nn.Conv2d(d_model, d_model, kernel_size=1),
            nn.GELU()
        )
        self.coord_embed = nn.Parameter(torch.randn(1, self.num_nodes, d_model) * 0.02)

        # 3. 8 Rounds of Edge-Typed Graph Attention Message Passing
        self.mpnn_layers = nn.ModuleList([
            EdgeTypedGraphAttentionLayer(
                d_model=d_model,
                num_heads=num_heads,
                num_edge_types=4
            )
            for _ in range(num_mpnn_rounds)
        ])
        self.final_norm = nn.LayerNorm(d_model)

        # 4. Pairwise Edge Swap Policy Head over the 144 Canonical Actions
        # Combines spatial conv readout with direct (u, v) swap pair interaction features
        self.policy_conv = nn.Sequential(
            nn.Conv2d(d_model, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.Conv2d(128, 64, kernel_size=1),
            nn.GELU()
        )
        self.horiz_pair_scorer = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1)
        )
        self.vert_pair_scorer = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1)
        )
        self.policy_fuse = nn.Linear(64 * self.num_nodes + num_actions, num_actions)

        # 5. Value & Win Probability Readout Heads
        self.value_head = nn.Sequential(
            nn.Linear(d_model * 2, 128),
            nn.GELU(),
            nn.Linear(128, 1),
            nn.Tanh()
        )
        self.win_head = nn.Sequential(
            nn.Linear(d_model * 2, 128),
            nn.GELU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def _score_swap_edges(self, h_2d: torch.Tensor) -> torch.Tensor:
        """
        Scores all 72 horizontal swaps (r, c) <-> (r, c+1) and 72 vertical swaps (r, c) <-> (r+1, c)
        directly from adjacent node embeddings.
        """
        B, D, H, W = h_2d.shape
        # Horizontal pairs: (B, H, W-1, 2*D) -> (B, 72)
        h_left = h_2d[:, :, :, :-1].permute(0, 2, 3, 1)   # (B, 9, 8, D)
        h_right = h_2d[:, :, :, 1:].permute(0, 2, 3, 1)   # (B, 9, 8, D)
        h_pairs = torch.cat([h_left, h_right], dim=-1).reshape(B, -1, 2 * D)
        horiz_logits = self.horiz_pair_scorer(h_pairs).squeeze(-1)  # (B, 72)

        # Vertical pairs: (B, H-1, W, 2*D) -> (B, 72)
        v_top = h_2d[:, :, :-1, :].permute(0, 2, 3, 1)    # (B, 8, 9, D)
        v_bot = h_2d[:, :, 1:, :].permute(0, 2, 3, 1)     # (B, 8, 9, D)
        v_pairs = torch.cat([v_top, v_bot], dim=-1).reshape(B, -1, 2 * D)
        vert_logits = self.vert_pair_scorer(v_pairs).squeeze(-1)   # (B, 72)

        return torch.cat([horiz_logits, vert_logits], dim=-1)       # (B, 144)

    def forward(
        self,
        board_tensor: torch.Tensor,
        legal_mask: Optional[torch.Tensor] = None,
        portal_pairs: Optional[List[List[Tuple[Tuple[int, int], Tuple[int, int]]]]] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Executes dynamic multigraph construction and 8 rounds of Edge-Typed MPNN.
        """
        B, C, H, W = board_tensor.shape

        # Step 1: Build 4-type directed multigraph (E_adj, E_gravity, E_portal, E_special_blast)
        edge_adj = self.graph_builder(board_tensor, portal_pairs=portal_pairs)  # (B, 4, 81, 81)

        # Step 2: Encode initial node embeddings
        stem_2d = self.node_stem(board_tensor)  # (B, d_model, 9, 9)
        h = stem_2d.flatten(2).transpose(1, 2) + self.coord_embed  # (B, 81, d_model)

        # Step 3: 8 Rounds of Edge-Typed Graph Attention Message Passing
        rel_attn_history = []
        for mpnn_layer in self.mpnn_layers:
            h, rel_summary = mpnn_layer(h, edge_adj)
            rel_attn_history.append(rel_summary)

        h = self.final_norm(h)  # (B, 81, d_model)
        relation_attention = torch.stack(rel_attn_history, dim=1)  # (B, 8, 4)

        # Step 4: Policy logits (B, 144) combining pairwise edge scores + spatial context
        h_2d = h.transpose(1, 2).view(B, self.d_model, H, W)
        conv_feat = self.policy_conv(h_2d).flatten(1)
        pair_logits = self._score_swap_edges(h_2d)
        policy_logits = pair_logits + self.policy_fuse(torch.cat([conv_feat, pair_logits], dim=-1))

        if legal_mask is not None:
            policy_logits = policy_logits.masked_fill(~legal_mask.bool(), -1e9)

        # Step 5: Global Graph Readout (Mean + Max pooling across all 81 nodes)
        h_mean = h.mean(dim=1)
        h_max = h.max(dim=1).values
        graph_readout = torch.cat([h_mean, h_max], dim=-1)  # (B, 2 * d_model)

        value = self.value_head(graph_readout)     # (B, 1)
        win_prob = self.win_head(graph_readout)    # (B, 1)

        return {
            "policy_logits": policy_logits,
            "value": value,
            "win_prob": win_prob,
            "node_embeddings": h,
            "edge_adj": edge_adj,
            "relation_attention": relation_attention,
        }

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


if __name__ == "__main__":
    torch.manual_seed(42)
    model = WormholeGraphNeuralNet()
    params = model.count_parameters()
    print(f"[WormholeGraphNeuralNet] Trainable Parameters: {params:,}")

    # Test with batch size 2 including synthetic portal (ch 16, 17) and striped/wrapped specials (ch 6, 7, 8)
    dummy_board = torch.zeros(2, 20, 9, 9)
    dummy_board[:, 18] = 1.0  # Active grid
    dummy_board[0, 6, 4, 4] = 1.0   # Striped H at (4,4)
    dummy_board[0, 8, 2, 2] = 1.0   # Wrapped at (2,2)
    dummy_board[0, 16, 8, 1] = 1.0  # Portal entrance at (8,1)
    dummy_board[0, 17, 0, 7] = 1.0  # Portal exit at (0,7)

    out = model(dummy_board)
    print("  policy_logits:", out["policy_logits"].shape)
    print("  value:", out["value"].shape)
    print("  win_prob:", out["win_prob"].shape)
    print("  edge_adj:", out["edge_adj"].shape)
    print("  relation_attention:", out["relation_attention"].shape)
