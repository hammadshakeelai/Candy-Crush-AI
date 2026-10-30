r"""
AlphaCandy Model Zoo — Part 6.2: Quantum-Inspired 2D Tensor Network & Matrix Product State
(`MatrixProductStateCandyNet`).
=========================================================================================
Based on:
  - Stoudenmire & Schwab (NeurIPS 2016), "Supervised Learning with Tensor Networks"
  - Oseledets (SIAM J. Sci. Comput. 2011), "Tensor-Train Decomposition"
  - Orus (Annals of Physics 2014), "A Practical Introduction to Tensor Networks: Matrix
    Product States and Projected Entangled Pair States"

Why Matrix Product States (MPS / Tensor-Train) for Candy Crush?
---------------------------------------------------------------
In Candy Crush, the highest-value game-winning patterns are strictly **high-order collinear
and cross-collinear correlations** across an entire 9-cell row or 9-cell column:
  1. **5-in-a-row Color Bomb creation**: Requires coherent 5-site collinear alignment $(c, c+1, c+2, c+3, c+4)$
     induced by a 1-step swap at one site.
  2. **T-shape / L-shape Wrapped Candy intersections**: Requires simultaneous 3-site horizontal
     and 3-site vertical tensor contractions meeting at a pivot vertex $(r, c)$.
  3. **Full-Row / Full-Column Striped Laser Clears**: Evaluates the product state across all 9
     sites in row $r$ or column $c$ simultaneously.

Standard $3\times 3$ CNNs only capture local 3-site windows per layer, while full 9-body tensor
products scale exponentially as $\mathcal{O}(d^9)$. `MatrixProductStateCandyNet` factorizes
each 9-cell row and 9-cell column into a **1D Matrix Product State (MPS / Tensor-Train)** chain
with virtual bond dimension $\chi = 64$:
    $$\Psi(x_1, x_2, \dots, x_9) = \text{Tr}\Big( A^{(1)}(x_1) \, A^{(2)}(x_2) \cdots A^{(9)}(x_9) \Big)$$
where each site tensor $A^{(i)}(x_i) \in \mathbb{R}^{\chi \times \chi}$ is parameterized via a compact
low-rank CP / Tucker factorized core to keep parameter count sleek while computing exact
left-environment ($L_i \in \mathbb{R}^{\chi}$) and right-environment ($R_i \in \mathbb{R}^{\chi}$)
tensor contractions at every board coordinate $(r, c)$.
"""

import math
from typing import Dict, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class MPSChainContractionLayer(nn.Module):
    """
    1D Matrix Product State (MPS / Tensor-Train) Chain Contraction Layer over a 9-site line
    (either a 9-cell horizontal row or a 9-cell vertical column) with virtual bond dimension
    $\chi = 64$.

    At each site $i \in \{0, \dots, 8\}$, maps local physical feature vector $x_i \in \mathbb{R}^{d_{\text{phys}}}$
    into a transfer matrix $M_i(x_i) \in \mathbb{R}^{\chi \times \chi}$ stabilized near the identity:
        $$M_i(x_i) = I_\chi + \frac{1}{\sqrt{\chi r}} \sum_{k=1}^{r} (U_{i,k} x_i) \otimes (V_{i,k} x_i)^T$$
    Computes:
      1. Prefix Left-Boundary Environments: $L_i = v_{\text{left}}^T M_0 M_1 \cdots M_{i-1} \in \mathbb{R}^{\chi}$
      2. Suffix Right-Boundary Environments: $R_i = M_{i+1} \cdots M_8 v_{\text{right}} \in \mathbb{R}^{\chi}$
      3. Full-Chain Tensor-Train Contraction: $\Psi_{\text{chain}} = L_8 M_8 \in \mathbb{R}^{\chi}$
      4. Site-Marginal Entangled Features: $E_i = (L_i M_i) \odot R_i \in \mathbb{R}^{\chi}$
    """

    def __init__(
        self,
        d_phys: int = 96,
        bond_dim: int = 64,
        chain_len: int = 9,
        cp_rank: int = 8
    ):
        super().__init__()
        self.d_phys = d_phys
        self.bond_dim = bond_dim
        self.chain_len = chain_len
        self.cp_rank = cp_rank
        self.scale = 1.0 / math.sqrt(bond_dim * cp_rank)

        # Site-specific physical-to-bond factor projections:
        # Maps physical state x_i -> left bond factors U_i (cp_rank, chi) and right bond factors V_i (cp_rank, chi)
        self.phys_to_u = nn.Linear(d_phys, cp_rank * bond_dim, bias=False)
        self.phys_to_v = nn.Linear(d_phys, cp_rank * bond_dim, bias=False)
        self.site_pos_bias = nn.Parameter(torch.randn(1, chain_len, cp_rank, bond_dim) * 0.02)

        # Diagonal bond modulation per site
        self.diag_proj = nn.Linear(d_phys, bond_dim)

        # Learnable left and right quantum boundary vectors |v_L>, |v_R> in R^{bond_dim}
        self.left_boundary = nn.Parameter(torch.ones(bond_dim) / math.sqrt(bond_dim))
        self.right_boundary = nn.Parameter(torch.ones(bond_dim) / math.sqrt(bond_dim))

        # Project entangled site marginals + full-chain state back to physical dimension
        self.marginal_out = nn.Sequential(
            nn.Linear(bond_dim * 2, d_phys),
            nn.LayerNorm(d_phys),
            nn.GELU()
        )

    def forward(self, chain_seq: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            chain_seq: (N_chains, 9, d_phys) sequence of 9 collinear cells
                       where N_chains = B * 9 (all rows or all columns in batch)

        Returns:
            site_features: (N_chains, 9, d_phys) entangled marginal features at every cell
            full_chain_mps: (N_chains, bond_dim) contracted 9-body tensor-train state vector
            site_entanglement: (N_chains, 9) proxy entanglement spectrum norm at each bond cut
        """
        N_c, L, _ = chain_seq.shape
        chi = self.bond_dim
        R = self.cp_rank

        # Construct low-rank + diagonal transfer matrices M_i(x_i) in R^(N_c, L, chi, chi)
        u_factors = self.phys_to_u(chain_seq).view(N_c, L, R, chi) + self.site_pos_bias
        v_factors = self.phys_to_v(chain_seq).view(N_c, L, R, chi)
        diag_gate = torch.tanh(self.diag_proj(chain_seq))  # (N_c, L, chi)

        # Low-rank outer product sum_{r=1}^R u_{i,r} \otimes v_{i,r}
        low_rank_m = torch.einsum("nlri,nlrj->nlij", u_factors, v_factors) * self.scale
        eye = torch.eye(chi, device=chain_seq.device, dtype=chain_seq.dtype).view(1, 1, chi, chi)
        diag_m = eye * (1.0 + 0.25 * diag_gate.unsqueeze(-1))

        # Transfer matrices M_i for i = 0..8: shape (N_c, 9, chi, chi)
        transfer_matrices = diag_m + low_rank_m

        # Forward sweep: compute left environments L_i in R^(N_c, chi)
        # L_0 = v_left, L_{i+1} = normalize(L_i @ M_i)
        v_l = self.left_boundary.unsqueeze(0).expand(N_c, chi)
        left_envs = [v_l]
        curr_l = v_l
        for i in range(L):
            curr_l = torch.bmm(curr_l.unsqueeze(1), transfer_matrices[:, i]).squeeze(1)
            # L2 stabilization to prevent exponential norm explosion/vanishing across 9-body product
            curr_l = curr_l / (torch.norm(curr_l, p=2, dim=-1, keepdim=True) / math.sqrt(chi) + 1e-6)
            left_envs.append(curr_l)

        # Full 9-site contracted MPS vector: L_9 in R^(N_c, chi)
        full_chain_mps = left_envs[-1]

        # Backward sweep: compute right environments R_i in R^(N_c, chi)
        # R_8 = v_right, R_{i-1} = normalize(M_i @ R_i)
        v_r = self.right_boundary.unsqueeze(0).expand(N_c, chi)
        right_envs = [None] * L
        curr_r = v_r
        for i in range(L - 1, -1, -1):
            right_envs[i] = curr_r
            curr_r = torch.bmm(transfer_matrices[:, i], curr_r.unsqueeze(2)).squeeze(2)
            curr_r = curr_r / (torch.norm(curr_r, p=2, dim=-1, keepdim=True) / math.sqrt(chi) + 1e-6)

        # Combine left environment L_i, site transfer M_i, and right environment R_i at every site i
        site_marginals = []
        bond_entanglements = []
        for i in range(L):
            l_i = left_envs[i]  # (N_c, chi)
            r_i = right_envs[i]  # (N_c, chi)
            m_i = transfer_matrices[:, i]  # (N_c, chi, chi)

            # Left-to-site contraction: l_i @ M_i -> (N_c, chi)
            lm_i = torch.bmm(l_i.unsqueeze(1), m_i).squeeze(1)
            # Hadamard bond contraction with right environment + full-chain context
            entangled_i = lm_i * r_i
            site_marginals.append(torch.cat([entangled_i, full_chain_mps], dim=-1))

            # Schmidt-like bond entanglement proxy across cut i: ||lm_i * r_i||_2
            schmidt_proxy = (entangled_i.pow(2).mean(dim=-1) + 1e-8).sqrt()
            bond_entanglements.append(schmidt_proxy)

        stacked_marginals = torch.stack(site_marginals, dim=1)       # (N_c, 9, 2*chi)
        site_features = self.marginal_out(stacked_marginals)         # (N_c, 9, d_phys)
        site_entanglement = torch.stack(bond_entanglements, dim=1)   # (N_c, 9)

        return site_features, full_chain_mps, site_entanglement


class DualAxis2DMPSEncoderBlock(nn.Module):
    """
    2D Projected Entangled Pair / Cross-Chain MPS Block that contracts:
      1. Horizontal Row MPS Chains ($9$ rows of length $9$, bond dimension $\chi=64$)
      2. Vertical Column MPS Chains ($9$ columns of length $9$, bond dimension $\chi=64$)
    and fuses their entangled bond representations at every 2D vertex $(r, c)$ to capture
    both 5-in-a-row collinear Color Bombs and T/L-shaped Wrapped intersections.
    """

    def __init__(self, d_model: int = 128, bond_dim: int = 64, cp_rank: int = 8):
        super().__init__()
        self.d_model = d_model
        self.bond_dim = bond_dim

        self.row_mps = MPSChainContractionLayer(
            d_phys=d_model,
            bond_dim=bond_dim,
            chain_len=9,
            cp_rank=cp_rank
        )
        self.col_mps = MPSChainContractionLayer(
            d_phys=d_model,
            bond_dim=bond_dim,
            chain_len=9,
            cp_rank=cp_rank
        )

        # Cross-axis vertex tensor fusion (row + col + bilinear Hadamard interaction for T/L intersections)
        self.vertex_fusion = nn.Sequential(
            nn.Linear(d_model * 3, d_model),
            nn.LayerNorm(d_model),
            nn.GELU()
        )
        self.local_conv = nn.Sequential(
            nn.Conv2d(d_model, d_model, kernel_size=3, padding=1, groups=4, bias=False),
            nn.BatchNorm2d(d_model),
            nn.GELU()
        )

    def forward(
        self,
        x_2d: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x_2d: (B, d_model, 9, 9) spatial feature grid

        Returns:
            out_2d: (B, d_model, 9, 9) updated 2D tensor network features
            row_states: (B, 9, bond_dim) contracted horizontal row MPS states
            col_states: (B, 9, bond_dim) contracted vertical column MPS states
            entanglement_2d: (B, 9, 9) combined row+col bond entanglement map
        """
        B, D, H, W = x_2d.shape

        # 1. Horizontal Row MPS Contraction: reshape (B, D, H, W) -> (B*H, W, D)
        row_seq = x_2d.permute(0, 2, 3, 1).contiguous().view(B * H, W, D)
        row_feat, row_mps_vec, row_ent = self.row_mps(row_seq)
        row_feat_2d = row_feat.view(B, H, W, D)              # (B, 9, 9, D)
        row_states = row_mps_vec.view(B, H, self.bond_dim)   # (B, 9, chi)
        row_ent_2d = row_ent.view(B, H, W)                   # (B, 9, 9)

        # 2. Vertical Column MPS Contraction: reshape (B, D, H, W) -> (B*W, H, D)
        col_seq = x_2d.permute(0, 3, 2, 1).contiguous().view(B * W, H, D)
        col_feat, col_mps_vec, col_ent = self.col_mps(col_seq)
        col_feat_2d = col_feat.view(B, W, H, D).transpose(1, 2).contiguous()  # (B, 9, 9, D)
        col_states = col_mps_vec.view(B, W, self.bond_dim)                    # (B, 9, chi)
        col_ent_2d = col_ent.view(B, W, H).transpose(1, 2).contiguous()       # (B, 9, 9)

        # 3. Vertex intersection contraction (captures T/L Wrapped shapes & row+col alignment)
        cross_term = row_feat_2d * col_feat_2d
        fused_vertex = self.vertex_fusion(
            torch.cat([row_feat_2d, col_feat_2d, cross_term], dim=-1)
        )  # (B, 9, 9, D)

        fused_2d = fused_vertex.permute(0, 3, 1, 2).contiguous()
        out_2d = x_2d + fused_2d + 0.5 * self.local_conv(fused_2d)
        entanglement_2d = 0.5 * (row_ent_2d + col_ent_2d)

        return out_2d, row_states, col_states, entanglement_2d


class MatrixProductStateCandyNet(nn.Module):
    """
    Quantum-Inspired 2D Tensor Network / Matrix Product State Policy-Value Network
    (`MatrixProductStateCandyNet`) with bond dimension $\chi = 64$.

    Designed specifically to capture high-order 9-cell collinear correlations
    (5-in-a-row Color Bombs, 4-in-a-row Striped candies, T/L Wrapped intersections,
    and full-row/full-column laser clears) with compact tensor-train parameters.
    """

    def __init__(
        self,
        in_channels: int = 20,
        board_h: int = 9,
        board_w: int = 9,
        d_model: int = 128,
        bond_dim: int = 64,
        num_mps_blocks: int = 3,
        cp_rank: int = 8,
        num_actions: int = 144
    ):
        super().__init__()
        self.in_channels = in_channels
        self.board_h = board_h
        self.board_w = board_w
        self.num_nodes = board_h * board_w
        self.d_model = d_model
        self.bond_dim = bond_dim
        self.num_mps_blocks = num_mps_blocks
        self.num_actions = num_actions

        # 1. Local Physical State Feature Encoder (maps 20 channels -> d_model physical Hilbert space)
        self.physical_stem = nn.Sequential(
            nn.Conv2d(in_channels, d_model, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(d_model),
            nn.GELU(),
            nn.Conv2d(d_model, d_model, kernel_size=1),
            nn.GELU()
        )
        self.coord_embed = nn.Parameter(torch.randn(1, d_model, board_h, board_w) * 0.02)

        # 2. Stack of Dual-Axis 2D MPS Contraction Blocks (chi = 64)
        self.mps_blocks = nn.ModuleList([
            DualAxis2DMPSEncoderBlock(
                d_model=d_model,
                bond_dim=bond_dim,
                cp_rank=cp_rank
            )
            for _ in range(num_mps_blocks)
        ])

        # 3. Auxiliary Collinear 5-in-a-Row / Full-Line Clear Predictor Head
        # Predicts per-row (9) and per-col (9) probability of a high-order >=4/5 collinear match
        self.collinear_line_head = nn.Sequential(
            nn.Linear(bond_dim, 32),
            nn.GELU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

        # 4. Bond-Aware Swap Policy Readout Head (144 actions)
        # Each swap (r, c) <-> (r', c') is scored using both vertex MPS features and
        # the contracted bond states of the affected rows & columns
        self.policy_conv = nn.Sequential(
            nn.Conv2d(d_model + 1, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.Conv2d(64, 32, kernel_size=1),
            nn.GELU()
        )
        pair_dim = d_model * 2 + bond_dim * 2
        self.horiz_swap_mlp = nn.Sequential(
            nn.Linear(pair_dim, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1)
        )
        self.vert_swap_mlp = nn.Sequential(
            nn.Linear(pair_dim, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1)
        )
        self.policy_fuse = nn.Linear(32 * self.num_nodes + num_actions, num_actions)

        # 5. Global Tensor-Train Value & Win Probability Readout Heads
        # Contracts the 9 row bond vectors (B, 9, chi) and 9 column bond vectors (B, 9, chi)
        value_in_dim = d_model + bond_dim * 2
        self.value_head = nn.Sequential(
            nn.Linear(value_in_dim, 128),
            nn.GELU(),
            nn.Linear(128, 1),
            nn.Tanh()
        )
        self.win_head = nn.Sequential(
            nn.Linear(value_in_dim, 128),
            nn.GELU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def _score_mps_swaps(
        self,
        x_2d: torch.Tensor,
        row_states: torch.Tensor,
        col_states: torch.Tensor
    ) -> torch.Tensor:
        """
        Scores all 72 horizontal swaps and 72 vertical swaps by conditioning each local
        pair (u, v) on the contracted $\chi=64$ MPS bond state of its row and column.
        """
        B, D, H, W = x_2d.shape
        chi = self.bond_dim

        # Expand row_states (B, 9, chi) and col_states (B, 9, chi) onto the 9x9 grid
        row_grid = row_states.unsqueeze(2).expand(B, H, W, chi)  # (B, 9, 9, chi)
        col_grid = col_states.unsqueeze(1).expand(B, H, W, chi)  # (B, 9, 9, chi)
        x_grid = x_2d.permute(0, 2, 3, 1)                        # (B, 9, 9, D)

        # 1. Horizontal swaps (r, c) <-> (r, c+1): 9x8 = 72
        h_u = x_grid[:, :, :-1, :]      # (B, 9, 8, D)
        h_v = x_grid[:, :, 1:, :]       # (B, 9, 8, D)
        h_row_bond = row_grid[:, :, :-1, :]  # Same row r for both u and v
        h_col_bond = 0.5 * (col_grid[:, :, :-1, :] + col_grid[:, :, 1:, :])
        horiz_in = torch.cat([h_u, h_v, h_row_bond, h_col_bond], dim=-1).reshape(B, 72, -1)
        horiz_logits = self.horiz_swap_mlp(horiz_in).squeeze(-1)  # (B, 72)

        # 2. Vertical swaps (r, c) <-> (r+1, c): 8x9 = 72
        v_u = x_grid[:, :-1, :, :]      # (B, 8, 9, D)
        v_v = x_grid[:, 1:, :, :]       # (B, 8, 9, D)
        v_row_bond = 0.5 * (row_grid[:, :-1, :, :] + row_grid[:, 1:, :, :])
        v_col_bond = col_grid[:, :-1, :, :]  # Same col c for both u and v
        vert_in = torch.cat([v_u, v_v, v_row_bond, v_col_bond], dim=-1).reshape(B, 72, -1)
        vert_logits = self.vert_swap_mlp(vert_in).squeeze(-1)     # (B, 72)

        return torch.cat([horiz_logits, vert_logits], dim=-1)     # (B, 144)

    def forward(
        self,
        board_tensor: torch.Tensor,
        legal_mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Executes 2D Matrix Product State (MPS) row & column chain contractions with bond
        dimension $\chi = 64$.
        """
        B, C, H, W = board_tensor.shape

        # Step 1: Map 20-channel board into physical site embeddings
        x_2d = self.physical_stem(board_tensor) + self.coord_embed  # (B, d_model, 9, 9)

        # Step 2: Execute stack of Dual-Axis 2D MPS Contraction Blocks (chi = 64)
        row_states = None
        col_states = None
        entanglement_acc = torch.zeros(B, H, W, device=board_tensor.device, dtype=board_tensor.dtype)

        for block in self.mps_blocks:
            x_2d, row_states, col_states, ent_2d = block(x_2d)
            entanglement_acc = entanglement_acc + ent_2d

        entanglement_spectrum_norm = entanglement_acc / float(len(self.mps_blocks))

        # Step 3: Predict collinear 5-in-a-row / Color Bomb & Striped line potential
        # Shape: (B, 2, 9) where dim 1 index 0 = 9 rows, index 1 = 9 columns
        row_5match_prob = self.collinear_line_head(row_states).squeeze(-1)  # (B, 9)
        col_5match_prob = self.collinear_line_head(col_states).squeeze(-1)  # (B, 9)
        collinear_5match_prob = torch.stack([row_5match_prob, col_5match_prob], dim=1)  # (B, 2, 9)

        # Step 4: Score all 144 swap actions using bond-contracted MPS features
        conv_in = torch.cat([x_2d, entanglement_spectrum_norm.unsqueeze(1)], dim=1)
        conv_flat = self.policy_conv(conv_in).flatten(1)
        pair_logits = self._score_mps_swaps(x_2d, row_states, col_states)
        policy_logits = pair_logits + self.policy_fuse(torch.cat([conv_flat, pair_logits], dim=-1))

        if legal_mask is not None:
            policy_logits = policy_logits.masked_fill(~legal_mask.bool(), -1e9)

        # Step 5: Global Value & Win Probability Readout from contracted MPS bond states
        spatial_mean = x_2d.mean(dim=(2, 3))    # (B, d_model)
        row_bond_mean = row_states.mean(dim=1)  # (B, 64)
        col_bond_mean = col_states.mean(dim=1)  # (B, 64)
        global_mps_feat = torch.cat([spatial_mean, row_bond_mean, col_bond_mean], dim=-1)

        value = self.value_head(global_mps_feat)
        win_prob = self.win_head(global_mps_feat)

        return {
            "policy_logits": policy_logits,
            "value": value,
            "win_prob": win_prob,
            "row_mps_states": row_states,
            "col_mps_states": col_states,
            "collinear_5match_prob": collinear_5match_prob,
            "entanglement_spectrum_norm": entanglement_spectrum_norm,
        }

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


if __name__ == "__main__":
    torch.manual_seed(42)
    model = MatrixProductStateCandyNet(bond_dim=64)
    params = model.count_parameters()
    print(f"[MatrixProductStateCandyNet] Trainable Parameters: {params:,}")

    # Test with batch size 2 including a near-5-in-a-row Color Bomb setup on row 4
    dummy_board = torch.zeros(2, 20, 9, 9)
    dummy_board[:, 18] = 1.0  # Active grid
    # Red candies (ch 0) at (4,1), (4,2), (4,4), (4,5) and (3,3) ready to drop into (4,3)
    for col in [1, 2, 4, 5]:
        dummy_board[0, 0, 4, col] = 1.0
    dummy_board[0, 0, 3, 3] = 1.0

    out = model(dummy_board)
    print("  policy_logits:", out["policy_logits"].shape)
    print("  value:", out["value"].shape)
    print("  win_prob:", out["win_prob"].shape)
    print("  row_mps_states (chi=64):", out["row_mps_states"].shape)
    print("  col_mps_states (chi=64):", out["col_mps_states"].shape)
    print("  collinear_5match_prob:", out["collinear_5match_prob"].shape)
    print("  entanglement_spectrum_norm:", out["entanglement_spectrum_norm"].shape)
