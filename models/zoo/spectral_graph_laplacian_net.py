r"""
AlphaCandy Model Zoo — Part 6.1: Spectral Graph Laplacian & Chebyshev Wavelet Network
(`SpectralGraphWaveletCandyNet`).
=====================================================================================
Based on:
  - Defferrard, Bresson, & Vandergheynst (NeurIPS 2016), "Convolutional Neural Networks on
    Graphs with Fast Localized Spectral Filtering" (ChebNet)
  - Hammond, Vandergheynst, & Gribonval (2011), "Wavelets on Graphs via Spectral Graph Theory"
  - Chung (1997), "Spectral Graph Theory" (Normalized Graph Laplacian & Fiedler Connectivity)

Why Spectral Graph Theory for Candy Crush?
------------------------------------------
Standard spatial 3x3 convolutions treat a 9x9 Candy Crush grid as a uniform Euclidean plane.
However, real Candy Crush levels (waffle cutouts, frostings, licorice locks, disconnected
islands, and non-local teleporter portals) form highly irregular, non-planar graphs with severe
topological bottlenecks:
  1. Narrow 1-cell chokepoints (bridges) where a single blocker freezes an entire chamber.
  2. Disconnected island chambers connected exclusively via non-local portal edges.
  3. High-resistance gravity funnels where cascades cannot propagate across chambers.

`SpectralGraphWaveletCandyNet` dynamically constructs the active board + portal adjacency
matrix $A \in \mathbb{R}^{B \times 81 \times 81}$, computes the symmetric normalized Graph
Laplacian:
    $$\mathcal{L} = I - D^{-1/2} A D^{-1/2}$$
and applies **Chebyshev Spectral Graph Convolutions ($K=4$ polynomial orders)** alongside
multi-scale **Spectral Graph Wavelet Filters** $\Psi_s(\mathcal{L})$ (low-pass island modes +
band/high-pass chokepoint modes) to detect topological bottlenecks and guide swap selection.
"""

import math
from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class DynamicSpectralLaplacianBuilder(nn.Module):
    """
    Constructs the weighted topological adjacency matrix $A \in \mathbb{R}^{B \times 81 \times 81}$
    from the canonical 20-channel board tensor (accounting for active mask channel 18,
    blocker permeability in channels 10-14, and portal wormholes in channels 16-17),
    and returns both the Symmetric Normalized Graph Laplacian:
        $\mathcal{L} = I - D^{-1/2} A D^{-1/2}$
    and the rescaled Chebyshev operator:
        $\tilde{\mathcal{L}} = \frac{2}{\lambda_{\max}} \mathcal{L} - I \approx \mathcal{L} - I = -D^{-1/2} A D^{-1/2}$
    whose spectrum lies strictly within $[-1, 1]$.
    """

    def __init__(self, height: int = 9, width: int = 9):
        super().__init__()
        self.height = height
        self.width = width
        self.num_nodes = height * width

        # Precompute 4-neighbor orthogonal grid adjacency and 8-neighbor diagonal adjacency
        ortho_adj = torch.zeros(self.num_nodes, self.num_nodes, dtype=torch.float32)
        diag_adj = torch.zeros(self.num_nodes, self.num_nodes, dtype=torch.float32)

        for r in range(height):
            for c in range(width):
                u = r * width + c
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < height and 0 <= nc < width:
                        v = nr * width + nc
                        ortho_adj[u, v] = 1.0
                for dr, dc in [(-1, -1), (-1, 1), (1, -1), (1, 1)]:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < height and 0 <= nc < width:
                        v = nr * width + nc
                        diag_adj[u, v] = 0.25

        self.register_buffer("ortho_adj", ortho_adj)
        self.register_buffer("diag_adj", diag_adj)
        self.register_buffer("identity", torch.eye(self.num_nodes, dtype=torch.float32))

        # Learnable edge conductance weights for blockers vs open cells vs portals
        self.portal_conductance = nn.Parameter(torch.tensor(1.25))
        self.blocker_attenuation = nn.Parameter(torch.tensor(0.35))

    def forward(
        self,
        board_tensor: torch.Tensor,
        portal_pairs: Optional[List[List[Tuple[Tuple[int, int], Tuple[int, int]]]]] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            board_tensor: (B, 20, 9, 9) canonical board state
            portal_pairs: Optional explicit list of portal coordinate pairs per batch item

        Returns:
            Dictionary containing:
              - `adjacency`: (B, 81, 81) symmetric non-negative weighted adjacency A
              - `normalized_laplacian`: (B, 81, 81) L = I - D^{-1/2} A D^{-1/2}
              - `rescaled_laplacian`: (B, 81, 81) L_tilde = L - I in [-1, 1]
              - `node_degrees`: (B, 81) weighted topological node degree D_ii
              - `active_mask`: (B, 81) active playable board cell mask
        """
        B, C, H, W = board_tensor.shape
        N = H * W

        # 1. Active playable grid mask (channel 18); fallback to 1s if unpopulated dummy tensor
        raw_active = board_tensor[:, 18].reshape(B, N)
        has_active = (raw_active.sum(dim=-1, keepdim=True) > 0).float()
        active_mask = (raw_active > 0.1).float() * has_active + (1.0 - has_active)  # (B, N)

        # 2. Blocker permeability: Frosting (ch 11), Lock (ch 12), Chocolate/Spawner (ch 13)
        #    attenuate edge conductance, creating spectral bottlenecks around heavy obstacles
        blocker_density = (
            board_tensor[:, 11] + board_tensor[:, 12] + board_tensor[:, 13]
        ).reshape(B, N).clamp(0.0, 1.0)
        attenuation = torch.sigmoid(self.blocker_attenuation)
        node_conductance = active_mask * (1.0 - (1.0 - attenuation) * blocker_density)  # (B, N)

        # Pairwise conductance between cells u and v: (B, N, N)
        pair_conductance = torch.sqrt(
            (node_conductance.unsqueeze(2) * node_conductance.unsqueeze(1)).clamp(min=1e-8)
        ) * (active_mask.unsqueeze(2) * active_mask.unsqueeze(1))

        # Base grid adjacency weighted by cell conductance
        base_grid = self.ortho_adj.unsqueeze(0) + self.diag_adj.unsqueeze(0)
        adj = base_grid * pair_conductance  # (B, N, N)

        # 3. Non-local teleporter portal connections (channels 16 & 17)
        ch16 = (board_tensor[:, 16].reshape(B, N) > 0.25).float() * active_mask
        ch17 = (board_tensor[:, 17].reshape(B, N) > 0.25).float() * active_mask
        portal_adj = torch.bmm(ch16.unsqueeze(2), ch17.unsqueeze(1))
        portal_adj = (portal_adj + portal_adj.transpose(1, 2)).clamp(0.0, 1.0)

        if portal_pairs is not None:
            for b_idx, b_portals in enumerate(portal_pairs):
                if b_idx < B and b_portals:
                    for (sr, sc), (dr, dc) in b_portals:
                        if 0 <= sr < H and 0 <= sc < W and 0 <= dr < H and 0 <= dc < W:
                            u = sr * W + sc
                            v = dr * W + dc
                            portal_adj[b_idx, u, v] = 1.0
                            portal_adj[b_idx, v, u] = 1.0

        portal_weight = F.softplus(self.portal_conductance)
        adj = adj + portal_weight * portal_adj

        # Ensure zero self-loops on A before Laplacian normalization
        eye = self.identity.unsqueeze(0)
        adj = adj * (1.0 - eye)

        # 4. Compute Symmetric Normalized Graph Laplacian: L = I - D^{-1/2} A D^{-1/2}
        degrees = adj.sum(dim=-1)  # (B, N)
        deg_inv_sqrt = torch.rsqrt(degrees.clamp(min=1e-5)) * active_mask  # (B, N)
        norm_adj = deg_inv_sqrt.unsqueeze(2) * adj * deg_inv_sqrt.unsqueeze(1)  # D^{-1/2} A D^{-1/2}

        normalized_laplacian = eye - norm_adj
        # Rescaled Laplacian for Chebyshev polynomials on [-1, 1] (with lambda_max = 2):
        # L_tilde = (2 / 2) * L - I = L - I = -D^{-1/2} A D^{-1/2}
        rescaled_laplacian = -norm_adj

        return {
            "adjacency": adj,
            "normalized_laplacian": normalized_laplacian,
            "rescaled_laplacian": rescaled_laplacian,
            "node_degrees": degrees,
            "active_mask": active_mask,
        }


class ChebyshevSpectralGraphConv(nn.Module):
    """
    Chebyshev Spectral Graph Convolution Layer (Defferrard et al., NeurIPS 2016)
    with polynomial order $K=4$.

    Evaluates the spectral filter $g_\theta(\mathcal{L}) X = \sum_{k=0}^{K} \Theta_k T_k(\tilde{\mathcal{L}}) X$
    using the stable Chebyshev recurrence:
      - $T_0(\tilde{\mathcal{L}}) X = X$
      - $T_1(\tilde{\mathcal{L}}) X = \tilde{\mathcal{L}} X$
      - $T_k(\tilde{\mathcal{L}}) X = 2 \tilde{\mathcal{L}} T_{k-1}(\tilde{\mathcal{L}}) X - T_{k-2}(\tilde{\mathcal{L}}) X$
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        k_order: int = 4,
        dropout: float = 0.05
    ):
        super().__init__()
        assert k_order >= 1, "Polynomial order K must be >= 1"
        self.in_features = in_features
        self.out_features = out_features
        self.k_order = k_order

        # K+1 learnable Chebyshev coefficient matrices Theta_0 ... Theta_K
        self.cheb_weights = nn.Parameter(
            torch.empty(k_order + 1, in_features, out_features)
        )
        self.bias = nn.Parameter(torch.zeros(out_features))
        self.norm1 = nn.LayerNorm(in_features)
        self.norm2 = nn.LayerNorm(out_features)
        self.dropout = nn.Dropout(dropout)

        # Residual projection if channel dimensions differ
        self.res_proj = (
            nn.Linear(in_features, out_features, bias=False)
            if in_features != out_features
            else nn.Identity()
        )

        # Channel-wise spectral band gating (learns how much weight to give each order k=0..K)
        self.order_gate = nn.Sequential(
            nn.Linear(in_features, k_order + 1),
            nn.Softmax(dim=-1)
        )

        # Post-spectral FFN
        self.ffn = nn.Sequential(
            nn.Linear(out_features, out_features * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(out_features * 2, out_features)
        )
        self._reset_parameters()

    def _reset_parameters(self):
        for k in range(self.k_order + 1):
            nn.init.xavier_uniform_(self.cheb_weights[k], gain=1.0 / math.sqrt(self.k_order + 1))

    def forward(
        self,
        x: torch.Tensor,
        rescaled_laplacian: torch.Tensor,
        active_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Node feature matrix (B, 81, in_features)
            rescaled_laplacian: L_tilde in [-1, 1] of shape (B, 81, 81)
            active_mask: (B, 81) binary mask of active nodes

        Returns:
            out: Updated node representations (B, 81, out_features)
            band_energies: Per-polynomial-order energy norms (B, K+1)
        """
        h = self.norm1(x)

        # Compute adaptive per-node polynomial order emphasis
        gates = self.order_gate(h.mean(dim=1))  # (B, K+1)

        # Recurrence for Chebyshev polynomials T_k(L_tilde) H
        t_k_minus_2 = h                                          # T_0(L_tilde) H = H
        t_k_minus_1 = torch.bmm(rescaled_laplacian, h)           # T_1(L_tilde) H = L_tilde H

        cheb_basis = [t_k_minus_2, t_k_minus_1]
        for _ in range(2, self.k_order + 1):
            t_k = 2.0 * torch.bmm(rescaled_laplacian, t_k_minus_1) - t_k_minus_2
            cheb_basis.append(t_k)
            t_k_minus_2 = t_k_minus_1
            t_k_minus_1 = t_k

        # Stack basis: (B, K+1, 81, in_features)
        basis_stack = torch.stack(cheb_basis, dim=1)

        # Weight each polynomial order by its adaptive gate
        gated_basis = basis_stack * gates.unsqueeze(-1).unsqueeze(-1)

        # Contract with Chebyshev filter banks: (B, K+1, 81, F_in) x (K+1, F_in, F_out) -> (B, 81, F_out)
        spectral_out = torch.einsum("bkni,kio->bno", gated_basis, self.cheb_weights) + self.bias
        spectral_out = self.dropout(F.gelu(spectral_out))

        # Residual + FFN masked by active cells
        mask_col = active_mask.unsqueeze(-1)
        out = (self.res_proj(x) + spectral_out) * mask_col
        out = (out + self.ffn(self.norm2(out))) * mask_col

        # Compute spectral energy per polynomial order k=0..K for diagnostics
        band_energies = torch.norm(basis_stack, p=2, dim=(-2, -1)) / math.sqrt(x.shape[1] * x.shape[2])
        return out, band_energies


class SpectralWaveletChokepointDetector(nn.Module):
    """
    Multi-Scale Spectral Graph Wavelet & Topological Chokepoint Detector
    (Hammond et al., 2011).

    Applies heat-diffusion low-pass scaling filters $h(\mathcal{L}) = (I - \tau \mathcal{L})^m$
    and band-pass wavelet filters $\psi_s(\mathcal{L}) = s \mathcal{L} (I - \alpha_s \mathcal{L})^2$
    to explicitly isolate:
      1. Low-frequency connected island chambers (Fiedler-mode clusters).
      2. High-frequency / band-pass 1-cell topological chokepoints (narrow bridges where
         Dirichlet energy $\|\nabla_{\mathcal{G}} f\|_2^2 = f^T \mathcal{L} f$ concentrates).
    """

    def __init__(self, d_model: int = 192, num_scales: int = 3):
        super().__init__()
        self.d_model = d_model
        self.num_scales = num_scales

        # Learnable wavelet diffusion scales s_1, s_2, s_3
        self.log_scales = nn.Parameter(torch.tensor([-0.7, 0.0, 0.5]))

        # Fuse multi-scale wavelet responses into a bottleneck chokepoint score & feature enrichment
        self.wavelet_fuse = nn.Sequential(
            nn.Linear(d_model * (num_scales + 1), d_model),
            nn.LayerNorm(d_model),
            nn.GELU()
        )
        self.chokepoint_head = nn.Sequential(
            nn.Linear(d_model + 3, 64),
            nn.GELU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(
        self,
        h: torch.Tensor,
        norm_laplacian: torch.Tensor,
        node_degrees: torch.Tensor,
        active_mask: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            h: (B, 81, d_model) node features
            norm_laplacian: (B, 81, 81) L = I - D^{-1/2} A D^{-1/2}
            node_degrees: (B, 81) topological degrees
            active_mask: (B, 81) active board mask
        """
        B, N, D = h.shape
        eye = torch.eye(N, device=h.device, dtype=h.dtype).unsqueeze(0)

        # Low-pass scaling operator (heat diffusion approximation): (I - 0.35 L)^2 H
        low_pass_op = eye - 0.35 * norm_laplacian
        h_low = torch.bmm(low_pass_op, torch.bmm(low_pass_op, h))

        # Band-pass spectral graph wavelets at 3 scales: psi_s(L) H = s * L * (I - 0.25*s*L) H
        scales = F.softplus(self.log_scales)
        wavelet_outputs = [h_low]
        scale_energies = []

        lh = torch.bmm(norm_laplacian, h)  # High-pass Laplacian response L H
        for s_idx in range(self.num_scales):
            s = scales[s_idx]
            damp_op = eye - (0.25 * s).clamp(max=0.48) * norm_laplacian
            psi_s_h = s * torch.bmm(damp_op, lh)
            wavelet_outputs.append(psi_s_h)
            energy_s = (psi_s_h.pow(2).mean(dim=(-2, -1))).sqrt()  # (B,)
            scale_energies.append(energy_s)

        fused_wavelets = self.wavelet_fuse(torch.cat(wavelet_outputs, dim=-1)) * active_mask.unsqueeze(-1)
        wavelet_energy = torch.stack(scale_energies, dim=-1)  # (B, num_scales)

        # Compute local Dirichlet gradient energy ||L h_u||_2 and degree bottleneck ratios
        local_dirichlet = torch.norm(lh, p=2, dim=-1, keepdim=True) / math.sqrt(D)  # (B, 81, 1)
        deg_norm = (node_degrees / 4.0).unsqueeze(-1)                                # (B, 81, 1)
        # Chokepoint indicator: active cell with low degree (1 or 2 open neighbors) & high spectral gradient
        degree_scarcity = torch.exp(-node_degrees).unsqueeze(-1) * active_mask.unsqueeze(-1)

        choke_in = torch.cat([fused_wavelets, local_dirichlet, deg_norm, degree_scarcity], dim=-1)
        chokepoint_scores = self.chokepoint_head(choke_in) * active_mask.unsqueeze(-1)  # (B, 81, 1)

        # Algebraic connectivity proxy (Rayleigh quotient of smoothed high-pass signal on active graph)
        # Lower values indicate nearly disconnected island chambers or severe chokepoints
        rayleigh_num = (h_low * torch.bmm(norm_laplacian, h_low)).sum(dim=(1, 2))
        rayleigh_den = h_low.pow(2).sum(dim=(1, 2)).clamp(min=1e-6)
        algebraic_connectivity_proxy = (rayleigh_num / rayleigh_den).unsqueeze(-1)  # (B, 1)

        return {
            "wavelet_features": fused_wavelets,
            "chokepoint_scores": chokepoint_scores,
            "spectral_wavelet_energy": wavelet_energy,
            "algebraic_connectivity_proxy": algebraic_connectivity_proxy,
        }


class SpectralGraphWaveletCandyNet(nn.Module):
    """
    Spectral Graph Laplacian & Chebyshev Wavelet Policy-Value Network for Candy Crush.

    Pipeline:
      1. Constructs the normalized Graph Laplacian $\mathcal{L} = I - D^{-1/2} A D^{-1/2}$
         from active cells, blocker conductances, and non-local portal connections.
      2. Encodes local cell features and applies a stack of $L=4$ **Chebyshev Spectral Graph
         Convolutions ($K=4$ polynomial orders)** to propagate multi-hop spectral information.
      3. Runs the **Spectral Wavelet Chokepoint Detector** to identify narrow 1-cell bottlenecks
         and disconnected island chambers, boosting moves that clear chokepoint blockers.
      4. Scores all 144 swap actions via spectral edge-difference + chokepoint-relief readout.
    """

    def __init__(
        self,
        in_channels: int = 20,
        board_h: int = 9,
        board_w: int = 9,
        d_model: int = 192,
        num_cheb_layers: int = 4,
        k_order: int = 4,
        num_actions: int = 144
    ):
        super().__init__()
        self.in_channels = in_channels
        self.board_h = board_h
        self.board_w = board_w
        self.num_nodes = board_h * board_w
        self.d_model = d_model
        self.num_cheb_layers = num_cheb_layers
        self.k_order = k_order
        self.num_actions = num_actions

        # 1. Dynamic Spectral Laplacian Builder
        self.laplacian_builder = DynamicSpectralLaplacianBuilder(height=board_h, width=board_w)

        # 2. Initial Spatial Stem + Coordinate Embeddings
        self.node_stem = nn.Sequential(
            nn.Conv2d(in_channels, d_model, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(d_model),
            nn.GELU(),
            nn.Conv2d(d_model, d_model, kernel_size=1),
            nn.GELU()
        )
        self.coord_embed = nn.Parameter(torch.randn(1, self.num_nodes, d_model) * 0.02)

        # 3. Stack of Chebyshev Spectral Graph Convolutions (K=4 polynomial orders)
        self.cheb_layers = nn.ModuleList([
            ChebyshevSpectralGraphConv(
                in_features=d_model,
                out_features=d_model,
                k_order=k_order
            )
            for _ in range(num_cheb_layers)
        ])

        # 4. Multi-Scale Spectral Graph Wavelet & Chokepoint Detector
        self.wavelet_detector = SpectralWaveletChokepointDetector(d_model=d_model, num_scales=3)
        self.final_norm = nn.LayerNorm(d_model)

        # 5. Chokepoint-Aware Pairwise Swap Policy Head (144 actions)
        self.policy_conv = nn.Sequential(
            nn.Conv2d(d_model + 1, 96, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(96),
            nn.GELU(),
            nn.Conv2d(96, 48, kernel_size=1),
            nn.GELU()
        )
        # Pairwise swap scorers receiving (h_u, h_v, |h_u - h_v|, chokepoint_u, chokepoint_v)
        pair_in_dim = d_model * 3 + 2
        self.horiz_swap_scorer = nn.Sequential(
            nn.Linear(pair_in_dim, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1)
        )
        self.vert_swap_scorer = nn.Sequential(
            nn.Linear(pair_in_dim, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1)
        )
        self.policy_fuse = nn.Linear(48 * self.num_nodes + num_actions, num_actions)

        # 6. Spectral Graph Value & Win Probability Readout Heads
        # Pools mean node feature, max node feature, chokepoint-weighted node feature, and algebraic connectivity
        readout_dim = d_model * 3 + 1
        self.value_head = nn.Sequential(
            nn.Linear(readout_dim, 128),
            nn.GELU(),
            nn.Linear(128, 1),
            nn.Tanh()
        )
        self.win_head = nn.Sequential(
            nn.Linear(readout_dim, 128),
            nn.GELU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def _score_spectral_swaps(self, h_2d: torch.Tensor, choke_2d: torch.Tensor) -> torch.Tensor:
        """
        Scores all 72 horizontal swaps and 72 vertical swaps using node embeddings,
        spectral gradient differences |h_u - h_v|, and topological chokepoint scores.
        """
        B, D, H, W = h_2d.shape

        # Horizontal pairs (r, c) <-> (r, c+1): 9x8 = 72
        h_l = h_2d[:, :, :, :-1].permute(0, 2, 3, 1)       # (B, 9, 8, D)
        h_r = h_2d[:, :, :, 1:].permute(0, 2, 3, 1)        # (B, 9, 8, D)
        c_l = choke_2d[:, :, :, :-1].permute(0, 2, 3, 1)   # (B, 9, 8, 1)
        c_r = choke_2d[:, :, :, 1:].permute(0, 2, 3, 1)    # (B, 9, 8, 1)
        horiz_in = torch.cat([h_l, h_r, (h_l - h_r).abs(), c_l, c_r], dim=-1).reshape(B, 72, -1)
        horiz_logits = self.horiz_swap_scorer(horiz_in).squeeze(-1)  # (B, 72)

        # Vertical pairs (r, c) <-> (r+1, c): 8x9 = 72
        v_t = h_2d[:, :, :-1, :].permute(0, 2, 3, 1)       # (B, 8, 9, D)
        v_b = h_2d[:, :, 1:, :].permute(0, 2, 3, 1)        # (B, 8, 9, D)
        cv_t = choke_2d[:, :, :-1, :].permute(0, 2, 3, 1)  # (B, 8, 9, 1)
        cv_b = choke_2d[:, :, 1:, :].permute(0, 2, 3, 1)   # (B, 8, 9, 1)
        vert_in = torch.cat([v_t, v_b, (v_t - v_b).abs(), cv_t, cv_b], dim=-1).reshape(B, 72, -1)
        vert_logits = self.vert_swap_scorer(vert_in).squeeze(-1)     # (B, 72)

        return torch.cat([horiz_logits, vert_logits], dim=-1)        # (B, 144)

    def forward(
        self,
        board_tensor: torch.Tensor,
        legal_mask: Optional[torch.Tensor] = None,
        portal_pairs: Optional[List[List[Tuple[Tuple[int, int], Tuple[int, int]]]]] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Executes Normalized Graph Laplacian construction, K=4 Chebyshev Spectral Convolutions,
        and Spectral Wavelet Chokepoint detection.
        """
        B, C, H, W = board_tensor.shape

        # Step 1: Compute Normalized Graph Laplacian L = I - D^{-1/2} A D^{-1/2}
        lap_info = self.laplacian_builder(board_tensor, portal_pairs=portal_pairs)
        norm_laplacian = lap_info["normalized_laplacian"]
        rescaled_laplacian = lap_info["rescaled_laplacian"]
        node_degrees = lap_info["node_degrees"]
        active_mask = lap_info["active_mask"]

        # Step 2: Initial node feature encoding
        stem_2d = self.node_stem(board_tensor)
        h = (stem_2d.flatten(2).transpose(1, 2) + self.coord_embed) * active_mask.unsqueeze(-1)

        # Step 3: K=4 Chebyshev Spectral Graph Convolutions
        cheb_band_energies = []
        for cheb_layer in self.cheb_layers:
            h, band_energy = cheb_layer(h, rescaled_laplacian, active_mask)
            cheb_band_energies.append(band_energy)

        # Step 4: Multi-Scale Spectral Wavelet Chokepoint Detection
        wavelet_out = self.wavelet_detector(h, norm_laplacian, node_degrees, active_mask)
        h = self.final_norm(h + wavelet_out["wavelet_features"]) * active_mask.unsqueeze(-1)

        chokepoint_scores = wavelet_out["chokepoint_scores"]  # (B, 81, 1)
        chokepoint_map = chokepoint_scores.transpose(1, 2).view(B, 1, H, W)

        # Step 5: Policy Readout over 144 Swaps
        h_2d = h.transpose(1, 2).view(B, self.d_model, H, W)
        conv_in = torch.cat([h_2d, chokepoint_map], dim=1)
        conv_flat = self.policy_conv(conv_in).flatten(1)
        pair_logits = self._score_spectral_swaps(h_2d, chokepoint_map)
        policy_logits = pair_logits + self.policy_fuse(torch.cat([conv_flat, pair_logits], dim=-1))

        if legal_mask is not None:
            policy_logits = policy_logits.masked_fill(~legal_mask.bool(), -1e9)

        # Step 6: Chokepoint-Weighted Spectral Graph Value Readout
        h_mean = h.mean(dim=1)
        h_max = h.max(dim=1).values
        choke_weights = chokepoint_scores / (chokepoint_scores.sum(dim=1, keepdim=True) + 1e-6)
        h_choke_pool = (h * choke_weights).sum(dim=1)
        alg_conn = wavelet_out["algebraic_connectivity_proxy"]

        global_readout = torch.cat([h_mean, h_max, h_choke_pool, alg_conn], dim=-1)
        value = self.value_head(global_readout)
        win_prob = self.win_head(global_readout)

        return {
            "policy_logits": policy_logits,
            "value": value,
            "win_prob": win_prob,
            "normalized_laplacian": norm_laplacian,
            "chokepoint_map": chokepoint_map,
            "algebraic_connectivity_proxy": alg_conn,
            "spectral_wavelet_energy": wavelet_out["spectral_wavelet_energy"],
            "chebyshev_band_energy": torch.stack(cheb_band_energies, dim=1),  # (B, num_layers, K+1)
        }

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


if __name__ == "__main__":
    torch.manual_seed(42)
    model = SpectralGraphWaveletCandyNet()
    params = model.count_parameters()
    print(f"[SpectralGraphWaveletCandyNet] Trainable Parameters: {params:,}")

    # Create synthetic board with a narrow 1-cell chokepoint bridge at (4, 4)
    dummy_board = torch.zeros(2, 20, 9, 9)
    dummy_board[:, 18] = 1.0  # Active cells
    # Carve out void holes leaving (4, 4) as a 1-cell bottleneck bridge between top & bottom chambers
    dummy_board[0, 18, 4, :4] = 0.0
    dummy_board[0, 18, 4, 5:] = 0.0
    # Add frosting blockers around the bridge
    dummy_board[0, 11, 3, 4] = 1.0
    dummy_board[0, 16, 8, 0] = 1.0  # Portal entrance
    dummy_board[0, 17, 0, 8] = 1.0  # Portal exit

    out = model(dummy_board)
    print("  policy_logits:", out["policy_logits"].shape)
    print("  value:", out["value"].shape)
    print("  win_prob:", out["win_prob"].shape)
    print("  normalized_laplacian:", out["normalized_laplacian"].shape)
    print("  chokepoint_map:", out["chokepoint_map"].shape)
    print("  algebraic_connectivity_proxy:", out["algebraic_connectivity_proxy"].shape)
    print("  spectral_wavelet_energy:", out["spectral_wavelet_energy"].shape)
    print("  chebyshev_band_energy:", out["chebyshev_band_energy"].shape)
