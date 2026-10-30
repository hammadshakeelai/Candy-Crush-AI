"""
AlphaCandy Model Zoo — Token-Level Sparse Mixture-of-Experts Transformer
(`SwitchMoECandyTransformer`).

Based on:
  - Fedus, Zoph & Shazeer (JMLR 2022), "Switch Transformers: Scaling to Trillion Parameter Models
    with Simple and Efficient Sparsity"
  - Dai et al. (2024), "DeepSeekMoE: Towards Ultimate Expert Specialization in Mixture-of-Experts"
  - Zoph et al. (2022), "ST-MoE: Designing Stable and Transferable Sparse Expert Models"

Each of the 81 board cells in Candy Crush exhibits distinct local mechanics (regular color cells,
Striped/Wrapped/Color Bomb specials, multi-layer Jelly, spreading Chocolate, Ingredient chutes,
and Teleporter portals). Applying a monolithic dense FFN wastes capacity.

`SwitchMoECandyTransformer` replaces dense Transformer FFN blocks with a DeepSeekMoE / Switch
hybrid architecture featuring:
  - 2 Always-Active Shared SwiGLU Experts (capture universal match-3 & gravity physics).
  - 6 Routed Specialist SwiGLU Experts per layer with Top-2 token routing over the 81 cell tokens:
      * Expert 0: Basic Color Match & Cascade Specialist
      * Expert 1: Special Candy (Striped/Wrapped/Color Bomb) Combo Specialist
      * Expert 2: Jelly & Multi-Layer Frosting Fracture Specialist
      * Expert 3: Spreading Chocolate & Licorice Lock Containment Specialist
      * Expert 4: Ingredient Drop Column & Exit Chute Specialist
      * Expert 5: Teleporter Portal & Boundary Topology Specialist
  - Switch Load-Balancing Auxiliary Loss + Router Z-Loss for balanced, non-collapsing routing.
"""

from typing import Dict, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class SwiGLUExpertFFN(nn.Module):
    """
    Fine-Grained SwiGLU Feed-Forward Expert:
        FFN_SwiGLU(x) = W_down ( SiLU(W_gate x) * (W_up x) )
    """

    def __init__(self, d_model: int = 192, d_ff: int = 256, dropout: float = 0.05):
        super().__init__()
        self.w_gate = nn.Linear(d_model, d_ff, bias=False)
        self.w_up = nn.Linear(d_model, d_ff, bias=False)
        self.w_down = nn.Linear(d_ff, d_model, bias=False)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.w_down(F.silu(self.w_gate(x)) * self.w_up(x)))


class DeepSeekSwitchMoELayer(nn.Module):
    """
    Hybrid Shared + Top-2 Routed Sparse Mixture-of-Experts Layer over the 81 board cell tokens.
      - 2 Shared SwiGLU Experts process all 81 tokens unconditionally.
      - 6 Routed Specialist SwiGLU Experts receive Top-2 routed cell tokens via softmax router gates.
      - Computes Switch Transformer load-balancing auxiliary loss + ST-MoE router z-loss.
    """

    def __init__(
        self,
        d_model: int = 192,
        d_expert: int = 192,
        num_shared_experts: int = 2,
        num_routed_experts: int = 6,
        top_k: int = 2,
        dropout: float = 0.05
    ):
        super().__init__()
        self.d_model = d_model
        self.num_shared_experts = num_shared_experts
        self.num_routed_experts = num_routed_experts
        self.top_k = top_k

        # 2 Always-Active Shared Experts
        self.shared_experts = nn.ModuleList([
            SwiGLUExpertFFN(d_model=d_model, d_ff=d_expert, dropout=dropout)
            for _ in range(num_shared_experts)
        ])

        # 6 Routed Specialist Experts
        self.routed_experts = nn.ModuleList([
            SwiGLUExpertFFN(d_model=d_model, d_ff=d_expert, dropout=dropout)
            for _ in range(num_routed_experts)
        ])

        # Token-level Router: maps cell token (B, 81, d_model) + 6 domain prior hints -> (B, 81, 6)
        self.router_gate = nn.Linear(d_model, num_routed_experts, bias=False)
        self.domain_prior_scale = nn.Parameter(torch.tensor(0.3))

    def forward(
        self,
        x: torch.Tensor,
        domain_priors: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (B, 81, d_model) normalized cell tokens
            domain_priors: (B, 81, 6) cell-type semantic hints extracted from raw board channels
        Returns:
            moe_out: (B, 81, d_model) combined shared + top-2 routed expert outputs
            aux_loss: Scalar Switch load-balancing loss + router z-loss
            expert_utilization: (B, 6) fraction of tokens routed to each specialist expert
            top_indices: (B, 81, 2) indices of the Top-2 chosen experts per cell
        """
        B, N, D = x.shape

        # 1. Evaluate 2 Always-Active Shared Experts
        shared_out = torch.zeros_like(x)
        for s_exp in self.shared_experts:
            shared_out = shared_out + s_exp(x)
        shared_out = shared_out / float(self.num_shared_experts)

        # 2. Compute Router Logits & Probabilities over the 6 Specialist Experts
        router_logits = self.router_gate(x) + self.domain_prior_scale * domain_priors  # (B, 81, 6)
        router_probs = F.softmax(router_logits, dim=-1)                                # (B, 81, 6)

        # Select Top-2 experts per token
        top_weights, top_indices = torch.topk(router_probs, k=self.top_k, dim=-1)      # (B, 81, 2)
        # Normalize Top-2 gating weights to sum to 1
        top_weights = top_weights / (top_weights.sum(dim=-1, keepdim=True) + 1e-8)

        # 3. Dispatch tokens to the 6 Specialist Experts (vectorized mask dispatch for 81 cells)
        routed_out = torch.zeros_like(x)
        for exp_idx, expert in enumerate(self.routed_experts):
            # Find which tokens selected `exp_idx` in their Top-2 choices
            match_mask = (top_indices == exp_idx)  # (B, 81, 2)
            if match_mask.any():
                # Weight assigned to this expert for each token: (B, 81, 1)
                token_weight = (top_weights * match_mask.float()).sum(dim=-1, keepdim=True)
                exp_feat = expert(x)
                routed_out = routed_out + token_weight * exp_feat

        # 4. Compute Switch Transformer Load-Balancing Loss + ST-MoE Router Z-Loss
        # Fraction of tokens routed to each expert i: f_i in R^(B, 6)
        one_hot_top = F.one_hot(top_indices, num_classes=self.num_routed_experts).float()  # (B, 81, 2, 6)
        token_dispatch_frac = one_hot_top.sum(dim=2).mean(dim=1) / float(self.top_k)       # (B, 6)
        mean_router_prob = router_probs.mean(dim=1)                                        # (B, 6)

        # Switch load-balancing loss: E * sum_{i=1}^E (f_i * P_i)
        load_balance_loss = self.num_routed_experts * (token_dispatch_frac * mean_router_prob).sum(dim=-1).mean()

        # ST-MoE Router Z-Loss: penalizes large router logits for numerical stability
        router_z_loss = torch.logsumexp(router_logits, dim=-1).pow(2).mean()
        aux_loss = load_balance_loss + 0.01 * router_z_loss

        moe_out = shared_out + routed_out
        return moe_out, aux_loss, token_dispatch_frac, top_indices


class SwitchMoETransformerBlock(nn.Module):
    """
    Transformer Block with 2D Depthwise Spatial Convolution, Multi-Head Self-Attention,
    and DeepSeekSwitchMoELayer.
    """

    def __init__(
        self,
        d_model: int = 192,
        num_heads: int = 6,
        d_expert: int = 192,
        num_shared_experts: int = 2,
        num_routed_experts: int = 6,
        top_k: int = 2,
        dropout: float = 0.05
    ):
        super().__init__()
        self.d_model = d_model

        # Local 3x3 depthwise positional enhancement
        self.dw_conv = nn.Conv2d(d_model, d_model, kernel_size=3, padding=1, groups=d_model, bias=False)
        self.dw_bn = nn.BatchNorm2d(d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        self.norm2 = nn.LayerNorm(d_model)
        self.moe = DeepSeekSwitchMoELayer(
            d_model=d_model,
            d_expert=d_expert,
            num_shared_experts=num_shared_experts,
            num_routed_experts=num_routed_experts,
            top_k=top_k,
            dropout=dropout
        )

    def forward(
        self,
        x: torch.Tensor,
        domain_priors: torch.Tensor,
        H: int = 9,
        W: int = 9
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        B, N, D = x.shape

        # Local 3x3 depthwise conv residual
        x_2d = x.transpose(1, 2).contiguous().view(B, D, H, W)
        x = x + F.silu(self.dw_bn(self.dw_conv(x_2d))).flatten(2).transpose(1, 2)

        # Pre-Norm Multi-Head Self-Attention
        x_norm1 = self.norm1(x)
        attn_out, _ = self.attn(x_norm1, x_norm1, x_norm1, need_weights=False)
        x = x + attn_out

        # Pre-Norm Sparse Mixture-of-Experts FFN
        moe_out, aux_loss, util, top_idx = self.moe(self.norm2(x), domain_priors)
        x = x + moe_out
        return x, aux_loss, util, top_idx


class SwitchMoECandyTransformer(nn.Module):
    """
    Token-Level Sparse Mixture-of-Experts Transformer (`SwitchMoECandyTransformer`).
    Features 2 Shared Experts + 6 Routed Specialist Experts per layer with Top-2 routing over
    the 81 board cells and Switch load-balancing auxiliary loss.
    """

    def __init__(
        self,
        in_channels: int = 20,
        d_model: int = 192,
        num_layers: int = 5,
        num_heads: int = 6,
        d_expert: int = 192,
        num_shared_experts: int = 2,
        num_routed_experts: int = 6,
        top_k: int = 2,
        num_actions: int = 144,
        board_h: int = 9,
        board_w: int = 9
    ):
        super().__init__()
        self.in_channels = in_channels
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_shared_experts = num_shared_experts
        self.num_routed_experts = num_routed_experts
        self.top_k = top_k
        self.num_actions = num_actions
        self.board_h = board_h
        self.board_w = board_w

        # Spatial Cell Token Stem
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, d_model, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(d_model),
            nn.SiLU()
        )
        self.pos_embed = nn.Parameter(torch.randn(1, board_h * board_w, d_model) * 0.02)

        # Stack of SwitchMoETransformerBlocks
        self.blocks = nn.ModuleList([
            SwitchMoETransformerBlock(
                d_model=d_model,
                num_heads=num_heads,
                d_expert=d_expert,
                num_shared_experts=num_shared_experts,
                num_routed_experts=num_routed_experts,
                top_k=top_k
            )
            for _ in range(num_layers)
        ])
        self.final_norm = nn.LayerNorm(d_model)

        # Canonical 144-Action Swap Policy Head
        self.policy_conv = nn.Sequential(
            nn.Conv2d(d_model, 48, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(48),
            nn.SiLU()
        )
        self.policy_head = nn.Sequential(
            nn.Linear(48 * board_h * board_w, 256),
            nn.SiLU(),
            nn.Linear(256, num_actions)
        )

        # Value & Win Probability Head conditioned on global pooled tokens + expert utilization profile
        self.value_head = nn.Sequential(
            nn.Linear(d_model + num_routed_experts, 128),
            nn.SiLU(),
            nn.Linear(128, 1)
        )

    def _extract_cell_domain_priors(self, board_tensor: torch.Tensor) -> torch.Tensor:
        """
        Extracts 6 cell-level domain specialization hints from the 20-channel board tensor
        for each of the 81 cells:
          0: Regular color candy density (ch 0..5)
          1: Special candy presence (Striped ch 6,7, Wrapped ch 8, Color Bomb ch 9)
          2: Jelly / Frosting blocker thickness (ch 10, 11)
          3: Chocolate / Licorice lock presence (ch 12, 13, 14)
          4: Ingredient presence (ch 15)
          5: Portal / Teleporter wormhole presence (ch 16, 17)
        Returns shape (B, 81, 6).
        """
        B, C, H, W = board_tensor.shape
        p0 = board_tensor[:, 0:6].sum(dim=1, keepdim=True)
        p1 = board_tensor[:, 6:10].sum(dim=1, keepdim=True)
        p2 = board_tensor[:, 10:12].sum(dim=1, keepdim=True)
        p3 = board_tensor[:, 12:15].sum(dim=1, keepdim=True)
        p4 = board_tensor[:, 15:16]
        p5 = board_tensor[:, 16:18].sum(dim=1, keepdim=True)

        priors_2d = torch.cat([p0, p1, p2, p3, p4, p5], dim=1)  # (B, 6, 9, 9)
        return priors_2d.flatten(2).transpose(1, 2)              # (B, 81, 6)

    def forward(
        self,
        board_tensor: torch.Tensor,
        legal_mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass returning canonical `policy_logits` (B, 144), `value` (B, 1),
        `win_prob` (B, 1), `moe_aux_loss` (scalar), `expert_utilization` (B, 6),
        and `routing_map` (B, num_layers, 81, 2).
        """
        B, C, H, W = board_tensor.shape
        domain_priors = self._extract_cell_domain_priors(board_tensor)  # (B, 81, 6)

        tokens = self.stem(board_tensor).flatten(2).transpose(1, 2) + self.pos_embed

        total_aux_loss = torch.tensor(0.0, device=board_tensor.device)
        util_per_layer = []
        routing_per_layer = []

        for block in self.blocks:
            tokens, layer_aux, layer_util, layer_top_idx = block(tokens, domain_priors, H=H, W=W)
            total_aux_loss = total_aux_loss + layer_aux
            util_per_layer.append(layer_util)
            routing_per_layer.append(layer_top_idx)

        tokens = self.final_norm(tokens)
        moe_aux_loss = total_aux_loss / float(self.num_layers)
        expert_utilization = torch.stack(util_per_layer, dim=1).mean(dim=1)  # (B, 6)
        routing_map = torch.stack(routing_per_layer, dim=1)                  # (B, L, 81, 2)

        # Policy Head
        feat_2d = tokens.transpose(1, 2).contiguous().view(B, self.d_model, H, W)
        policy_logits = self.policy_head(self.policy_conv(feat_2d).flatten(1))

        if legal_mask is not None:
            policy_logits = policy_logits.masked_fill(~legal_mask.bool(), -1e9)

        # Value & Win Probability Head
        global_feat = tokens.mean(dim=1)
        val_raw = self.value_head(torch.cat([global_feat, expert_utilization], dim=-1))
        value = torch.tanh(val_raw)
        win_prob = torch.sigmoid(val_raw)

        return {
            "policy_logits": policy_logits,
            "value": value,
            "win_prob": win_prob,
            "moe_aux_loss": moe_aux_loss,
            "expert_utilization": expert_utilization,
            "routing_map": routing_map,
        }

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def create_switch_moe_candy_transformer(device: str = "cpu") -> SwitchMoECandyTransformer:
    model = SwitchMoECandyTransformer(
        in_channels=20,
        d_model=192,
        num_layers=5,
        num_heads=6,
        d_expert=192,
        num_shared_experts=2,
        num_routed_experts=6,
        top_k=2,
        num_actions=144
    )
    return model.to(device)


if __name__ == "__main__":
    torch.manual_seed(42)
    net = create_switch_moe_candy_transformer("cpu")
    params = net.count_parameters()
    print(f"[SwitchMoECandyTransformer] Trainable parameters: {params:,} ({params / 1e6:.2f}M)")

    dummy_board = torch.rand(2, 20, 9, 9)
    out = net(dummy_board)
    print(f"  policy_logits:      {out['policy_logits'].shape}")
    print(f"  value:              {out['value'].shape}")
    print(f"  win_prob:           {out['win_prob'].shape}")
    print(f"  moe_aux_loss:       {out['moe_aux_loss'].item():.4f}")
    print(f"  expert_utilization: {out['expert_utilization'].shape} (sum={out['expert_utilization'][0].sum().item():.2f})")
    print(f"  routing_map:        {out['routing_map'].shape}")
