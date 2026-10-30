"""
AlphaCandy Model Zoo — 3-Step Autoregressive Move Sequence Planner
(`MultiStepActionPlannerNet`).

Based on:
  - Vinyals et al. (NeurIPS 2015), "Pointer Networks"
  - Kool et al. (ICLR 2019), "Attention, Learn to Solve Routing Problems!"

Standard reactive RL policies greedily score single moves a_t in isolation, failing to discover
multi-move setups where move 1 has low immediate score (e.g., making a 3-match higher up in a
column to drop a Striped candy directly beside a Color Bomb on move 2, then detonating the
Striped + Color Bomb board wipe on move 3).

`MultiStepActionPlannerNet` uses a 3-Step Autoregressive GRU + Transformer Pointer Decoder:
  - Encodes the 9x9 board into 81 spatial cell tokens and 144 explicit swap-edge embeddings.
  - Across planning steps k = 0, 1, 2 (corresponding to moves a_t, a_{t+1}, a_{t+2}):
    1. Updates the recurrent tactical state via a GRU cell conditioned on the previously selected swap.
    2. Applies cross-attention over the 81 board tokens and mutates the latent board state to
       reflect gravity drop and special-candy creation from the previous step.
    3. Computes Pointer Network attention logits over all 144 swap actions, producing the joint
       3-step plan tensor `multi_step_logits` in R^(B x 3 x 144).
"""

import math
from typing import Dict, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class SpatialSwapEdgeBank(nn.Module):
    """
    Constructs explicit action embeddings E_actions in R^(B x 144 x d_model) for all
    72 horizontal swaps (r, c)<->(r, c+1) and 72 vertical swaps (r, c)<->(r+1, c)
    directly from the 9x9 spatial feature map.
    """

    def __init__(self, d_model: int = 192):
        super().__init__()
        self.d_model = d_model
        self.horiz_proj = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.LayerNorm(d_model),
            nn.SiLU()
        )
        self.vert_proj = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.LayerNorm(d_model),
            nn.SiLU()
        )
        self.action_pos_embed = nn.Parameter(torch.randn(1, 144, d_model) * 0.02)

    def forward(self, feat_2d: torch.Tensor) -> torch.Tensor:
        """
        Args:
            feat_2d: (B, d_model, 9, 9)
        Returns:
            action_embeds: (B, 144, d_model)
        """
        B, D, H, W = feat_2d.shape

        # 72 Horizontal swap edges
        h_left = feat_2d[:, :, :, :-1].permute(0, 2, 3, 1)
        h_right = feat_2d[:, :, :, 1:].permute(0, 2, 3, 1)
        h_edges = self.horiz_proj(torch.cat([h_left, h_right], dim=-1).reshape(B, 72, 2 * D))

        # 72 Vertical swap edges
        v_top = feat_2d[:, :, :-1, :].permute(0, 2, 3, 1)
        v_bot = feat_2d[:, :, 1:, :].permute(0, 2, 3, 1)
        v_edges = self.vert_proj(torch.cat([v_top, v_bot], dim=-1).reshape(B, 72, 2 * D))

        return torch.cat([h_edges, v_edges], dim=1) + self.action_pos_embed


class AutoregressivePointerStep(nn.Module):
    """
    Single step of the GRU + Transformer Cross-Attention Pointer Decoder.
    Updates recurrent state h_k, attends to the 81 spatial board tokens, mutates the
    board tokens to simulate the planned move's causal effect, and points to 144 swap edges.
    """

    def __init__(self, d_model: int = 192, num_heads: int = 6):
        super().__init__()
        self.d_model = d_model
        self.gru_cell = nn.GRUCell(d_model, d_model)

        # Cross-attention from planner query h_k to 81 board tokens
        self.cross_attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=num_heads, batch_first=True)
        self.query_norm = nn.LayerNorm(d_model)

        # Latent board state transition updater (simulates board change after choosing move a_k)
        self.board_mutation_gate = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.Sigmoid()
        )
        self.board_mutation_update = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.LayerNorm(d_model),
            nn.SiLU()
        )

        # Scaled dot-product Pointer Network projections over the 144 action embeddings
        self.pointer_query = nn.Linear(d_model, d_model)
        self.pointer_key = nn.Linear(d_model, d_model)
        self.pointer_temperature = math.sqrt(d_model)

        # Step value / combo synergy estimator
        self.step_value_head = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.SiLU(),
            nn.Linear(64, 1)
        )

    def forward(
        self,
        prev_action_embed: torch.Tensor,
        h_prev: torch.Tensor,
        board_tokens: torch.Tensor,
        action_bank: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            prev_action_embed: (B, d_model) embedding of action selected at step k-1
            h_prev: (B, d_model) previous GRU hidden state
            board_tokens: (B, 81, d_model) current imagined spatial board tokens
            action_bank: (B, 144, d_model) current swap edge embeddings
        Returns:
            step_logits: (B, 144) pointer logits for step k
            h_next: (B, d_model) updated GRU hidden state
            next_board_tokens: (B, 81, d_model) mutated board tokens for step k+1
            step_val: (B, 1) predicted cumulative value at step k
            cell_attn_weights: (B, 81) spatial attention weights over the 81 cells
        """
        # 1. GRU state update
        h_gru = self.gru_cell(prev_action_embed, h_prev)  # (B, d_model)

        # 2. Cross-attention over 81 spatial board cells
        q = h_gru.unsqueeze(1)  # (B, 1, d_model)
        ctx, cell_attn = self.cross_attn(q, board_tokens, board_tokens, need_weights=True)
        h_next = self.query_norm(h_gru + ctx.squeeze(1))  # (B, d_model)

        # 3. Pointer Network scoring against the 144 swap-edge embeddings
        p_q = self.pointer_query(h_next).unsqueeze(1)     # (B, 1, d_model)
        p_k = self.pointer_key(action_bank)               # (B, 144, d_model)
        step_logits = torch.bmm(p_q, p_k.transpose(1, 2)).squeeze(1) / self.pointer_temperature  # (B, 144)

        # 4. Compute expected action embedding for latent board mutation
        step_probs = F.softmax(step_logits, dim=-1)       # (B, 144)
        expected_action_vec = torch.bmm(step_probs.unsqueeze(1), action_bank).squeeze(1)  # (B, d_model)

        # Mutate the 81 board tokens to imagine the post-move board for step k+1
        expanded_act = expected_action_vec.unsqueeze(1).expand_as(board_tokens)
        cat_tokens = torch.cat([board_tokens, expanded_act], dim=-1)
        gate = self.board_mutation_gate(cat_tokens)
        delta_tokens = self.board_mutation_update(cat_tokens)
        next_board_tokens = board_tokens + gate * delta_tokens

        # 5. Predict step value
        step_val = self.step_value_head(h_next)
        return step_logits, h_next, next_board_tokens, step_val, cell_attn.squeeze(1)


class MultiStepActionPlannerNet(nn.Module):
    """
    3-Step Autoregressive GRU/Transformer Pointer Decoder Network (`MultiStepActionPlannerNet`).
    Outputs a coordinated 3-move tactical sequence (a_t, a_{t+1}, a_{t+2}) in R^(B x 3 x 144)
    alongside canonical first-step `policy_logits` (B, 144), `value` (B, 1), and `win_prob` (B, 1).
    """

    def __init__(
        self,
        in_channels: int = 20,
        d_model: int = 192,
        num_encoder_layers: int = 4,
        horizon: int = 3,
        num_actions: int = 144,
        board_h: int = 9,
        board_w: int = 9
    ):
        super().__init__()
        self.in_channels = in_channels
        self.d_model = d_model
        self.num_encoder_layers = num_encoder_layers
        self.horizon = horizon
        self.num_actions = num_actions
        self.board_h = board_h
        self.board_w = board_w

        # 1. Convolutional + Transformer Spatial Encoder
        self.conv_stem = nn.Sequential(
            nn.Conv2d(in_channels, d_model, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(d_model),
            nn.SiLU(),
            nn.Conv2d(d_model, d_model, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(d_model),
            nn.SiLU()
        )
        self.pos_embed = nn.Parameter(torch.randn(1, board_h * board_w, d_model) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=6,
            dim_feedforward=d_model * 2,
            dropout=0.05,
            activation="gelu",
            batch_first=True,
            norm_first=True
        )
        self.spatial_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)

        # 2. Swap Edge Embedding Bank (maps 9x9 tokens -> 144 swap embeddings)
        self.edge_bank = SpatialSwapEdgeBank(d_model=d_model)

        # 3. Learnable Start-of-Plan (<SOP>) token and 3-step Autoregressive Pointer Decoder
        self.start_action_token = nn.Parameter(torch.randn(1, d_model) * 0.02)
        self.step_embed = nn.Embedding(horizon, d_model)
        self.pointer_decoder = AutoregressivePointerStep(d_model=d_model, num_heads=6)

        # 4. Lookahead Combo Synergy Aggregator (fuses step 0 logits with lookahead value advantage)
        self.combo_synergy_proj = nn.Sequential(
            nn.Linear(horizon * num_actions, num_actions),
            nn.Tanh()
        )
        self.synergy_scale = nn.Parameter(torch.tensor(0.25))

        # 5. Global Value & Win Probability Readout across the 3-step tactical plan
        self.global_value_head = nn.Sequential(
            nn.Linear(d_model * 2 + horizon, 128),
            nn.SiLU(),
            nn.Linear(128, 1)
        )

    def forward(
        self,
        board_tensor: torch.Tensor,
        legal_mask: Optional[torch.Tensor] = None,
        target_sequence: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            board_tensor: (B, 20, 9, 9) canonical state tensor
            legal_mask: Optional (B, 144) legal action mask for the current board state s_t
            target_sequence: Optional (B, 3) ground-truth action sequence for teacher-forced training

        Returns:
            Dictionary containing:
              - `policy_logits`: (B, 144) immediate move logits enhanced by 3-step lookahead synergy
              - `value`: (B, 1) expected game outcome in [-1, 1]
              - `win_prob`: (B, 1) calibrated win probability in [0, 1]
              - `multi_step_logits`: (B, 3, 144) autoregressive logits for (a_t, a_{t+1}, a_{t+2})
              - `planned_sequence`: (B, 3) selected 3-move tactical plan
              - `step_values`: (B, 3, 1) intermediate value estimates across the 3 steps
              - `spatial_attention_maps`: (B, 3, 9, 9) cell attention heatmaps per planning step
        """
        B, C, H, W = board_tensor.shape

        # Encode initial board into 81 tokens
        stem_feat = self.conv_stem(board_tensor)
        board_tokens = self.spatial_encoder(stem_feat.flatten(2).transpose(1, 2) + self.pos_embed)

        # Initialize GRU hidden state from global board context
        h_state = board_tokens.mean(dim=1)
        h_init = h_state
        prev_action_embed = self.start_action_token.expand(B, -1)

        step_logits_list = []
        step_values_list = []
        planned_actions_list = []
        spatial_attn_list = []

        for k in range(self.horizon):
            # Recompute 144 swap-edge embeddings from current imagined board tokens
            curr_2d = board_tokens.transpose(1, 2).contiguous().view(B, self.d_model, H, W)
            action_bank = self.edge_bank(curr_2d)  # (B, 144, d_model)

            step_pos = self.step_embed(torch.full((B,), k, device=board_tensor.device, dtype=torch.long))
            step_in = prev_action_embed + step_pos

            logits_k, h_state, board_tokens, val_k, cell_attn_k = self.pointer_decoder(
                step_in, h_state, board_tokens, action_bank
            )

            # Apply legal mask on step k = 0 (current known board state)
            if k == 0 and legal_mask is not None:
                logits_k = logits_k.masked_fill(~legal_mask.bool(), -1e9)

            step_logits_list.append(logits_k)
            step_values_list.append(val_k)
            spatial_attn_list.append(cell_attn_k.view(B, H, W))

            # Select action embedding to feed into step k + 1
            if target_sequence is not None:
                chosen_k = target_sequence[:, k]
                planned_actions_list.append(chosen_k)
                prev_action_embed = action_bank.gather(
                    1, chosen_k.view(B, 1, 1).expand(B, 1, self.d_model)
                ).squeeze(1)
            else:
                chosen_k = torch.argmax(logits_k, dim=-1)
                planned_actions_list.append(chosen_k)
                # Differentiable soft-pointer embedding feedback
                probs_k = F.softmax(logits_k, dim=-1)
                prev_action_embed = torch.bmm(probs_k.unsqueeze(1), action_bank).squeeze(1)

        # Stack 3-step trajectory tensors
        multi_step_logits = torch.stack(step_logits_list, dim=1)       # (B, 3, 144)
        step_values = torch.stack(step_values_list, dim=1)             # (B, 3, 1)
        planned_sequence = torch.stack(planned_actions_list, dim=1)    # (B, 3)
        spatial_attention_maps = torch.stack(spatial_attn_list, dim=1) # (B, 3, 9, 9)

        # First-step canonical policy_logits boosted by 3-step combo synergy
        synergy_bonus = self.combo_synergy_proj(multi_step_logits.flatten(1))
        policy_logits = multi_step_logits[:, 0, :] + self.synergy_scale * synergy_bonus
        if legal_mask is not None:
            policy_logits = policy_logits.masked_fill(~legal_mask.bool(), -1e9)

        # Global value conditioned on initial state, final imagined state, and 3-step value trajectory
        val_in = torch.cat([h_init, h_state, step_values.squeeze(-1)], dim=-1)
        val_raw = self.global_value_head(val_in)
        value = torch.tanh(val_raw)
        win_prob = torch.sigmoid(val_raw)

        return {
            "policy_logits": policy_logits,
            "value": value,
            "win_prob": win_prob,
            "multi_step_logits": multi_step_logits,
            "planned_sequence": planned_sequence,
            "step_values": step_values,
            "spatial_attention_maps": spatial_attention_maps,
        }

    def multi_step_sequence_loss(
        self,
        multi_step_logits: torch.Tensor,
        target_sequence: torch.Tensor,
        step_discounts: Tuple[float, float, float] = (1.0, 0.7, 0.5)
    ) -> torch.Tensor:
        """
        Discounted autoregressive cross-entropy loss across the 3-step tactical move sequence.
        """
        total_loss = torch.tensor(0.0, device=multi_step_logits.device)
        for k in range(self.horizon):
            w = step_discounts[k] if k < len(step_discounts) else 0.5
            ce_k = F.cross_entropy(multi_step_logits[:, k, :], target_sequence[:, k])
            total_loss = total_loss + w * ce_k
        return total_loss / sum(step_discounts[:self.horizon])

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def create_multi_step_action_planner_net(device: str = "cpu") -> MultiStepActionPlannerNet:
    model = MultiStepActionPlannerNet(
        in_channels=20,
        d_model=192,
        num_encoder_layers=4,
        horizon=3,
        num_actions=144
    )
    return model.to(device)


if __name__ == "__main__":
    torch.manual_seed(42)
    net = create_multi_step_action_planner_net("cpu")
    params = net.count_parameters()
    print(f"[MultiStepActionPlannerNet] Trainable parameters: {params:,} ({params / 1e6:.2f}M)")

    dummy_board = torch.randn(2, 20, 9, 9)
    dummy_targets = torch.tensor([[12, 45, 108], [3, 77, 19]])
    out = net(dummy_board)
    seq_loss = net.multi_step_sequence_loss(out["multi_step_logits"], dummy_targets)

    print(f"  policy_logits:          {out['policy_logits'].shape}")
    print(f"  value:                  {out['value'].shape}")
    print(f"  win_prob:               {out['win_prob'].shape}")
    print(f"  multi_step_logits:      {out['multi_step_logits'].shape}")
    print(f"  planned_sequence:       {out['planned_sequence'].tolist()}")
    print(f"  step_values:            {out['step_values'].shape}")
    print(f"  spatial_attention_maps: {out['spatial_attention_maps'].shape}")
    print(f"  sequence_loss:          {seq_loss.item():.4f}")
