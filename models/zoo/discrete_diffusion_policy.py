"""
CandyDiffusionPolicyNet: Score-Based Iterative Denoising Action Policy for Multimodal Match-3 Boards.
Based on:
  - Chi et al. (RSS 2023), "Diffusion Policy: Visuomotor Policy Learning via Action Diffusion"
  - Song et al. (ICLR 2021), "Denoising Diffusion Implicit Models (DDIM)"
  - Austin et al. (NeurIPS 2021), "Structured Denoising Diffusion Models in Discrete State-Spaces"

Why Diffusion for Candy Crush?
In disconnected island topologies or symmetric multi-blocker boards, multiple distinct optimal regions
exist (e.g., clearing top-left chocolate vs. bottom-right frostings). Standard cross-entropy policies
average across modes or suffer entropy collapse. Iterative DDIM denoising over T=4 steps locks onto
a single coherent strategic mode conditioned on spatial board context.
"""

import math
from typing import Dict, Tuple, Optional, List
import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalTimestepEmbedding(nn.Module):
    """Sinusoidal diffusion timestep embedding mapping t in [0, T] to R^{d_model}."""
    def __init__(self, d_model: int = 256):
        super().__init__()
        self.d_model = d_model
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.SiLU(),
            nn.Linear(d_model * 2, d_model)
        )

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        """
        Args:
            timesteps: Tensor of shape (B,) containing diffusion step indices or continuous t in [0, 1]
        """
        half_dim = self.d_model // 2
        freqs = torch.exp(
            -math.log(10000.0) * torch.arange(half_dim, device=timesteps.device, dtype=torch.float32) / max(1, half_dim - 1)
        )
        args = timesteps.float().unsqueeze(-1) * freqs.unsqueeze(0)
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if self.d_model % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return self.mlp(emb)


class AdaLNDenoisingBlock(nn.Module):
    """
    Adaptive LayerNorm (AdaLN-Zero / DiT-style) Denoising Block.
    Modulates the evolving 144-action score vector using both the diffusion timestep t
    and cross-attention into the 81 spatial board tokens.
    """
    def __init__(self, d_model: int = 256, num_heads: int = 8):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model, elementwise_affine=False)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            batch_first=True
        )
        self.norm2 = nn.LayerNorm(d_model, elementwise_affine=False)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.SiLU(),
            nn.Linear(d_model * 4, d_model)
        )
        # AdaLN modulation parameters (shift_1, scale_1, gate_1, shift_2, scale_2, gate_2)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(d_model, d_model * 6)
        )

    def forward(
        self,
        action_latents: torch.Tensor,
        spatial_tokens: torch.Tensor,
        cond_vec: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            action_latents: (B, 1, d_model) current noisy action state representation
            spatial_tokens: (B, 81, d_model) encoded 9x9 board tokens
            cond_vec: (B, d_model) fused timestep + global board conditioning vector
        """
        mod = self.adaLN_modulation(cond_vec).unsqueeze(1)  # (B, 1, 6*d_model)
        shift1, scale1, gate1, shift2, scale2, gate2 = mod.chunk(6, dim=-1)

        # 1. AdaLN-modulated Cross-Attention to 81 spatial board cells
        q = self.norm1(action_latents) * (1.0 + scale1) + shift1
        attn_out, _ = self.cross_attn(query=q, key=spatial_tokens, value=spatial_tokens, need_weights=False)
        action_latents = action_latents + torch.tanh(gate1) * attn_out

        # 2. AdaLN-modulated MLP refinement
        h = self.norm2(action_latents) * (1.0 + scale2) + shift2
        action_latents = action_latents + torch.tanh(gate2) * self.mlp(h)
        return action_latents


class CandyDiffusionPolicyNet(nn.Module):
    """
    Iterative Score-Based / DDIM Denoising Action Policy Network for Candy Crush Saga.
    Operates over T=4 cosine-scheduled denoising steps in the 144-dimensional simplex logit space.
    """
    def __init__(
        self,
        in_channels: int = 20,
        d_model: int = 256,
        num_spatial_blocks: int = 5,
        num_denoising_blocks: int = 4,
        num_heads: int = 8,
        num_actions: int = 144,
        num_diffusion_steps: int = 4,
        board_h: int = 9,
        board_w: int = 9
    ):
        super().__init__()
        self.in_channels = in_channels
        self.d_model = d_model
        self.num_actions = num_actions
        self.num_diffusion_steps = num_diffusion_steps
        self.board_h = board_h
        self.board_w = board_w

        # Precompute cosine noise schedule alphas_cumprod for t = 1..T
        steps = torch.arange(num_diffusion_steps + 1, dtype=torch.float32) / num_diffusion_steps
        alpha_bar = torch.cos((steps + 0.008) / 1.008 * (math.pi / 2.0)) ** 2
        alpha_bar = alpha_bar / alpha_bar[0]
        self.register_buffer("alphas_cumprod", alpha_bar[1:].clamp(min=1e-4, max=0.9999))

        # 1. Spatial Board Encoder -> (B, 81, d_model)
        self.spatial_stem = nn.Sequential(
            nn.Conv2d(in_channels, d_model, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(d_model),
            nn.SiLU()
        )
        self.spatial_blocks = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(d_model, d_model, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(d_model),
                nn.SiLU(),
                nn.Conv2d(d_model, d_model, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(d_model)
            )
            for _ in range(num_spatial_blocks)
        ])
        self.pos_embed = nn.Parameter(torch.randn(1, board_h * board_w, d_model) * 0.02)

        # Warm-start prior head providing initial multimodal mean estimate a_prior
        self.prior_policy_head = nn.Sequential(
            nn.Conv2d(d_model, 64, kernel_size=1),
            nn.SiLU(),
            nn.Flatten(),
            nn.Linear(64 * board_h * board_w, 256),
            nn.SiLU(),
            nn.Linear(256, num_actions)
        )

        # 2. Denoising Score Network epsilon_theta(x_t, t, s)
        self.time_embed = SinusoidalTimestepEmbedding(d_model=d_model)
        self.noisy_action_proj = nn.Sequential(
            nn.Linear(num_actions, d_model),
            nn.LayerNorm(d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model)
        )
        self.denoising_blocks = nn.ModuleList([
            AdaLNDenoisingBlock(d_model=d_model, num_heads=num_heads)
            for _ in range(num_denoising_blocks)
        ])
        self.score_out_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.SiLU(),
            nn.Linear(d_model, num_actions)
        )

        # 3. Value / Win Probability & Energy Guidance Head
        self.win_head = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.SiLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def encode_board(self, board_tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Encodes (B, 20, 9, 9) board into spatial tokens, global vector, and warm-start prior logits."""
        h = self.spatial_stem(board_tensor)
        for block in self.spatial_blocks:
            h = F.silu(h + block(h))
        prior_logits = self.prior_policy_head(h)
        spatial_tokens = h.flatten(2).transpose(1, 2) + self.pos_embed  # (B, 81, d_model)
        global_ctx = spatial_tokens.mean(dim=1)                        # (B, d_model)
        return spatial_tokens, global_ctx, prior_logits

    def denoise_step(
        self,
        noisy_action_logits: torch.Tensor,
        timestep_t: torch.Tensor,
        spatial_tokens: torch.Tensor,
        global_ctx: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Single denoising score evaluation predicting both the noise residual epsilon_theta
        and the clean action preference x_0_hat at diffusion step t.
        """
        t_emb = self.time_embed(timestep_t)
        cond_vec = global_ctx + t_emb

        act_lat = self.noisy_action_proj(noisy_action_logits).unsqueeze(1)  # (B, 1, d_model)
        for block in self.denoising_blocks:
            act_lat = block(act_lat, spatial_tokens, cond_vec)

        pred_score_delta = self.score_out_head(act_lat.squeeze(1))  # (B, 144)
        return pred_score_delta

    def forward(
        self,
        board_tensor: torch.Tensor,
        noisy_action_logits: Optional[torch.Tensor] = None,
        timestep_t: Optional[torch.Tensor] = None,
        num_steps: Optional[int] = None,
        eta: float = 0.0
    ) -> Dict[str, torch.Tensor]:
        """
        Supports two execution modes:
          1. Single-Step Denoising Mode (when `noisy_action_logits` and `timestep_t` are provided):
             Computes the score update at timestep `timestep_t` for diffusion training.
          2. Full Iterative DDIM Reverse Denoising (when called with `board_tensor` alone):
             Iteratively denoises over T=4 cosine steps to lock onto a single coherent action mode.
        """
        B = board_tensor.shape[0]
        device = board_tensor.device
        spatial_tokens, global_ctx, prior_logits = self.encode_board(board_tensor)
        win_prob = self.win_head(global_ctx)

        # Mode 1: Single-step score evaluation at specified (noisy_action_logits, timestep_t)
        if noisy_action_logits is not None and timestep_t is not None:
            if timestep_t.dim() == 0:
                timestep_t = timestep_t.expand(B)
            score_delta = self.denoise_step(noisy_action_logits, timestep_t, spatial_tokens, global_ctx)
            denoised_logits = prior_logits + score_delta
            return {
                "policy_logits": denoised_logits,
                "win_prob": win_prob,
                "value": win_prob,
                "predicted_score": score_delta,
                "prior_logits": prior_logits
            }

        # Mode 2: Iterative T=4 DDIM Reverse Denoising Trajectory
        T = num_steps if num_steps is not None else self.num_diffusion_steps
        x_t = prior_logits.clone()
        trajectory_steps: List[torch.Tensor] = [x_t]

        for step_idx in reversed(range(T)):
            t_tensor = torch.full((B,), float(step_idx + 1) / float(T), device=device)
            alpha_t = self.alphas_cumprod[min(step_idx, len(self.alphas_cumprod) - 1)]
            alpha_prev = self.alphas_cumprod[step_idx - 1] if step_idx > 0 else torch.tensor(1.0, device=device)

            # Predict clean mode x_0_hat via score network
            score_delta = self.denoise_step(x_t, t_tensor, spatial_tokens, global_ctx)
            x_0_hat = prior_logits + score_delta

            # DDIM deterministic (eta=0) or stochastic update toward x_0_hat
            pred_eps = (x_t - torch.sqrt(alpha_t) * x_0_hat) / torch.sqrt(1.0 - alpha_t).clamp(min=1e-4)
            x_t = torch.sqrt(alpha_prev) * x_0_hat + torch.sqrt(1.0 - alpha_prev) * pred_eps
            if eta > 0.0 and step_idx > 0:
                x_t = x_t + eta * torch.sqrt(1.0 - alpha_prev) * torch.randn_like(x_t)

            trajectory_steps.append(x_t)

        return {
            "policy_logits": x_t,
            "win_prob": win_prob,
            "value": win_prob,
            "prior_logits": prior_logits,
            "denoising_trajectory": torch.stack(trajectory_steps, dim=1)  # (B, T+1, 144)
        }

    def compute_diffusion_loss(
        self,
        board_tensor: torch.Tensor,
        target_action_indices: torch.Tensor,
        target_win: torch.Tensor
    ) -> torch.Tensor:
        """
        Computes combined DDPM/DDIM denoising score matching loss + value calibration loss.
        """
        B = board_tensor.shape[0]
        device = board_tensor.device
        spatial_tokens, global_ctx, prior_logits = self.encode_board(board_tensor)
        win_prob = self.win_head(global_ctx)

        # Construct smooth target simplex preference centered on expert/MCTS action
        x_0 = F.one_hot(target_action_indices, num_classes=self.num_actions).float() * 6.0 - 3.0
        step_idx = torch.randint(0, self.num_diffusion_steps, (B,), device=device)
        alpha_bar = self.alphas_cumprod[step_idx].unsqueeze(-1)
        noise = torch.randn_like(x_0)
        x_t = torch.sqrt(alpha_bar) * x_0 + torch.sqrt(1.0 - alpha_bar) * noise

        t_norm = (step_idx.float() + 1.0) / float(self.num_diffusion_steps)
        score_delta = self.denoise_step(x_t, t_norm, spatial_tokens, global_ctx)
        x_0_pred = prior_logits + score_delta

        score_loss = F.mse_loss(x_0_pred, x_0) + F.cross_entropy(x_0_pred, target_action_indices)
        value_loss = F.binary_cross_entropy(win_prob, target_win.view(B, 1).float())
        return score_loss + 0.5 * value_loss

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def create_candy_diffusion_policy_net(device: str = "cpu") -> CandyDiffusionPolicyNet:
    model = CandyDiffusionPolicyNet(
        in_channels=20,
        d_model=256,
        num_spatial_blocks=5,
        num_denoising_blocks=4,
        num_heads=8,
        num_actions=144,
        num_diffusion_steps=4
    )
    return model.to(device)


if __name__ == "__main__":
    torch.manual_seed(42)
    net = create_candy_diffusion_policy_net("cpu")
    params = net.count_parameters()
    print(f"[CandyDiffusionPolicyNet] Initialized with {params:,} trainable parameters ({params / 1e6:.2f}M)")

    dummy_board = torch.randn(2, 20, 9, 9)
    # Test 1: Full T=4 DDIM Iterative Denoising Pass
    out_ddim = net(dummy_board)
    print(f"  [T=4 DDIM Mode] policy_logits:        {out_ddim['policy_logits'].shape}")
    print(f"  [T=4 DDIM Mode] win_prob:             {out_ddim['win_prob'].shape}")
    print(f"  [T=4 DDIM Mode] denoising_trajectory: {out_ddim['denoising_trajectory'].shape}")

    # Test 2: Explicit (board_tensor, noisy_action_logits, timestep_t) Single Denoising Step
    noisy_logits = torch.randn(2, 144)
    t_step = torch.tensor([0.5, 0.75])
    out_step = net(dummy_board, noisy_action_logits=noisy_logits, timestep_t=t_step)
    print(f"  [Single Step Mode] predicted_score:   {out_step['predicted_score'].shape}")
