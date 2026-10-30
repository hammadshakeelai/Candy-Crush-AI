"""
Darwinian Neuro-Evolutionary Strategy & CMA-ES / PEPG Ensemble (`CMAESNeuroEvolutionAgent`).
Based on:
  - Hansen & Ostermeier (Evolutionary Computation 2001), "Completely Derandomized Self-Adaptation in Evolution Strategies (CMA-ES)"
  - Sehnke et al. (Neural Networks 2010), "Parameter-exploring Policy Gradients (PEPG)"
  - Salimans et al. (OpenAI 2017), "Evolution Strategies as a Scalable Alternative to RL"

Maintains a population of M specialist policy-head perturbations around a shared deep residual
spatial backbone, parameterized by a Gaussian search distribution N(mu, sigma^2 (D^2 + L L^T))
in latent policy-parameter space. Supports both differentiable Darwinian fitness-weighted ensemble
inference and explicit CMA-ES / PEPG covariance matrix adaptation across attempts on stubborn levels.
"""

import math
from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class EvolutionaryResBlock(nn.Module):
    """Residual spatial block with Squeeze-and-Excitation channel gating."""
    def __init__(self, channels: int = 256):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, max(16, channels // 8), kernel_size=1),
            nn.SiLU(),
            nn.Conv2d(max(16, channels // 8), channels, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.silu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out * self.se(out)
        return F.silu(x + out)


class LowRankCMAESPopulationHead(nn.Module):
    """
    Maintains a population of M policy head weight perturbations governed by a Low-Rank + Diagonal
    Covariance Matrix Adaptation Evolution Strategy (VD-CMA-ES / PEPG):
        theta_m = mu + sigma * (diag(d) * z_m + L * u_m)
    where theta_m parameterizes a rank-R perturbation of the 256 -> 144 policy projection matrix.
    """
    def __init__(
        self,
        feat_dim: int = 256,
        num_actions: int = 144,
        population_size: int = 8,
        cov_rank: int = 4
    ):
        super().__init__()
        self.feat_dim = feat_dim
        self.num_actions = num_actions
        self.population_size = population_size
        self.cov_rank = cov_rank

        # Compact subspace dimension for fast evolutionary adaptation:
        # Each individual m perturbs a (num_actions + feat_dim) modulation vector
        self.gene_dim = feat_dim + num_actions

        # Base deterministic policy projection
        self.base_policy_fc = nn.Linear(feat_dim, num_actions)

        # Learnable evolutionary distribution center mu in R^{gene_dim}
        self.evo_mean = nn.Parameter(torch.zeros(self.gene_dim))
        # Diagonal standard deviation vector d in R^{gene_dim}
        self.evo_log_diag = nn.Parameter(torch.zeros(self.gene_dim) - 1.5)
        # Low-rank covariance factor matrix L in R^{gene_dim x cov_rank}
        self.evo_low_rank_L = nn.Parameter(torch.randn(self.gene_dim, cov_rank) * 0.05)

        # Standing population of M specialist perturbation genomes (antithetic mirrored pairs)
        init_pop = torch.randn(population_size, self.gene_dim) * 0.15
        self.population_genomes = nn.Parameter(init_pop)

        # CMA-ES cumulative evolution path buffers p_sigma and p_c
        self.register_buffer("step_size_sigma", torch.tensor(0.25))
        self.register_buffer("evo_path_sigma", torch.zeros(self.gene_dim))
        self.register_buffer("evo_path_c", torch.zeros(self.gene_dim))
        self.register_buffer("generation_counter", torch.tensor(0, dtype=torch.long))

    def sample_antithetic_population(self) -> torch.Tensor:
        """
        Samples M antithetic (mirror-sampled) candidate genomes from N(mu, sigma^2 (D^2 + L L^T)).
        Returns:
            genomes: Tensor of shape (M, gene_dim)
        """
        half_m = self.population_size // 2
        device = self.evo_mean.device

        eps_diag = torch.randn(half_m, self.gene_dim, device=device)
        eps_diag = torch.cat([eps_diag, -eps_diag], dim=0)  # (M, gene_dim) antithetic pairs

        eps_rank = torch.randn(half_m, self.cov_rank, device=device)
        eps_rank = torch.cat([eps_rank, -eps_rank], dim=0)  # (M, cov_rank)

        diag_std = torch.exp(self.evo_log_diag).clamp(min=1e-3, max=2.0)
        perturbation = diag_std.unsqueeze(0) * eps_diag + torch.matmul(eps_rank, self.evo_low_rank_L.t())
        genomes = self.evo_mean.unsqueeze(0) + self.step_size_sigma * perturbation
        return genomes

    def evaluate_population(
        self,
        latent_feat: torch.Tensor,
        resample: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Evaluates all M population policy heads in parallel on `latent_feat` (B, feat_dim).
        Returns:
            population_logits: (B, M, 144)
            active_genomes: (M, gene_dim)
        """
        B, F_dim = latent_feat.shape
        if resample:
            genomes = self.sample_antithetic_population()
        else:
            diag_std = torch.exp(self.evo_log_diag).clamp(min=1e-3, max=2.0)
            genomes = self.evo_mean.unsqueeze(0) + diag_std.unsqueeze(0) * self.population_genomes

        # Split each genome into feature multiplicative gate g_f in R^{feat_dim} and action bias g_a in R^{144}
        g_feat = 1.0 + 0.3 * torch.tanh(genomes[:, :self.feat_dim])    # (M, feat_dim)
        g_act = 0.3 * genomes[:, self.feat_dim:]                       # (M, 144)

        # Modulate latent features for each of the M individuals: (B, 1, F) * (1, M, F) -> (B, M, F)
        modulated_feat = latent_feat.unsqueeze(1) * g_feat.unsqueeze(0)
        population_logits = self.base_policy_fc(modulated_feat) + g_act.unsqueeze(0)  # (B, M, 144)
        return population_logits, genomes

    @torch.no_grad()
    def update_cmaes(
        self,
        candidate_genomes: torch.Tensor,
        fitness_scores: torch.Tensor,
        c_m: float = 0.5,
        c_cov: float = 0.15,
        c_sigma: float = 0.2
    ) -> Dict[str, float]:
        """
        Executes a complete CMA-ES / PEPG distribution update given M candidate genomes
        and their empirical episode returns `fitness_scores` of shape (M,).
        Adapts mean `evo_mean`, diagonal variance `evo_log_diag`, evolution path `evo_path_c`,
        low-rank covariance factor `evo_low_rank_L`, and global step size `step_size_sigma`.
        """
        M = candidate_genomes.shape[0]
        mu_eff_count = max(2, M // 2)

        # Rank candidates by descending fitness
        sorted_fitness, rank_idx = torch.sort(fitness_scores, descending=True)
        elite_genomes = candidate_genomes[rank_idx[:mu_eff_count]]  # (mu_eff_count, gene_dim)

        # Recombination weights w_i proportional to log(mu + 0.5) - log(i) (Hansen & Ostermeier, 2001)
        ranks = torch.arange(1, mu_eff_count + 1, dtype=torch.float32, device=candidate_genomes.device)
        raw_weights = math.log(mu_eff_count + 0.5) - torch.log(ranks)
        weights = raw_weights / raw_weights.sum()  # (mu_eff_count,)
        mu_eff = 1.0 / (weights.pow(2).sum().item() + 1e-8)

        # 1. Update distribution mean mu_{t+1}
        old_mean = self.evo_mean.data.clone()
        weighted_elite = (weights.unsqueeze(-1) * elite_genomes).sum(dim=0)
        mean_step = (weighted_elite - old_mean) / (self.step_size_sigma + 1e-6)
        self.evo_mean.data.copy_(old_mean + c_m * (weighted_elite - old_mean))

        # 2. Update anisotropic evolution path p_c
        self.evo_path_c.mul_(1.0 - c_cov).add_(math.sqrt(c_cov * (2.0 - c_cov) * mu_eff) * mean_step)

        # 3. Rank-mu + Rank-1 Diagonal Covariance adaptation (PEPG / sep-CMA-ES)
        elite_centered = (elite_genomes - old_mean.unsqueeze(0)) / (self.step_size_sigma + 1e-6)
        diag_var_est = (weights.unsqueeze(-1) * elite_centered.pow(2)).sum(dim=0)
        current_var = torch.exp(2.0 * self.evo_log_diag.data)
        new_var = (1.0 - c_cov) * current_var + c_cov * (0.5 * diag_var_est + 0.5 * self.evo_path_c.pow(2))
        self.evo_log_diag.data.copy_(0.5 * torch.log(new_var.clamp(min=1e-5, max=4.0)))

        # 4. Update low-rank covariance factor column via cyclic rank-1 evolution path replacement
        gen_idx = int(self.generation_counter.item()) % self.cov_rank
        self.evo_low_rank_L.data[:, gen_idx] = (
            (1.0 - c_cov) * self.evo_low_rank_L.data[:, gen_idx] + c_cov * self.evo_path_c
        )

        # 5. Cumulative step-size control (CSA)
        self.evo_path_sigma.mul_(1.0 - c_sigma).add_(math.sqrt(c_sigma * (2.0 - c_sigma) * mu_eff) * mean_step)
        expected_norm = math.sqrt(self.gene_dim)
        path_norm_ratio = (self.evo_path_sigma.norm().item() / expected_norm) - 1.0
        new_sigma = float(self.step_size_sigma.item()) * math.exp(0.1 * max(-1.0, min(1.0, path_norm_ratio)))
        self.step_size_sigma.fill_(max(0.05, min(1.5, new_sigma)))

        # Store updated elites into standing population
        self.population_genomes.data[:mu_eff_count].copy_(elite_centered)
        self.generation_counter.add_(1)

        return {
            "generation": int(self.generation_counter.item()),
            "best_fitness": float(sorted_fitness[0].item()),
            "mean_elite_fitness": float(sorted_fitness[:mu_eff_count].mean().item()),
            "step_size_sigma": float(self.step_size_sigma.item()),
            "mean_diag_std": float(torch.exp(self.evo_log_diag.data).mean().item())
        }


class CMAESNeuroEvolutionAgent(nn.Module):
    """
    Darwinian Neuro-Evolutionary Strategy Ensemble (`CMAESNeuroEvolutionAgent`) for Candy Crush Saga.
    Combines a deep residual spatial feature extractor with a CMA-ES / PEPG population of M
    perturbed policy heads and a state-conditioned Darwinian Fitness Router.
    """
    def __init__(
        self,
        in_channels: int = 20,
        d_model: int = 256,
        num_res_blocks: int = 6,
        population_size: int = 8,
        cov_rank: int = 4,
        num_actions: int = 144,
        board_h: int = 9,
        board_w: int = 9
    ):
        super().__init__()
        self.in_channels = in_channels
        self.d_model = d_model
        self.num_res_blocks = num_res_blocks
        self.population_size = population_size
        self.cov_rank = cov_rank
        self.num_actions = num_actions
        self.board_h = board_h
        self.board_w = board_w
        self.num_tokens = board_h * board_w

        # Shared Deep Spatial Representation Backbone
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, d_model, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(d_model),
            nn.SiLU()
        )
        self.res_tower = nn.ModuleList([
            EvolutionaryResBlock(d_model) for _ in range(num_res_blocks)
        ])

        # Spatial compression to policy feature space
        self.spatial_compress = nn.Sequential(
            nn.Conv2d(d_model, 64, kernel_size=1, bias=False),
            nn.BatchNorm2d(64),
            nn.SiLU(),
            nn.Flatten(),
            nn.Linear(64 * self.num_tokens, d_model),
            nn.LayerNorm(d_model),
            nn.SiLU()
        )

        # CMA-ES / PEPG Population Policy Head
        self.cmaes_population = LowRankCMAESPopulationHead(
            feat_dim=d_model,
            num_actions=num_actions,
            population_size=population_size,
            cov_rank=cov_rank
        )

        # State-dependent Darwinian Fitness Router predicting survival weight of each specialist m in 1..M
        self.darwin_fitness_router = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.SiLU(),
            nn.Linear(128, population_size)
        )

        # Value / Win Probability Head conditioned on ensemble consensus
        self.value_head = nn.Sequential(
            nn.Linear(d_model + population_size, 128),
            nn.SiLU(),
            nn.Linear(128, 1)
        )

    def forward(
        self,
        board_tensor: torch.Tensor,
        legal_mask: Optional[torch.Tensor] = None,
        resample_population: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Canonical forward pass accepting (B, 20, 9, 9) board tensor.
        Evaluates all M evolutionary population candidates and aggregates them via Darwinian
        fitness-weighted softmax consensus.
        """
        B, C, H, W = board_tensor.shape

        h = self.stem(board_tensor)
        for block in self.res_tower:
            h = block(h)

        latent_feat = self.spatial_compress(h)  # (B, d_model)

        # Evaluate all M population policy perturbations: (B, M, 144)
        population_logits, active_genomes = self.cmaes_population.evaluate_population(
            latent_feat, resample=resample_population
        )

        # Compute Darwinian fitness weights over the M population members: (B, M)
        fitness_logits = self.darwin_fitness_router(latent_feat)
        fitness_weights = F.softmax(fitness_logits, dim=-1)  # (B, M)

        # Consensus policy logits weighted by predicted survival fitness
        policy_logits = torch.einsum("bm,bma->ba", fitness_weights, population_logits)  # (B, 144)

        if legal_mask is not None:
            policy_logits = policy_logits.masked_fill(~legal_mask.bool(), -1e9)

        # Compute population epistemic disagreement (variance across the M candidates)
        pop_probs = F.softmax(population_logits, dim=-1)  # (B, M, 144)
        population_disagreement = pop_probs.std(dim=1).mean(dim=-1, keepdim=True)  # (B, 1)

        # Value & Win Probability
        val_raw = self.value_head(torch.cat([latent_feat, fitness_weights], dim=-1))
        win_prob = torch.sigmoid(val_raw)
        value = torch.tanh(val_raw)

        return {
            "policy_logits": policy_logits,
            "win_prob": win_prob,
            "value": value,
            "population_logits": population_logits,
            "population_fitness_weights": fitness_weights,
            "population_disagreement": population_disagreement,
            "active_genomes": active_genomes,
            "step_size_sigma": self.cmaes_population.step_size_sigma
        }

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def create_cmaes_evolution_agent(device: str = "cpu") -> CMAESNeuroEvolutionAgent:
    """Factory function initializing CMAESNeuroEvolutionAgent on the target device."""
    model = CMAESNeuroEvolutionAgent(
        in_channels=20,
        d_model=256,
        num_res_blocks=6,
        population_size=8,
        cov_rank=4,
        num_actions=144
    )
    return model.to(device)


if __name__ == "__main__":
    torch.manual_seed(42)
    net = create_cmaes_evolution_agent("cpu")
    params = net.count_parameters()
    print(f"[CMAESNeuroEvolutionAgent] Initialized with {params:,} trainable parameters ({params / 1e6:.2f}M)")

    dummy_board = torch.randn(2, 20, 9, 9)
    out = net(dummy_board, resample_population=True)
    print(f"  policy_logits:              {out['policy_logits'].shape}")
    print(f"  win_prob:                   {out['win_prob'].shape}")
    print(f"  value:                      {out['value'].shape}")
    print(f"  population_logits:          {out['population_logits'].shape}")
    print(f"  population_fitness_weights: {out['population_fitness_weights'].shape}")

    # Test CMA-ES covariance adaptation step
    dummy_returns = torch.tensor([12.5, 8.2, 15.0, 3.1, 9.8, 18.4, 6.0, 11.2])
    cma_stats = net.cmaes_population.update_cmaes(out["active_genomes"], dummy_returns)
    print(f"  CMA-ES Generation Update:   {cma_stats}")
