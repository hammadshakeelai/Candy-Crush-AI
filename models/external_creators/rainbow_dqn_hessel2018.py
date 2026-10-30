"""
Reproduction of Hessel et al. (AAAI 2018) Rainbow DQN Architecture:
"Rainbow: Combining Improvements in Deep Reinforcement Learning" —
Matteo Hessel, Joseph Modayil, Hado van Hasselt, Tom Schaul, Georg Ostrovski,
Will Dabney, Dan Horgan, Bilal Piot, Mohammad Azar, David Silver (DeepMind).

Architectural Specifications (faithful to Hessel et al., AAAI 2018):
  1. Factorized Gaussian Noisy Linear Layers (`NoisyLinear`, Fortunato et al., ICLR 2018):
     - Replaces standard Linear layers in both Dueling streams with parametric noise:
       W = mu_w + sigma_w * (eps_out (x) eps_in),  b = mu_b + sigma_b * eps_out
       where f(x) = sgn(x) * sqrt(|x|), sigma_0 = 0.5.
  2. Dueling Distributional Network Architecture (Wang et al., 2016 + Bellemare et al., 2017):
     - Value Stream V_i(s): outputs (B, 1, N_atoms) atom logits
     - Advantage Stream A_i(s, a): outputs (B, 144, N_atoms) atom logits
     - Combined Atom Logits (Hessel et al., Eq. 9):
       L_i(s, a) = V_i(s) + A_i(s, a) - (1 / |A|) sum_{a'} A_i(s, a')
  3. Categorical 51-Atom Return Distribution (C51, V_min = -10.0, V_max = 10.0, N_atoms = 51):
     - Support z_i = linspace(-10.0, 10.0, 51)
     - Action-value expectation: Q(s, a) = sum_{i=1}^{51} z_i * softmax_i(L(s, a)) -> (B, 144)
     - State-value expectation:  V(s)    = sum_{i=1}^{51} z_i * softmax_i(V(s))    -> (B, 1)
"""

import math
from typing import List, Tuple, Dict, Optional, Any, Union
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from core.reverse_engine import (
    ReverseEngineState, ReverseCandyEngine, Direction, LevelConfig
)
from models.spatial_transformer_net import (
    encode_spatial_board_tensor, encode_action_index, decode_action_index, NUM_CANONICAL_ACTIONS
)


class NoisyLinear(nn.Module):
    """
    Factorized Gaussian NoisyNet Layer (Fortunato et al., ICLR 2018; Hessel et al., AAAI 2018).
    Used for state-dependent parametric exploration in Rainbow DQN.
    """

    def __init__(self, in_features: int, out_features: int, sigma_init: float = 0.5):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.sigma_init = sigma_init

        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.register_buffer("weight_epsilon", torch.empty(out_features, in_features))

        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))
        self.register_buffer("bias_epsilon", torch.empty(out_features))

        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self) -> None:
        mu_range = 1.0 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.sigma_init / math.sqrt(self.in_features))
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.sigma_init / math.sqrt(self.out_features))

    @staticmethod
    def _scale_noise(size: int, device: torch.device) -> torch.Tensor:
        x = torch.randn(size, device=device)
        return x.sign().mul_(x.abs().sqrt_())

    def reset_noise(self) -> None:
        device = self.weight_mu.device
        eps_in = self._scale_noise(self.in_features, device)
        eps_out = self._scale_noise(self.out_features, device)
        self.weight_epsilon.copy_(eps_out.ger(eps_in))
        self.bias_epsilon.copy_(eps_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
            bias = self.bias_mu + self.bias_sigma * self.bias_epsilon
        else:
            weight = self.weight_mu
            bias = self.bias_mu
        return F.linear(x, weight, bias)


class RainbowDQN2018Net(nn.Module):
    """
    Faithful reproduction of Hessel et al. (AAAI 2018) Rainbow DQN adapted for 9x9 Candy Crush:
      - 3-Layer Convolutional Trunk + Residual Refinement
      - Factorized Gaussian NoisyLinear Layers
      - Dueling Value & Advantage Streams over 51 Categorical Atoms (C51, V_min=-10, V_max=10)
    """

    def __init__(
        self,
        in_channels: int = 20,
        hidden_channels: int = 128,
        hidden_fc: int = 512,
        num_actions: int = NUM_CANONICAL_ACTIONS,
        num_atoms: int = 51,
        v_min: float = -10.0,
        v_max: float = 10.0,
        sigma_init: float = 0.5
    ):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.hidden_fc = hidden_fc
        self.num_actions = num_actions
        self.num_atoms = num_atoms
        self.v_min = v_min
        self.v_max = v_max

        # Register fixed C51 categorical support z in [-10, 10]
        support = torch.linspace(v_min, v_max, num_atoms, dtype=torch.float32)
        self.register_buffer("support", support)
        self.delta_z = (v_max - v_min) / float(num_atoms - 1)

        # Convolutional Spatial Feature Trunk
        self.conv_trunk = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Flatten()
        )
        flat_dim = 64 * 9 * 9

        # Dueling Distributional Value Stream: V(s) -> (B, 1, 51)
        self.value_noisy_fc1 = NoisyLinear(flat_dim, hidden_fc, sigma_init=sigma_init)
        self.value_noisy_fc2 = NoisyLinear(hidden_fc, num_atoms, sigma_init=sigma_init)

        # Dueling Distributional Advantage Stream: A(s, a) -> (B, 144, 51)
        self.advantage_noisy_fc1 = NoisyLinear(flat_dim, hidden_fc, sigma_init=sigma_init)
        self.advantage_noisy_fc2 = NoisyLinear(hidden_fc, num_actions * num_atoms, sigma_init=sigma_init)

    def reset_noise(self) -> None:
        """Resamples factorized Gaussian exploration noise across all NoisyLinear layers."""
        self.value_noisy_fc1.reset_noise()
        self.value_noisy_fc2.reset_noise()
        self.advantage_noisy_fc1.reset_noise()
        self.advantage_noisy_fc2.reset_noise()

    def _prepare_input(
        self, state_or_tensor: Union[ReverseEngineState, torch.Tensor, np.ndarray]
    ) -> torch.Tensor:
        if isinstance(state_or_tensor, ReverseEngineState):
            arr = encode_spatial_board_tensor(state_or_tensor)
            x = torch.from_numpy(arr).unsqueeze(0)
        elif isinstance(state_or_tensor, np.ndarray):
            x = torch.from_numpy(state_or_tensor)
            if x.dim() == 3:
                x = x.unsqueeze(0)
        else:
            x = state_or_tensor
            if x.dim() == 3:
                x = x.unsqueeze(0)

        device = next(self.parameters()).device
        return x.to(device=device, dtype=torch.float32)

    def forward_distributional(
        self,
        state_or_tensor: Union[ReverseEngineState, torch.Tensor, np.ndarray],
        legal_action_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Computes:
          - q_values: (B, 144) expected action values sum_i z_i p_i(s, a)
          - state_value: (B, 1) expected state value sum_i z_i p_i^V(s)
          - atom_probs: (B, 144, 51) categorical probability distribution over [-10, 10]
        """
        x = self._prepare_input(state_or_tensor)
        B = x.shape[0]
        feat = self.conv_trunk(x)

        # Value stream: (B, 1, 51)
        v_hid = F.relu(self.value_noisy_fc1(feat), inplace=True)
        v_logits = self.value_noisy_fc2(v_hid).view(B, 1, self.num_atoms)

        # Advantage stream: (B, 144, 51)
        a_hid = F.relu(self.advantage_noisy_fc1(feat), inplace=True)
        a_logits = self.advantage_noisy_fc2(a_hid).view(B, self.num_actions, self.num_atoms)

        # Dueling Distributional combination (Hessel et al., AAAI 2018 Eq. 9)
        q_atom_logits = v_logits + a_logits - a_logits.mean(dim=1, keepdim=True)
        atom_probs = F.softmax(q_atom_logits, dim=-1)  # (B, 144, 51)

        # Expected Q(s, a) = sum_i z_i * p_i(s, a)
        q_values = torch.sum(atom_probs * self.support.view(1, 1, -1), dim=-1)  # (B, 144)

        if legal_action_mask is not None:
            q_values = q_values.masked_fill(~legal_action_mask.to(q_values.device), -1e9)

        # Expected V(s) from value stream distribution
        v_probs = F.softmax(v_logits.squeeze(1), dim=-1)  # (B, 51)
        state_value = torch.sum(v_probs * self.support.view(1, -1), dim=-1, keepdim=True)  # (B, 1)

        return q_values, state_value, atom_probs

    def forward(
        self,
        state_or_tensor: Union[ReverseEngineState, torch.Tensor, np.ndarray],
        legal_action_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Canonical forward pass returning (q_action_logits: (B, 144), state_value: (B, 1)).
        """
        q_values, state_value, _ = self.forward_distributional(
            state_or_tensor, legal_action_mask=legal_action_mask
        )
        return q_values, state_value

    def forward_actor_critic(
        self,
        state_or_tensor: Union[ReverseEngineState, torch.Tensor, np.ndarray],
        legal_action_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.forward(state_or_tensor, legal_action_mask=legal_action_mask)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class RainbowDQN2018Agent:
    """
    Agent wrapper around RainbowDQN2018Net for benchmarking and level evaluation.
    """

    def __init__(self, engine: Optional[ReverseCandyEngine] = None, device: str = "cpu"):
        self.device = torch.device(device)
        self.engine = engine or ReverseCandyEngine()
        self.net = RainbowDQN2018Net().to(self.device)
        self.net.eval()

    def select_best_move(self, state: ReverseEngineState) -> Optional[Tuple[int, int, Direction, float]]:
        legal_moves = self.engine.get_legal_moves(state)
        if not legal_moves:
            return None
        if len(legal_moves) == 1:
            r, c, d = legal_moves[0]
            return (r, c, d, 1.0)

        mask = torch.zeros((1, NUM_CANONICAL_ACTIONS), dtype=torch.bool, device=self.device)
        for r, c, d in legal_moves:
            mask[0, encode_action_index(r, c, d)] = True

        with torch.no_grad():
            q_values, _ = self.net(state, legal_action_mask=mask)
            q_vec = q_values.squeeze(0)

        best_q = -1e18
        best_move = legal_moves[0]
        for r, c, d in legal_moves:
            idx = encode_action_index(r, c, d)
            q_val = float(q_vec[idx].item())
            if q_val > best_q:
                best_q = q_val
                best_move = (r, c, d)

        return (best_move[0], best_move[1], best_move[2], best_q)

    def solve_level(
        self,
        config_or_state: Union[LevelConfig, ReverseEngineState],
        max_moves: Optional[int] = None,
        seed: int = 42
    ) -> Dict[str, Any]:
        if isinstance(config_or_state, ReverseEngineState):
            state = config_or_state.copy()
        else:
            state = ReverseEngineState(config_or_state)
            self.engine.initialize_board(state, seed=seed)

        t0 = time.time()
        moves_limit = max_moves or state.moves_left
        step = 0
        shuffles = 0

        while not state.has_won() and not state.has_lost() and step < moves_limit:
            best = self.select_best_move(state)
            if best is None:
                shuffles += 1
                if shuffles > 3:
                    break
                self.engine.initialize_board(state)
                continue
            shuffles = 0
            r, c, d, _ = best
            self.engine.execute_move(state, r, c, d)
            step += 1

        return {
            "model_name": "RainbowDQN2018Net (Hessel et al., 2018)",
            "level_id": state.config.level_id,
            "won": state.has_won(),
            "moves_used": step,
            "moves_remaining": state.moves_left,
            "final_score": state.score,
            "objectives_remaining": state.remaining_objectives_count(),
            "duration_sec": time.time() - t0,
            "parameters": self.net.count_parameters()
        }


if __name__ == "__main__":
    net = RainbowDQN2018Net()
    dummy = torch.randn(2, 20, 9, 9)
    q_vals, v_val, dist = net.forward_distributional(dummy)
    print(
        f"RainbowDQN2018Net | Params: {net.count_parameters():,} | "
        f"Q: {q_vals.shape} | V: {v_val.shape} | Dist: {dist.shape}"
    )
