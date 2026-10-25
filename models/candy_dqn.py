"""
Rainbow Double Dueling Deep Q-Network (DQN) for Candy Crush Saga.
Features:
  1. Dueling Architecture: Decoupled State-Value V(s) and Action-Advantage A(s, a)
  2. Double Q-Learning: Decoupled action selection and target evaluation to prevent overestimation
  3. Action Masking: Restricts Q-learning and argmax strictly to legal canonical swaps [0..143]
  4. Prioritized Experience Replay (PER) with Importance Sampling weights
  5. Multi-Step Return (n-step Bellman targets)
"""

from typing import List, Tuple, Dict, Any, Optional
import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from core.reverse_engine import (
    ReverseEngineState, ReverseCandyEngine, Direction, CandyColor, SpecialCandy, Blocker, Ingredient, LevelType
)
from models.spatial_transformer_net import (
    encode_spatial_board_tensor, encode_action_index, decode_action_index, NUM_CANONICAL_ACTIONS
)


class DuelingQNetwork(nn.Module):
    """
    Dueling Deep Q-Network.
    Computes Q(s, a) = V(s) + (A(s, a) - mean(A(s, a_legal)))
    over all 144 canonical swaps.
    """

    def __init__(self, in_channels: int = 20, hidden_dim: int = 128, num_actions: int = 144):
        super().__init__()
        self.num_actions = num_actions

        # Shared Feature Representation Trunk
        self.trunk = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.Mish(),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.Mish(),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.Mish()
        )

        # 1. State Value Stream: V(s) -> (B, 1)
        self.value_stream = nn.Sequential(
            nn.Conv2d(hidden_dim, 32, kernel_size=1),
            nn.BatchNorm2d(32),
            nn.Mish(),
            nn.Flatten(),
            nn.Linear(32 * 9 * 9, 128),
            nn.Mish(),
            nn.Linear(128, 1)
        )

        # 2. Advantage Stream: A(s, a) -> (B, 144)
        self.advantage_stream = nn.Sequential(
            nn.Conv2d(hidden_dim, 64, kernel_size=1),
            nn.BatchNorm2d(64),
            nn.Mish(),
            nn.Flatten(),
            nn.Linear(64 * 9 * 9, 256),
            nn.Mish(),
            nn.Linear(256, num_actions)
        )

    def forward(self, x: torch.Tensor, legal_action_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass computing Q-values for all actions.
        legal_action_mask: (B, 144) boolean mask where True=legal, False=illegal.
        """
        features = self.trunk(x)
        val = self.value_stream(features)         # (B, 1)
        adv = self.advantage_stream(features)     # (B, 144)

        if legal_action_mask is not None:
            # Mean centering over only legal actions
            mask_float = legal_action_mask.float()
            num_legal = torch.clamp(mask_float.sum(dim=-1, keepdim=True), min=1.0)
            adv_masked = adv * mask_float
            mean_adv = adv_masked.sum(dim=-1, keepdim=True) / num_legal
            q_values = val + (adv - mean_adv)
            # Mask illegal moves with -1e9
            q_values = torch.where(legal_action_mask, q_values, torch.tensor(-1e9, device=x.device))
        else:
            mean_adv = adv.mean(dim=-1, keepdim=True)
            q_values = val + (adv - mean_adv)

        return q_values

    def select_action(
        self,
        state_tensor: torch.Tensor,
        legal_moves: List[Tuple[int, int, Direction]],
        epsilon: float = 0.05
    ) -> Tuple[int, Tuple[int, int, Direction], float]:
        """
        Epsilon-greedy action selection strictly restricted to legal moves.
        Returns: (action_idx, (r, c, direction), q_value)
        """
        if not legal_moves:
            raise ValueError("Cannot select action with empty legal moves!")

        legal_indices = [encode_action_index(r, c, d) for r, c, d in legal_moves]

        # Epsilon exploration
        if random.random() < epsilon:
            chosen_idx = random.choice(legal_indices)
            r, c, d = decode_action_index(chosen_idx)
            return chosen_idx, (r, c, d), 0.0

        # Greedy Q-value exploitation
        self.eval()
        with torch.no_grad():
            mask = torch.zeros((1, self.num_actions), dtype=torch.bool, device=state_tensor.device)
            for idx in legal_indices:
                if 0 <= idx < self.num_actions:
                    mask[0, idx] = True

            q_vals = self.forward(state_tensor, legal_action_mask=mask)
            best_idx = int(q_vals.argmax(dim=-1).item())
            q_max = float(q_vals[0, best_idx].item())

        r, c, d = decode_action_index(best_idx)
        return best_idx, (r, c, d), q_max


class PrioritizedReplayBuffer:
    """
    Proportional Prioritized Experience Replay (PER) buffer.
    Samples transitions with probability proportional to TD error |delta|^alpha.
    """

    def __init__(self, capacity: int = 50000, alpha: float = 0.6, beta: float = 0.4, beta_increment: float = 1e-4):
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment
        self.buffer: List[Dict[str, Any]] = []
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        self.pos = 0
        self.max_priority = 1.0

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
        legal_next_indices: List[int]
    ) -> None:
        """Stores transition in buffer with max current priority."""
        data = {
            "state": state,
            "action": action,
            "reward": reward,
            "next_state": next_state,
            "done": done,
            "legal_next": legal_next_indices
        }

        if len(self.buffer) < self.capacity:
            self.buffer.append(data)
        else:
            self.buffer[self.pos] = data

        self.priorities[self.pos] = self.max_priority
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size: int = 64) -> Tuple[Dict[str, Any], np.ndarray, np.ndarray]:
        """
        Samples a batch with importance sampling weights.
        Returns: (batch_dict, sample_indices, is_weights)
        """
        n = len(self.buffer)
        if n == 0:
            raise ValueError("Buffer is empty!")

        prios = self.priorities[:n]
        probs = prios ** self.alpha
        probs /= probs.sum()

        indices = np.random.choice(n, batch_size, p=probs)
        self.beta = min(1.0, self.beta + self.beta_increment)

        # Importance sampling weights: w_i = (N * P(i))^(-beta) / max(w)
        weights = (n * probs[indices]) ** (-self.beta)
        weights /= weights.max()
        weights = np.array(weights, dtype=np.float32)

        batch_states = np.array([self.buffer[idx]["state"] for idx in indices], dtype=np.float32)
        batch_actions = np.array([self.buffer[idx]["action"] for idx in indices], dtype=np.int64)
        batch_rewards = np.array([self.buffer[idx]["reward"] for idx in indices], dtype=np.float32)
        batch_next = np.array([self.buffer[idx]["next_state"] for idx in indices], dtype=np.float32)
        batch_dones = np.array([self.buffer[idx]["done"] for idx in indices], dtype=np.float32)
        batch_legal_next = [self.buffer[idx]["legal_next"] for idx in indices]

        batch = {
            "states": batch_states,
            "actions": batch_actions,
            "rewards": batch_rewards,
            "next_states": batch_next,
            "dones": batch_dones,
            "legal_next": batch_legal_next
        }

        return batch, indices, weights

    def update_priorities(self, indices: np.ndarray, td_errors: np.ndarray) -> None:
        """Updates transition priorities based on latest absolute TD errors."""
        for idx, td in zip(indices, td_errors):
            p = float(abs(td) + 1e-6)
            self.priorities[idx] = p
            self.max_priority = max(self.max_priority, p)

    def __len__(self) -> int:
        return len(self.buffer)
