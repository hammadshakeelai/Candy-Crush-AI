"""
Reproduction of Schrittwieser et al. (Nature 2020) Vanilla MuZero Architecture:
"Mastering Atari, Go, chess and shogi by planning with a learned model" —
Julian Schrittwieser, Ioannis Antonoglou, Thomas Hubert, Karen Simonyan, Laurent Sifre,
Simon Schmitt, Arthur Guez, Edward Lockhart, Demis Hassabis, Thore Graepel,
Timothy Lillicrap, David Silver (DeepMind).

Architectural Specifications (faithful to Schrittwieser et al., Nature 2020, Appendix F/G):
  1. Representation Function h_theta(o_t) -> s^0:
     - 3x3 Conv stem + Residual Tower mapping (B, 20, 9, 9) board observation to latent spatial state
       s^0 of shape (B, C, 9, 9), scaled to [0, 1] via MuZero per-state min-max normalization.
  2. Spatial Action Plane Encoding:
     - Encodes discrete action a in {0..143} into a 4-channel 9x9 spatial plane tensor:
       * Channel 0: One-hot swap source coordinate (r, c)
       * Channel 1: One-hot swap destination coordinate (nr, nc)
       * Channel 2: Horizontal swap indicator plane
       * Channel 3: Vertical swap indicator plane
  3. Dynamics Function g_theta(s^{k-1}, a^k) -> (r^k, s^k):
     - Concatenates s^{k-1} (C channels) with spatial action planes (4 channels) -> 3x3 Conv +
       Residual Tower -> min-max normalized next latent state s^k (B, C, 9, 9) and scalar reward r^k (B, 1).
  4. Prediction Function f_theta(s^k) -> (p^k, v^k):
     - 1x1 Conv Policy Head -> (B, 144) action logits
     - 1x1 Conv Scalar Value Head -> (B, 1) state value estimate
"""

from typing import List, Tuple, Dict, Optional, Any, Union
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from core.reverse_engine import (
    ReverseEngineState, ReverseCandyEngine, Direction, LevelConfig, OFFSETS
)
from models.spatial_transformer_net import (
    encode_spatial_board_tensor, encode_action_index, decode_action_index, NUM_CANONICAL_ACTIONS
)


def _muzero_minmax_normalize(s: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """
    MuZero latent state normalization (Schrittwieser et al., Nature 2020, Appendix G):
    Scales each hidden state tensor independently into [0, 1] across spatial & channel axes.
    """
    B = s.shape[0]
    flat = s.view(B, -1)
    s_min = flat.min(dim=1, keepdim=True)[0].view(B, 1, 1, 1)
    s_max = flat.max(dim=1, keepdim=True)[0].view(B, 1, 1, 1)
    return (s - s_min) / (s_max - s_min + eps)


class MuZeroResBlock(nn.Module):
    """Standard 3x3 Residual Block used in MuZero Representation and Dynamics networks."""

    def __init__(self, channels: int = 128):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        return F.relu(out + residual, inplace=True)


class MuZeroRepresentationNet(nn.Module):
    """h_theta(o_t) -> s^0: Encodes (B, 20, 9, 9) board tensor into latent state (B, C, 9, 9)."""

    def __init__(self, in_channels: int = 20, latent_channels: int = 128, num_blocks: int = 6):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, latent_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(latent_channels),
            nn.ReLU(inplace=True)
        )
        self.res_tower = nn.Sequential(
            *[MuZeroResBlock(latent_channels) for _ in range(num_blocks)]
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        s0 = self.res_tower(self.stem(obs))
        return _muzero_minmax_normalize(s0)


class MuZeroDynamicsNet(nn.Module):
    """
    g_theta(s^{k-1}, a^k) -> (s^k, r^k):
    Predicts next latent state s^k (B, C, 9, 9) and immediate scalar reward r^k (B, 1)
    given previous latent state and 4-channel one-hot spatial action plane encoding.
    """

    def __init__(self, latent_channels: int = 128, action_planes: int = 4, num_blocks: int = 4):
        super().__init__()
        self.conv_transition = nn.Sequential(
            nn.Conv2d(latent_channels + action_planes, latent_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(latent_channels),
            nn.ReLU(inplace=True)
        )
        self.res_tower = nn.Sequential(
            *[MuZeroResBlock(latent_channels) for _ in range(num_blocks)]
        )
        self.reward_head = nn.Sequential(
            nn.Conv2d(latent_channels, 2, kernel_size=1, bias=False),
            nn.BatchNorm2d(2),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(2 * 9 * 9, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 1)
        )

    def forward(self, s_prev: torch.Tensor, action_plane_tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = torch.cat([s_prev, action_plane_tensor], dim=1)
        s_next_raw = self.res_tower(self.conv_transition(x))
        s_next = _muzero_minmax_normalize(s_next_raw)
        reward = self.reward_head(s_next_raw)
        return s_next, reward


class MuZeroPredictionNet(nn.Module):
    """
    f_theta(s^k) -> (p^k, v^k):
    Predicts policy logits (B, 144) and scalar value (B, 1) from latent state s^k.
    """

    def __init__(self, latent_channels: int = 128, num_actions: int = NUM_CANONICAL_ACTIONS):
        super().__init__()
        self.policy_conv = nn.Sequential(
            nn.Conv2d(latent_channels, 2, kernel_size=1, bias=False),
            nn.BatchNorm2d(2),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(2 * 9 * 9, num_actions)
        )
        self.value_head = nn.Sequential(
            nn.Conv2d(latent_channels, 1, kernel_size=1, bias=False),
            nn.BatchNorm2d(1),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(1 * 9 * 9, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 1),
            nn.Tanh()
        )

    def forward(self, s: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        policy_logits = self.policy_conv(s)
        value = self.value_head(s)
        return policy_logits, value


class MuZeroVanilla2020Net(nn.Module):
    """
    Complete Vanilla MuZero Network (Schrittwieser et al., Nature 2020) for 9x9 Candy Crush:
      - Representation Network h(o_t) -> s^0
      - Spatial One-Hot Action Plane Encoder (a -> 4x9x9 planes)
      - Dynamics Network g(s^{k-1}, a^k) -> (s^k, r^k)
      - Prediction Network f(s^k) -> (policy_logits, value)
    """

    def __init__(
        self,
        in_channels: int = 20,
        latent_channels: int = 128,
        repr_blocks: int = 6,
        dyn_blocks: int = 4,
        num_actions: int = NUM_CANONICAL_ACTIONS
    ):
        super().__init__()
        self.in_channels = in_channels
        self.latent_channels = latent_channels
        self.num_actions = num_actions

        self.representation = MuZeroRepresentationNet(in_channels, latent_channels, repr_blocks)
        self.dynamics = MuZeroDynamicsNet(latent_channels, action_planes=4, num_blocks=dyn_blocks)
        self.prediction = MuZeroPredictionNet(latent_channels, num_actions)

        # Precompute lookup table of (144, 4, 9, 9) one-hot spatial action planes
        action_planes = torch.zeros((num_actions, 4, 9, 9), dtype=torch.float32)
        for a_idx in range(num_actions):
            r, c, d = decode_action_index(a_idx)
            dr, dc = OFFSETS[d]
            nr, nc = min(8, max(0, r + dr)), min(8, max(0, c + dc))
            action_planes[a_idx, 0, r, c] = 1.0
            action_planes[a_idx, 1, nr, nc] = 1.0
            if d == Direction.RIGHT:
                action_planes[a_idx, 2, r, c] = 1.0
                action_planes[a_idx, 2, nr, nc] = 1.0
            else:
                action_planes[a_idx, 3, r, c] = 1.0
                action_planes[a_idx, 3, nr, nc] = 1.0
        self.register_buffer("action_plane_table", action_planes)

    def encode_action_planes(self, actions: Union[torch.Tensor, List[int], int]) -> torch.Tensor:
        """Converts action indices (B,) into spatial one-hot planes (B, 4, 9, 9)."""
        device = self.action_plane_table.device
        if isinstance(actions, int):
            idx = torch.tensor([actions], dtype=torch.long, device=device)
        elif isinstance(actions, list):
            idx = torch.tensor(actions, dtype=torch.long, device=device)
        else:
            idx = actions.to(device=device, dtype=torch.long).view(-1)
        return self.action_plane_table[idx]

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

    def initial_inference(
        self,
        state_or_tensor: Union[ReverseEngineState, torch.Tensor, np.ndarray],
        legal_action_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Executes h(o_t) and f(s^0), returning (s_0, policy_logits, value).
        """
        x = self._prepare_input(state_or_tensor)
        s0 = self.representation(x)
        policy_logits, value = self.prediction(s0)
        if legal_action_mask is not None:
            policy_logits = policy_logits.masked_fill(~legal_action_mask.to(policy_logits.device), -1e9)
        return s0, policy_logits, value

    def recurrent_inference(
        self,
        latent_state: torch.Tensor,
        actions: Union[torch.Tensor, List[int], int]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Executes g(s^{k-1}, a^k) and f(s^k), returning (s_next, reward, policy_logits, value).
        """
        action_planes = self.encode_action_planes(actions)
        s_next, reward = self.dynamics(latent_state, action_planes)
        policy_logits, value = self.prediction(s_next)
        return s_next, reward, policy_logits, value

    def forward(
        self,
        state_or_tensor: Union[ReverseEngineState, torch.Tensor, np.ndarray],
        legal_action_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Canonical forward pass returning (policy_logits: (B, 144), value: (B, 1)).
        """
        _, policy_logits, value = self.initial_inference(state_or_tensor, legal_action_mask=legal_action_mask)
        return policy_logits, value

    def forward_actor_critic(
        self,
        state_or_tensor: Union[ReverseEngineState, torch.Tensor, np.ndarray],
        legal_action_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.forward(state_or_tensor, legal_action_mask=legal_action_mask)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class MuZeroVanilla2020Agent:
    """
    Agent wrapper around MuZeroVanilla2020Net performing 1-step latent dynamics lookahead
    in MuZero's learned latent space.
    """

    def __init__(self, engine: Optional[ReverseCandyEngine] = None, device: str = "cpu"):
        self.device = torch.device(device)
        self.engine = engine or ReverseCandyEngine()
        self.net = MuZeroVanilla2020Net().to(self.device)
        self.net.eval()

    def select_best_move(self, state: ReverseEngineState) -> Optional[Tuple[int, int, Direction, float]]:
        legal_moves = self.engine.get_legal_moves(state)
        if not legal_moves:
            return None
        if len(legal_moves) == 1:
            r, c, d = legal_moves[0]
            return (r, c, d, 1.0)

        mask = torch.zeros((1, NUM_CANONICAL_ACTIONS), dtype=torch.bool, device=self.device)
        legal_indices = []
        for r, c, d in legal_moves:
            idx = encode_action_index(r, c, d)
            mask[0, idx] = True
            legal_indices.append(idx)

        with torch.no_grad():
            s0, logits, _ = self.net.initial_inference(state, legal_action_mask=mask)
            priors = F.softmax(logits.squeeze(0), dim=-1)

            # Evaluate top-K candidate actions via 1-step latent dynamics g(s0, a)
            k = min(6, len(legal_indices))
            top_indices = torch.topk(priors, k=k).indices
            s0_batch = s0.expand(k, -1, -1, -1)
            _, r_pred, _, v_next = self.net.recurrent_inference(s0_batch, top_indices)
            q_latent = (r_pred.squeeze(-1) + 0.99 * v_next.squeeze(-1))

            best_k = int(torch.argmax(priors[top_indices] + 0.25 * torch.sigmoid(q_latent)).item())
            best_action_idx = int(top_indices[best_k].item())
            best_score = float(priors[best_action_idx].item())

        r, c, d = decode_action_index(best_action_idx)
        return (r, c, d, best_score)

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
            "model_name": "MuZeroVanilla2020Net (Schrittwieser et al., 2020)",
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
    net = MuZeroVanilla2020Net()
    dummy = torch.randn(2, 20, 9, 9)
    s0, logits, val = net.initial_inference(dummy)
    s1, rew, logits1, val1 = net.recurrent_inference(s0, torch.tensor([0, 72]))
    print(
        f"MuZeroVanilla2020Net | Params: {net.count_parameters():,} | "
        f"s0: {s0.shape} | Logits: {logits.shape} | Val: {val.shape} | Rew: {rew.shape}"
    )
