"""
Reproduction of Espeholt et al. (ICML 2018) IMPALA Deep ResNet + LSTM Architecture:
"IMPALA: Scalable Distributed Deep-RL with Importance Weighted Actor-Learner Architectures" —
Lasse Espeholt, Hubert Soyer, Remi Munos, Karen Simonyan, Volodymyr Mnih, Tom Ward,
Yotam Doron, Vlad Firoiu, Tim Harley, Iain Dunning, Shane Legg, Koray Kavukcuoglu (DeepMind).

Architectural Specifications (faithful to Espeholt et al., ICML 2018, Figure 3):
  1. 3 ConvSequence Stages with channel depths [16, 32, 32]:
     - Each ConvSequence stage applies:
       * 3x3 Conv2d(in_ch, out_ch, stride=1, padding=1)
       * 3x3 MaxPool2d(kernel_size=3, stride=2, padding=1)
       * 2 Pre-Activation Residual Blocks: x + Conv3x3(ReLU(Conv3x3(ReLU(x))))
  2. Flatten + ReLU Linear Bottleneck (256 units)
  3. Recurrent LSTM Core (`nn.LSTMCell(256, 256)`) for episode memory across cascades/moves
  4. Actor Head -> (B, 144) action logits & V-trace Critic Head -> (B, 1) baseline value V(s)
"""

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


class ImpalaResidualBlock(nn.Module):
    """
    Pre-activation Residual Block from IMPALA (Espeholt et al., ICML 2018, Fig. 3):
    out = x + Conv3x3(ReLU(Conv3x3(ReLU(x))))
    """

    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(x)
        out = self.conv1(out)
        out = F.relu(out)
        out = self.conv2(out)
        return out + x


class ImpalaConvSequence(nn.Module):
    """
    Canonical IMPALA ConvSequence Stage:
    Conv3x3 -> MaxPool3x3(stride=2) -> ResidualBlock -> ResidualBlock
    """

    def __init__(self, in_channels: int, out_channels: int, use_maxpool: bool = True):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.use_maxpool = use_maxpool
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.res_block1 = ImpalaResidualBlock(out_channels)
        self.res_block2 = ImpalaResidualBlock(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        if self.use_maxpool:
            x = self.maxpool(x)
        x = self.res_block1(x)
        x = self.res_block2(x)
        return x


class ImpalaResNetBaseline(nn.Module):
    """
    Faithful reproduction of Espeholt et al. (ICML 2018) IMPALA Deep ResNet + LSTM
    Actor-Critic baseline (3 ConvSequence stages [16, 32, 32] + LSTM core + Actor-Critic heads).
    """

    def __init__(
        self,
        in_channels: int = 20,
        conv_channels: Tuple[int, int, int] = (16, 32, 32),
        lstm_hidden_dim: int = 256,
        num_actions: int = NUM_CANONICAL_ACTIONS
    ):
        super().__init__()
        self.in_channels = in_channels
        self.conv_channels = conv_channels
        self.lstm_hidden_dim = lstm_hidden_dim
        self.num_actions = num_actions

        # 3 IMPALA ConvSequence stages [16, 32, 32] with 3x3 max-pooling (9x9 -> 5x5 -> 3x3 -> 2x2)
        c1, c2, c3 = conv_channels
        self.conv_seq1 = ImpalaConvSequence(in_channels, c1, use_maxpool=True)
        self.conv_seq2 = ImpalaConvSequence(c1, c2, use_maxpool=True)
        self.conv_seq3 = ImpalaConvSequence(c2, c3, use_maxpool=True)

        # After 3 stride-2 maxpools on 9x9: spatial size is 2x2 -> flat dim = c3 * 2 * 2
        flat_dim = c3 * 2 * 2
        self.fc_trunk = nn.Linear(flat_dim, lstm_hidden_dim)

        # Recurrent LSTM Core (Espeholt et al., 2018)
        self.lstm_core = nn.LSTMCell(lstm_hidden_dim, lstm_hidden_dim)

        # Policy (Actor) and Value (V-trace Critic) Heads
        self.policy_head = nn.Linear(lstm_hidden_dim, num_actions)
        self.value_head = nn.Linear(lstm_hidden_dim, 1)

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        nn.init.orthogonal_(self.policy_head.weight, gain=0.01)
        nn.init.orthogonal_(self.value_head.weight, gain=1.0)

    def initial_lstm_state(
        self, batch_size: int = 1, device: Optional[torch.device] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        dev = device or next(self.parameters()).device
        h0 = torch.zeros((batch_size, self.lstm_hidden_dim), dtype=torch.float32, device=dev)
        c0 = torch.zeros((batch_size, self.lstm_hidden_dim), dtype=torch.float32, device=dev)
        return h0, c0

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

    def forward_step(
        self,
        state_or_tensor: Union[ReverseEngineState, torch.Tensor, np.ndarray],
        lstm_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        legal_action_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Single-step IMPALA forward pass with recurrent LSTM state:
        Returns (policy_logits: (B, 144), value: (B, 1), next_lstm_state: (hx, cx)).
        """
        x = self._prepare_input(state_or_tensor)
        B = x.shape[0]

        out = self.conv_seq1(x)
        out = self.conv_seq2(out)
        out = self.conv_seq3(out)
        out = F.relu(out)
        out = out.flatten(start_dim=1)
        embed = F.relu(self.fc_trunk(out))

        if lstm_state is None or lstm_state[0].shape[0] != B:
            lstm_state = self.initial_lstm_state(batch_size=B, device=x.device)

        hx, cx = self.lstm_core(embed, lstm_state)
        policy_logits = self.policy_head(hx)
        value = self.value_head(hx)

        if legal_action_mask is not None:
            policy_logits = policy_logits.masked_fill(~legal_action_mask.to(policy_logits.device), -1e9)

        return policy_logits, value, (hx, cx)

    def forward(
        self,
        state_or_tensor: Union[ReverseEngineState, torch.Tensor, np.ndarray],
        legal_action_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Canonical forward pass returning (policy_logits: (B, 144), value: (B, 1)).
        """
        policy_logits, value, _ = self.forward_step(
            state_or_tensor, lstm_state=None, legal_action_mask=legal_action_mask
        )
        return policy_logits, value

    def forward_actor_critic(
        self,
        state_or_tensor: Union[ReverseEngineState, torch.Tensor, np.ndarray],
        legal_action_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.forward(state_or_tensor, legal_action_mask=legal_action_mask)

    @staticmethod
    def compute_vtrace_targets(
        behavior_log_probs: torch.Tensor,
        target_log_probs: torch.Tensor,
        rewards: torch.Tensor,
        values: torch.Tensor,
        bootstrap_value: torch.Tensor,
        gamma: float = 0.99,
        rho_bar: float = 1.0,
        c_bar: float = 1.0
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Computes V-trace target values v_s and policy gradient advantages (Espeholt et al., 2018 Eq. 1).
        Inputs have shape (T, B).
        """
        with torch.no_grad():
            log_rhos = target_log_probs - behavior_log_probs
            rhos = torch.exp(log_rhos)
            clipped_rhos = torch.clamp(rhos, max=rho_bar)
            cs = torch.clamp(rhos, max=c_bar)

            values_t_plus_1 = torch.cat([values[1:], bootstrap_value.unsqueeze(0)], dim=0)
            deltas = clipped_rhos * (rewards + gamma * values_t_plus_1 - values)

            T, B = rewards.shape
            vs_minus_v_xs = torch.zeros_like(values)
            acc = torch.zeros(B, dtype=values.dtype, device=values.device)
            for t in range(T - 1, -1, -1):
                acc = deltas[t] + gamma * cs[t] * acc
                vs_minus_v_xs[t] = acc

            vs = vs_minus_v_xs + values
            vs_t_plus_1 = torch.cat([vs[1:], bootstrap_value.unsqueeze(0)], dim=0)
            pg_advantages = clipped_rhos * (rewards + gamma * vs_t_plus_1 - values)
            return vs, pg_advantages

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class ImpalaVTraceAgent2018:
    """
    Agent wrapper around ImpalaResNetBaseline maintaining recurrent LSTM state across a level episode.
    """

    def __init__(self, engine: Optional[ReverseCandyEngine] = None, device: str = "cpu"):
        self.device = torch.device(device)
        self.engine = engine or ReverseCandyEngine()
        self.net = ImpalaResNetBaseline().to(self.device)
        self.net.eval()
        self.lstm_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None

    def reset_episode_state(self) -> None:
        self.lstm_state = self.net.initial_lstm_state(batch_size=1, device=self.device)

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
            logits, _, self.lstm_state = self.net.forward_step(
                state, lstm_state=self.lstm_state, legal_action_mask=mask
            )
            probs = F.softmax(logits.squeeze(0), dim=-1)

        best_prob = -1.0
        best_move = legal_moves[0]
        for r, c, d in legal_moves:
            idx = encode_action_index(r, c, d)
            p = float(probs[idx].item())
            if p > best_prob:
                best_prob = p
                best_move = (r, c, d)

        return (best_move[0], best_move[1], best_move[2], best_prob)

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

        self.reset_episode_state()
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
            "model_name": "ImpalaResNetBaseline (Espeholt et al., 2018)",
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
    net = ImpalaResNetBaseline()
    dummy = torch.randn(2, 20, 9, 9)
    logits, val = net(dummy)
    print(f"ImpalaResNetBaseline | Params: {net.count_parameters():,} | Logits: {logits.shape} | Val: {val.shape}")
