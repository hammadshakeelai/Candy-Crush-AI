"""
Double Dueling Deep Q-Learning (DQN) Trainer for Candy Crush Saga.
Features:
  - Double DQN target decoupling
  - Prioritized Experience Replay (PER)
  - Action-masked Q-value maximization
  - Polyak target network stabilization
  - Adaptive epsilon-greedy exploration
"""

from typing import List, Tuple, Dict, Any, Optional
import os
import sys
import copy
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from core.reverse_engine import (
    ReverseEngineState, ReverseCandyEngine, Direction, LevelType, CandyColor, SpecialCandy, Blocker
)
from core.king_level_loader import KingLevelLoader
from models.spatial_transformer_net import (
    encode_spatial_board_tensor, encode_action_index, decode_action_index
)
from models.candy_dqn import DuelingQNetwork, PrioritizedReplayBuffer


class CandyCrushDQNTrainer:
    """Trainer orchestrating Double Dueling Deep Q-Learning."""

    def __init__(
        self,
        online_net: Optional[DuelingQNetwork] = None,
        gamma: float = 0.95,
        lr: float = 5e-4,
        tau: float = 0.005,
        buffer_size: int = 25000,
        batch_size: int = 64,
        device: Optional[torch.device] = None
    ):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size

        self.online_net = (online_net or DuelingQNetwork(in_channels=20, hidden_dim=128)).to(self.device)
        self.target_net = copy.deepcopy(self.online_net).to(self.device)
        self.target_net.eval()

        self.optimizer = torch.optim.AdamW(self.online_net.parameters(), lr=lr, weight_decay=1e-4)
        self.memory = PrioritizedReplayBuffer(capacity=buffer_size)

    def update_network(self) -> float:
        """Executes a single Double DQN optimization step with PER weights."""
        if len(self.memory) < self.batch_size:
            return 0.0

        batch, indices, is_weights = self.memory.sample(self.batch_size)

        states_t = torch.from_numpy(batch["states"]).to(self.device)
        actions_t = torch.from_numpy(batch["actions"]).unsqueeze(1).to(self.device)
        rewards_t = torch.from_numpy(batch["rewards"]).unsqueeze(1).to(self.device)
        next_states_t = torch.from_numpy(batch["next_states"]).to(self.device)
        dones_t = torch.from_numpy(batch["dones"]).unsqueeze(1).to(self.device)
        weights_t = torch.from_numpy(is_weights).unsqueeze(1).to(self.device)

        # 1. Current Q(s, a) predicted by Online Net
        q_eval = self.online_net(states_t).gather(1, actions_t)

        # 2. Double DQN target computation:
        # a* = argmax_{a' in legal} Q_online(s', a')
        with torch.no_grad():
            next_q_online = self.online_net(next_states_t)  # (B, 144)

            # Mask illegal moves in next state
            next_masks = torch.zeros_like(next_q_online, dtype=torch.bool)
            for b_idx, legal_indices in enumerate(batch["legal_next"]):
                for a_idx in legal_indices:
                    if 0 <= a_idx < 144:
                        next_masks[b_idx, a_idx] = True

            next_q_online_masked = torch.where(next_masks, next_q_online, torch.tensor(-1e9, device=self.device))
            next_best_actions = next_q_online_masked.argmax(dim=-1, keepdim=True)

            # Evaluate with Target Net: Q_target(s', a*)
            next_q_target = self.target_net(next_states_t).gather(1, next_best_actions)
            # Bellman target
            q_target = rewards_t + (1.0 - dones_t) * self.gamma * next_q_target

        td_errors = (q_eval - q_target).detach().squeeze(-1).cpu().numpy()
        self.memory.update_priorities(indices, td_errors)

        # Weighted Smooth L1 (Huber) loss
        loss = (weights_t * F.smooth_l1_loss(q_eval, q_target, reduction="none")).mean()

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.online_net.parameters(), max_norm=1.0)
        self.optimizer.step()

        # Polyak Target Update
        for target_param, online_param in zip(self.target_net.parameters(), self.online_net.parameters()):
            target_param.data.copy_(self.tau * online_param.data + (1.0 - self.tau) * target_param.data)

        return float(loss.item())

    def train_on_levels(
        self,
        episodes: int = 50,
        max_moves_per_ep: int = 25,
        eps_start: float = 0.5,
        eps_end: float = 0.05,
        checkpoint_out: str = "checkpoints/candy_crush_dueling_dqn.pt"
    ) -> DuelingQNetwork:
        """Runs the complete training loop across King levels."""
        loader = KingLevelLoader()
        engine = ReverseCandyEngine()

        print("=================================================================")
        print("  DOUBLE DUELING DEEP Q-LEARNING (DQN) TRAINING PIPELINE         ")
        print(f"  Device: {self.device} | Total Episodes: {episodes}             ")
        print("=================================================================")

        test_levels = [1, 5, 10, 15, 20, 25, 30, 50, 100]
        total_wins = 0

        for ep in range(1, episodes + 1):
            lvl_id = test_levels[(ep - 1) % len(test_levels)]
            try:
                cfg = loader.load_level(lvl_id)
            except Exception:
                continue

            state = ReverseEngineState(cfg)
            engine.initialize_board(state, seed=42 + ep * 73)

            epsilon = max(eps_end, eps_start - (ep / float(episodes)) * (eps_start - eps_end))
            step = 0
            ep_reward = 0.0
            ep_loss = 0.0
            loss_steps = 0

            while not state.has_won() and not state.has_lost() and step < max_moves_per_ep:
                legal_moves = engine.get_legal_moves(state)
                if not legal_moves:
                    engine.initialize_board(state)
                    legal_moves = engine.get_legal_moves(state)
                    if not legal_moves:
                        break

                s_np = encode_spatial_board_tensor(state)
                s_t = torch.from_numpy(s_np).unsqueeze(0).to(self.device)

                a_idx, (r, c, direction), q_est = self.online_net.select_action(
                    s_t, legal_moves, epsilon=epsilon
                )

                prev_obj = state.remaining_objectives_count()
                score_gained, stats = engine.execute_move(state, r, c, direction)
                new_obj = state.remaining_objectives_count()

                obj_cleared = max(0, prev_obj - new_obj)
                cascades = stats.get("cascades", 0)

                # Shaped Q-learning reward
                reward = (obj_cleared * 10.0) + (cascades * 2.0) + (score_gained / 1000.0)

                done = state.has_won() or state.has_lost() or (step + 1 >= max_moves_per_ep)
                if state.has_won():
                    reward += 100.0
                elif state.has_lost():
                    reward -= 20.0

                s_next_np = encode_spatial_board_tensor(state)
                next_legal = engine.get_legal_moves(state)
                next_legal_indices = [encode_action_index(nr, nc, nd) for nr, nc, nd in next_legal]

                self.memory.push(s_np, a_idx, reward, s_next_np, done, next_legal_indices)

                # Update DQN parameters
                loss_val = self.update_network()
                if loss_val > 0.0:
                    ep_loss += loss_val
                    loss_steps += 1

                ep_reward += reward
                step += 1

            if state.has_won():
                total_wins += 1

            if ep % max(1, episodes // 5) == 0 or ep == episodes:
                avg_l = ep_loss / max(1, loss_steps)
                win_pct = (total_wins / float(ep)) * 100.0
                print(f"DQN Episode {ep:3d}/{episodes:3d} | Level: {lvl_id:3d} | Win Rate: {win_pct:.1f}% | Ep Reward: {ep_reward:6.1f} | Loss: {avg_l:.4f} | Eps: {epsilon:.2f}")

        os.makedirs(os.path.dirname(os.path.abspath(checkpoint_out)), exist_ok=True)
        torch.save(self.online_net.state_dict(), checkpoint_out)
        print(f"\n[DQN Training Complete] Checkpoint saved: {checkpoint_out}")
        return self.online_net


if __name__ == "__main__":
    trainer = CandyCrushDQNTrainer()
    trainer.train_on_levels(episodes=15, max_moves_per_ep=20)
