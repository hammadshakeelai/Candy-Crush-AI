"""
Group Relative Policy Optimization (GRPO) for Candy Crush Saga.
Synthesizes modern RL reasoning algorithms (DeepSeek GRPO) with Match-3 action masking.

For each update step:
  1. Sample G complete episodes from the same level seed using the current policy.
  2. Compute total reward per episode.
  3. Group-relative advantage: A_i = (R_i - mean_R) / (std_R + eps)
     Eliminates random board seed variance without requiring a complex critic network!
  4. PPO-clip policy loss + action masking + KL divergence penalty vs reference policy.
  5. Save high-performing Grandmaster checkpoint.
"""

import os
import sys
import copy
import time
import argparse
from typing import List, Tuple, Dict, Any, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from core.reverse_engine import (
    ReverseEngineState, ReverseCandyEngine, Direction, LevelType, OFFSETS
)
from core.king_level_loader import KingLevelLoader
from models.network import CandyCrushPolicyValueNet
from train_grandmaster_net import _encode_state_tensor


def run_grpo_episode(
    model: CandyCrushPolicyValueNet,
    engine: ReverseCandyEngine,
    state: ReverseEngineState,
    device: torch.device,
    temperature: float = 1.0,
    max_moves: int = 30
) -> Tuple[List[torch.Tensor], List[int], List[float], float]:
    """
    Executes one rollout.
    Returns: (states_tensors, actions, action_log_probs, total_reward)
    """
    states_list = []
    actions_list = []
    log_probs_list = []
    total_reward = 0.0

    step_count = 0
    moves_limit = min(max_moves, state.moves_left)

    while not state.has_won() and not state.has_lost() and step_count < moves_limit:
        legal_moves = engine.get_legal_moves(state)
        if not legal_moves:
            engine.initialize_board(state)
            legal_moves = engine.get_legal_moves(state)
            if not legal_moves:
                break

        # Encode state to tensor
        obs_np = _encode_state_tensor(state)
        obs_t = torch.from_numpy(obs_np).unsqueeze(0).to(device)

        # Forward pass
        with torch.no_grad():
            policy_logits, _ = model(obs_t)
            policy_logits = policy_logits[0]  # (162,)

            # Create action mask
            action_mask = torch.full((162,), float("-inf"), device=device)
            valid_action_indices = []
            for r, c, direction in legal_moves:
                dir_idx = 0 if direction == Direction.RIGHT else 1
                action_idx = (r * state.width + c) * 2 + dir_idx
                action_mask[action_idx] = 0.0
                valid_action_indices.append(action_idx)

            masked_logits = (policy_logits + action_mask) / max(1e-4, temperature)
            probs = F.softmax(masked_logits, dim=-1)
            dist = torch.distributions.Categorical(probs)
            action = dist.sample()
            action_idx = action.item()
            log_prob = dist.log_prob(action).item()

        # Decode action
        dir_idx = action_idx % 2
        cell_idx = action_idx // 2
        r = cell_idx // state.width
        c = cell_idx % state.width
        direction = Direction.RIGHT if dir_idx == 0 else Direction.DOWN

        # Execute in simulator
        score_before = state.score
        jellies_before = np.sum(state.jellies > 0)
        blockers_before = np.sum(state.blockers > 0)

        score_gained, stats = engine.execute_move(state, r, c, direction)
        step_count += 1

        jellies_cleared = jellies_before - np.sum(state.jellies > 0)
        blockers_cleared = blockers_before - np.sum(state.blockers > 0)

        # Step reward
        step_r = (score_gained / 500.0) + (jellies_cleared * 15.0) + (blockers_cleared * 10.0)
        if state.has_won():
            step_r += 100.0

        total_reward += step_r

        states_list.append(obs_t.squeeze(0))
        actions_list.append(action_idx)
        log_probs_list.append(log_prob)

    return states_list, actions_list, log_probs_list, total_reward


def train_grpo(
    episodes: int = 50,
    group_size: int = 4,
    lr: float = 3e-4,
    eps_clip: float = 0.2,
    beta_kl: float = 0.02,
    checkpoint_out: str = "checkpoints/candy_crush_grpo_v1.pt",
    initial_checkpoint: Optional[str] = None
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== LAUNCHING GRPO TRAINING PIPELINE ===")
    print(f"Device: {device} | Group Size (G): {group_size} | Iterations: {episodes}")

    loader = KingLevelLoader()
    engine = ReverseCandyEngine()

    # Load baseline model if available, else new
    model = CandyCrushPolicyValueNet().to(device)
    base_ckpt = initial_checkpoint or "checkpoints/candy_crush_grandmaster_v2.pt"
    if base_ckpt and os.path.exists(base_ckpt):
        print(f"Loading existing baseline weights from: {base_ckpt}")
        model.load_state_dict(torch.load(base_ckpt, map_location=device, weights_only=True))

    ref_model = copy.deepcopy(model).to(device)
    ref_model.eval()

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    os.makedirs(os.path.dirname(checkpoint_out) or ".", exist_ok=True)
    t0 = time.time()

    for it in range(1, episodes + 1):
        # Pick random King level
        lvl_num = np.random.randint(1, min(500, loader.get_total_levels() + 1))
        cfg = loader.load_level(lvl_num)
        level_seed = np.random.randint(1, 100000)

        group_states = []
        group_actions = []
        group_old_log_probs = []
        group_rewards = []

        # 1. Sample G episodes from the exact same starting seed
        for g in range(group_size):
            engine_seeded = ReverseCandyEngine(seed=level_seed)
            state = ReverseEngineState(cfg)
            engine_seeded.initialize_board(state)

            states, actions, log_probs, r_tot = run_grpo_episode(
                model=model,
                engine=engine_seeded,
                state=state,
                device=device,
                temperature=1.0,
                max_moves=cfg.moves_limit
            )
            group_states.append(states)
            group_actions.append(actions)
            group_old_log_probs.append(log_probs)
            group_rewards.append(r_tot)

        # 2. Compute Group-Relative Advantage
        rewards_arr = np.array(group_rewards, dtype=np.float32)
        mean_r = np.mean(rewards_arr)
        std_r = np.std(rewards_arr) + 1e-8
        advantages = (rewards_arr - mean_r) / std_r  # (G,)

        # 3. Policy gradient update
        total_loss = torch.tensor(0.0, device=device)
        total_steps = 0

        for g in range(group_size):
            if not group_states[g]:
                continue
            adv = float(advantages[g])

            obs_batch = torch.stack(group_states[g]).to(device)  # (T, 19, 9, 9)
            act_batch = torch.tensor(group_actions[g], dtype=torch.long, device=device)
            old_log_probs = torch.tensor(group_old_log_probs[g], dtype=torch.float32, device=device)

            logits, _ = model(obs_batch)
            log_probs = F.log_softmax(logits, dim=-1)
            new_act_log_probs = log_probs.gather(1, act_batch.unsqueeze(1)).squeeze(1)

            # Ratio r_t(theta)
            ratio = torch.exp(new_act_log_probs - old_log_probs)

            # PPO surrogate loss
            surr1 = ratio * adv
            surr2 = torch.clamp(ratio, 1.0 - eps_clip, 1.0 + eps_clip) * adv
            policy_loss = -torch.min(surr1, surr2).mean()

            # KL penalty vs frozen reference model
            with torch.no_grad():
                ref_logits, _ = ref_model(obs_batch)
                ref_probs = F.softmax(ref_logits, dim=-1)
            curr_probs = F.softmax(logits, dim=-1)
            kl = (curr_probs * (curr_probs.add(1e-8).log() - ref_probs.add(1e-8).log())).sum(dim=-1).mean()

            loss = policy_loss + (beta_kl * kl)
            total_loss = total_loss + loss
            total_steps += len(group_states[g])

        if total_steps > 0:
            optimizer.zero_grad()
            total_loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        # Update reference model periodically
        if it % 10 == 0:
            ref_model = copy.deepcopy(model).to(device)
            ref_model.eval()

        if it % 5 == 0 or it == episodes:
            print(f"Iteration {it:3d}/{episodes} | Lvl: {lvl_num:4d} | Mean Reward: {mean_r:7.1f} | Loss: {total_loss.item():.4f} | Steps: {total_steps:3d}")

    torch.save(model.state_dict(), checkpoint_out)
    print(f"\n--> Successfully saved GRPO checkpoint to: {checkpoint_out}")
    print(f"Elapsed Time: {time.time() - t0:.2f}s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Candy Crush GRPO Training")
    parser.add_argument("--episodes", type=int, default=20, help="Training iterations")
    parser.add_argument("--group_size", type=int, default=4, help="Rollouts per seed (G)")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--out", type=str, default="checkpoints/candy_crush_grpo_v1.pt", help="Output checkpoint")
    args = parser.parse_args()

    train_grpo(
        episodes=args.episodes,
        group_size=args.group_size,
        lr=args.lr,
        checkpoint_out=args.out
    )
