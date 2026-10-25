"""
Production GPU Training Pipeline for AlphaCandy Spatial Transformer.
Combines:
  1. Multi-Head Spatial Self-Attention (4-head, 20-channel input)
  2. Expert Playout Imitation Pre-training
  3. Group Relative Policy Optimization (GRPO) for Match-3 reasoning
Optimized for local verification and NVIDIA Tesla T4 / P100 Kaggle GPUs.
"""

from typing import List, Tuple, Dict, Any, Optional
import os
import sys
import time
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from core.reverse_engine import (
    ReverseEngineState, ReverseCandyEngine, Direction, LevelType, CandyColor, SpecialCandy, Blocker
)
from core.king_level_loader import KingLevelLoader
from models.spatial_transformer_net import (
    AlphaCandySpatialNet, encode_action_index, decode_action_index, encode_spatial_board_tensor
)
from models.superhuman_solver import SuperhumanSolver


def collect_expert_plannings(
    loader: KingLevelLoader,
    engine: ReverseCandyEngine,
    solver: SuperhumanSolver,
    num_levels: int = 50,
    seed_offset: int = 1000
) -> List[Dict[str, Any]]:
    """
    Collects expert demonstration transitions (state_tensor, action_idx, value, won)
    from the Superhuman tactical solver across official King levels.
    """
    dataset: List[Dict[str, Any]] = []
    load_failures = 0

    for i in range(num_levels):
        lvl_id = (i % 500) + 1
        try:
            cfg = loader.load_level(lvl_id)
        except Exception as e:
            load_failures += 1
            if load_failures <= 3:
                print(f"[WARN] Failed to load level {lvl_id}: {type(e).__name__}: {e}")
            continue

        state = ReverseEngineState(cfg)
        engine.initialize_board(state, seed=seed_offset + i)

        moves_limit = min(20, state.moves_left)
        step = 0

        while not state.has_won() and not state.has_lost() and step < moves_limit:
            legal_moves = engine.get_legal_moves(state)
            if not legal_moves:
                engine.initialize_board(state)
                legal_moves = engine.get_legal_moves(state)
                if not legal_moves:
                    break

            # Expert action choice
            best = solver.select_best_move(state)
            if best is None:
                break
            r, c, direction, score = best
            a_idx = encode_action_index(r, c, direction)

            x_np = encode_spatial_board_tensor(state)

            dataset.append({
                "state": x_np,
                "action": a_idx,
                "score": score,
                "step": step,
                "won": False  # Back-filled after game completes
            })

            score_gained, _ = engine.execute_move(state, r, c, direction)
            step += 1

        is_won = state.has_won()
        # Back-fill outcome
        for entry in dataset[-step:] if step > 0 else []:
            entry["won"] = is_won

    if load_failures == num_levels:
        raise RuntimeError(
            f"All {num_levels} levels failed to load — level data is missing. "
            f"sqlite_path={loader.sqlite_path} config_path={loader.config_path}"
        )
    print(f"[Expert Collection] {len(dataset)} transitions from {num_levels - load_failures}/{num_levels} levels")
    return dataset


def pretrain_spatial_transformer(
    model: AlphaCandySpatialNet,
    dataset: List[Dict[str, Any]],
    device: torch.device,
    epochs: int = 5,
    batch_size: int = 64,
    lr: float = 1e-3,
    checkpoint_out: Optional[str] = None
) -> None:
    """Pre-trains AlphaCandy Spatial Transformer on expert demonstrations."""
    if not dataset:
        raise RuntimeError("Expert dataset is empty — refusing to continue with an untrained model.")

    print(f"\n[AlphaCandy Pre-Training] Dataset size: {len(dataset)} transitions | Epochs: {epochs}")
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion_ce = nn.CrossEntropyLoss()
    criterion_bce = nn.BCELoss()

    indices = np.arange(len(dataset))

    for epoch in range(1, epochs + 1):
        np.random.shuffle(indices)
        total_loss = 0.0
        total_p_acc = 0
        batches = 0

        for start_idx in range(0, len(dataset), batch_size):
            batch_indices = indices[start_idx : start_idx + batch_size]
            b_states = np.array([dataset[idx]["state"] for idx in batch_indices], dtype=np.float32)
            b_actions = np.array([dataset[idx]["action"] for idx in batch_indices], dtype=np.int64)
            b_wons = np.array([[1.0 if dataset[idx]["won"] else 0.0] for idx in batch_indices], dtype=np.float32)

            t_x = torch.from_numpy(b_states).to(device)
            t_actions = torch.from_numpy(b_actions).to(device)
            t_wons = torch.from_numpy(b_wons).to(device)

            optimizer.zero_grad()
            logits, val_scalar, win_prob = model(t_x)

            loss_p = criterion_ce(logits, t_actions)
            loss_w = criterion_bce(win_prob, t_wons)
            loss = loss_p + (0.5 * loss_w)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            preds = logits.argmax(dim=-1)
            total_p_acc += (preds == t_actions).sum().item()
            batches += 1

        avg_loss = total_loss / max(1, batches)
        acc = (total_p_acc / len(dataset)) * 100.0
        print(f"Epoch {epoch:2d}/{epochs:2d} | Loss: {avg_loss:.4f} | Policy Match Acc: {acc:.2f}%")

    if checkpoint_out:
        os.makedirs(os.path.dirname(os.path.abspath(checkpoint_out)), exist_ok=True)
        torch.save(model.state_dict(), checkpoint_out)
        print(f"[AlphaCandy Pre-Training] Checkpoint saved: {checkpoint_out}")


def run_grpo_spatial_episode(
    model: AlphaCandySpatialNet,
    engine: ReverseCandyEngine,
    state: ReverseEngineState,
    device: torch.device,
    temperature: float = 0.8,
    max_moves: int = 25
) -> Tuple[List[torch.Tensor], List[int], List[float], float]:
    """Runs a single GRPO exploration rollout with spatial action sampling."""
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

        x_np = encode_spatial_board_tensor(state)
        x_t = torch.from_numpy(x_np).unsqueeze(0).to(device)

        with torch.no_grad():
            masked_logits, probs, _, _ = model.forward_with_mask(x_t, legal_moves)

        # Temperature-scaled sampling over legal actions
        scaled_logits = masked_logits / max(0.1, temperature)
        action_dist = torch.distributions.Categorical(logits=scaled_logits)
        a_idx = int(action_dist.sample().item())
        log_prob = float(action_dist.log_prob(torch.tensor(a_idx)).item())

        r, c, direction = decode_action_index(a_idx)

        # Fallback if invalid
        if (r, c, direction) not in legal_moves:
            r, c, direction = legal_moves[0]
            a_idx = encode_action_index(r, c, direction)

        states_list.append(x_t.squeeze(0).cpu())
        actions_list.append(a_idx)
        log_probs_list.append(log_prob)

        prev_obj = state.remaining_objectives_count()
        score_gained, stats = engine.execute_move(state, r, c, direction)
        new_obj = state.remaining_objectives_count()

        obj_cleared = max(0, prev_obj - new_obj)
        cascades = stats.get("cascades", 0)

        step_reward = (obj_cleared * 15.0) + (cascades * 2.0) + (score_gained / 500.0)
        total_reward += step_reward
        step_count += 1

    if state.has_won():
        total_reward += 200.0 + (state.moves_left * 5.0)
    elif state.has_lost():
        total_reward -= 50.0

    return states_list, actions_list, log_probs_list, total_reward


def train_alphacandy_transformer_pipeline(
    episodes: int = 30,
    pretrain_levels: int = 25,
    epochs: int = 3,
    group_size: int = 4,
    lr: float = 1e-4,
    checkpoint_out: str = "checkpoints/alphacandy_spatial_transformer.pt"
) -> AlphaCandySpatialNet:
    """Full execution pipeline: Demonstration Imitation + GRPO Reinforcement Learning."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=================================================================")
    print("  ALPHACANDY SPATIAL TRANSFORMER GPU TRAINING PIPELINE           ")
    print(f"  Device: {device} | Total GRPO Episodes: {episodes}            ")
    print("=================================================================")

    loader = KingLevelLoader()
    engine = ReverseCandyEngine()
    solver = SuperhumanSolver(engine=engine, search_depth=2, top_k_candidates=4)

    model = AlphaCandySpatialNet(in_channels=20, hidden_dim=128, num_actions=144).to(device)

    # 1. Collect Demonstrations and Pre-Train
    print("\n>>> STAGE 1: SPATIAL TRANSFORMER DEMONSTRATION PRE-TRAINING")
    dataset = collect_expert_plannings(loader, engine, solver, num_levels=pretrain_levels)
    pretrain_spatial_transformer(
        model, dataset, device=device, epochs=epochs, batch_size=64, lr=1e-3
    )

    # 2. GRPO Reinforcement Learning
    print("\n>>> STAGE 2: GROUP RELATIVE POLICY OPTIMIZATION (GRPO)")
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    ref_model = copy.deepcopy(model)
    ref_model.eval()

    test_levels = [1, 5, 10, 15, 20, 23, 25, 30, 50, 100]
    completed_episodes = 0

    for ep in range(1, episodes + 1):
        lvl_id = test_levels[(ep - 1) % len(test_levels)]
        try:
            cfg = loader.load_level(lvl_id)
        except Exception as e:
            print(f"[WARN] GRPO: failed to load level {lvl_id}: {type(e).__name__}: {e}")
            continue
        completed_episodes += 1

        seed = 42 + (ep * 137)
        group_trajectories = []
        group_rewards = []

        # Sample G rollouts from identical starting seed
        for _ in range(group_size):
            state = ReverseEngineState(cfg)
            engine.initialize_board(state, seed=seed)
            states_t, actions_t, old_log_probs, rew = run_grpo_spatial_episode(
                model, engine, state, device, temperature=0.8
            )
            group_trajectories.append((states_t, actions_t, old_log_probs))
            group_rewards.append(rew)

        r_mean = float(np.mean(group_rewards))
        r_std = float(np.std(group_rewards)) + 1e-8
        advantages = [(r - r_mean) / r_std for r in group_rewards]

        # PPO / GRPO Update Step
        model.train()
        optimizer.zero_grad()
        total_loss = 0.0
        total_steps = 0

        for i in range(group_size):
            states_i, actions_i, old_lp_i = group_trajectories[i]
            adv_i = advantages[i]
            if not states_i:
                continue

            batch_x = torch.stack(states_i).to(device)
            batch_a = torch.tensor(actions_i, dtype=torch.int64).to(device)
            batch_old_lp = torch.tensor(old_lp_i, dtype=torch.float32).to(device)

            logits, _, _ = model(batch_x)
            dist = torch.distributions.Categorical(logits=logits)
            new_lp = dist.log_prob(batch_a)

            ratio = torch.exp(new_lp - batch_old_lp)
            surr1 = ratio * adv_i
            surr2 = torch.clamp(ratio, 0.8, 1.2) * adv_i
            policy_loss = -torch.min(surr1, surr2).mean()

            with torch.no_grad():
                ref_logits, _, _ = ref_model(batch_x)
                ref_dist = torch.distributions.Categorical(logits=ref_logits)

            kl_div = torch.distributions.kl_divergence(dist, ref_dist).mean()
            loss = policy_loss + (0.015 * kl_div)

            loss.backward()
            total_loss += loss.item()
            total_steps += len(states_i)

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        if ep % max(1, episodes // 5) == 0 or ep == episodes:
            print(f"GRPO Episode {ep:3d}/{episodes:3d} | Level: {lvl_id:3d} | Mean Reward: {r_mean:6.1f} | Loss: {total_loss:.4f}")

    if completed_episodes == 0:
        raise RuntimeError("No GRPO episodes ran — level data is missing.")
    print(f"[GRPO] Completed {completed_episodes}/{episodes} episodes")

    # Save final checkpoint
    os.makedirs(os.path.dirname(os.path.abspath(checkpoint_out)), exist_ok=True)
    torch.save(model.state_dict(), checkpoint_out)
    print(f"\n[AlphaCandy Complete] Checkpoint saved successfully: {checkpoint_out}")

    return model


if __name__ == "__main__":
    train_alphacandy_transformer_pipeline(
        episodes=10,
        pretrain_levels=10,
        epochs=2,
        group_size=4,
        lr=1e-4,
        checkpoint_out="checkpoints/alphacandy_spatial_transformer.pt"
    )
