#!/usr/bin/env python3
"""
Comprehensive Dataset Generator for Candy Crush AI
===================================================
Constructs two major foundation datasets from the 10,295-level campaign conquest:
1. Direct Preference Optimization (DPO) Dataset (`data/candycrush_dpo_preference_pairs.pt`)
   - Pairs (state, winning_action, losing_action, margin) mined from 20,098 won transitions.
2. Dense Offline RL Replay Buffer (`data/candycrush_offline_rl_buffer.pt`)
   - Transition tuples (s, a, r, s', done, return_to_go, moves_left) for Decision Transformer / CQL / IQL.
"""

import os
import sys
import time
from pathlib import Path
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

DATA_DIR = REPO_ROOT / "data"
RUNS_DIR = REPO_ROOT / "runs"

def build_dpo_and_offline_rl_datasets(max_dpo_pairs: int = 150000, gamma: float = 0.99):
    print("=" * 80)
    print("🍬 ALPHACANDY COMPREHENSIVE DATASET GENERATOR")
    print("=" * 80)
    
    t0 = time.time()
    golden_path = RUNS_DIR / "fortress_golden_trajectories.pt"
    if not golden_path.exists():
        print(f"[ERROR] Golden trajectories file not found at {golden_path}")
        return

    print(f"[1/4] Loading golden trajectories from {golden_path}...")
    trajectories = torch.load(golden_path, map_location="cpu", weights_only=False)
    n_transitions = len(trajectories)
    print(f"      Loaded {n_transitions:,} golden transitions across solved campaign levels.")

    # -------------------------------------------------------------
    # 1. Build Direct Preference Optimization (DPO) Contrastive Pairs
    # -------------------------------------------------------------
    print(f"\n[2/4] Mining DPO Preference Pairs (Target: ~{max_dpo_pairs:,} pairs)...")
    dpo_states = []
    dpo_globals = []
    dpo_preferred = []
    dpo_rejected = []
    dpo_margins = []

    pairs_count = 0
    rng = np.random.RandomState(42)

    for item in trajectories:
        mask = item["mask"]
        pi = item["pi"]
        planes = item["planes"]   # (24, 9, 9) uint8
        globals_vec = item["globals"] # (16,) float32
        
        a_win = int(np.argmax(pi))
        legal_actions = np.where(mask)[0]
        suboptimal_actions = [a for a in legal_actions if a != a_win]
        
        if not suboptimal_actions:
            continue
            
        # Select up to 8 rejected actions per winning state to avoid over-representation of simple boards
        if len(suboptimal_actions) > 8:
            selected_rejected = rng.choice(suboptimal_actions, size=8, replace=False)
        else:
            selected_rejected = suboptimal_actions

        for a_rej in selected_rejected:
            # Margin is 1.0 (win) minus probability of suboptimal move
            margin = float(1.0 - pi[a_rej])
            dpo_states.append(planes)
            dpo_globals.append(globals_vec)
            dpo_preferred.append(a_win)
            dpo_rejected.append(int(a_rej))
            dpo_margins.append(margin)
            pairs_count += 1
            if pairs_count >= max_dpo_pairs:
                break
        if pairs_count >= max_dpo_pairs:
            break

    print(f"      Mined {pairs_count:,} high-contrast preference pairs.")
    dpo_dataset = {
        "states": torch.from_numpy(np.stack(dpo_states)),           # (N, 24, 9, 9) uint8
        "globals": torch.from_numpy(np.stack(dpo_globals)),         # (N, 16) float32
        "preferred_actions": torch.tensor(dpo_preferred, dtype=torch.long),
        "rejected_actions": torch.tensor(dpo_rejected, dtype=torch.long),
        "margins": torch.tensor(dpo_margins, dtype=torch.float32),
        "num_pairs": pairs_count,
        "format": "DPO_v2_24Planes_16Globals",
        "description": "Direct Preference Optimization pairs mined from 10,295-level Tree DAG winning solutions."
    }
    
    dpo_output_path = DATA_DIR / "candycrush_dpo_preference_pairs.pt"
    print(f"      Saving DPO dataset to {dpo_output_path}...")
    torch.save(dpo_dataset, dpo_output_path)
    print(f"      Saved {dpo_output_path.name} ({dpo_output_path.stat().st_size / (1024*1024):.2f} MB)")

    # -------------------------------------------------------------
    # 2. Build Dense Offline RL Transition Buffer
    # -------------------------------------------------------------
    print(f"\n[3/4] Unrolling Trajectories into Offline RL Transition Buffer...")
    
    states_buf = []
    globals_buf = []
    actions_buf = []
    rewards_buf = []
    next_states_buf = []
    next_globals_buf = []
    dones_buf = []
    episode_ids_buf = []
    timesteps_buf = []
    
    episode_id = 0
    timestep = 0
    
    # We step through trajectories and identify transitions (s_t, a_t, r_t, s_{t+1}, done)
    for i in range(n_transitions - 1):
        curr = trajectories[i]
        nxt = trajectories[i+1]
        
        curr_moves = curr["globals"][0]
        nxt_moves = nxt["globals"][0]
        curr_rem = curr["globals"][1]
        nxt_rem = nxt["globals"][1]
        
        # Action taken at current state
        a_t = int(np.argmax(curr["pi"]))
        
        # Check if episode ends at current step
        # If next moves_ratio jumps up or remaining ratio increases significantly, it's a new episode
        is_done = (nxt_moves >= curr_moves) or (nxt_rem > curr_rem + 0.1)
        
        # Calculate step reward:
        # Objective progress delta + step penalty + win bonus
        obj_delta = max(0.0, float(curr_rem - nxt_rem))
        step_reward = obj_delta * 10.0 - 0.05
        if is_done:
            step_reward += 10.0  # Terminal win bonus
            
        states_buf.append(curr["planes"])
        globals_buf.append(curr["globals"])
        actions_buf.append(a_t)
        rewards_buf.append(step_reward)
        next_states_buf.append(nxt["planes"] if not is_done else curr["planes"])
        next_globals_buf.append(nxt["globals"] if not is_done else curr["globals"])
        dones_buf.append(is_done)
        episode_ids_buf.append(episode_id)
        timesteps_buf.append(timestep)
        
        if is_done:
            episode_id += 1
            timestep = 0
        else:
            timestep += 1

    # Compute Returns-To-Go (Discounted Return G_t)
    print(f"      Computing discounted returns-to-go (gamma={gamma})...")
    n_buf = len(rewards_buf)
    returns_to_go = np.zeros(n_buf, dtype=np.float32)
    
    # Backward pass per episode to compute G_t = r_t + gamma * G_{t+1}
    running_return = 0.0
    for t in reversed(range(n_buf)):
        if dones_buf[t]:
            running_return = rewards_buf[t]
        else:
            running_return = rewards_buf[t] + gamma * running_return
        returns_to_go[t] = running_return

    offline_rl_buffer = {
        "states": torch.from_numpy(np.stack(states_buf)),             # (N, 24, 9, 9) uint8
        "globals": torch.from_numpy(np.stack(globals_buf)),           # (N, 16) float32
        "actions": torch.tensor(actions_buf, dtype=torch.long),
        "rewards": torch.tensor(rewards_buf, dtype=torch.float32),
        "next_states": torch.from_numpy(np.stack(next_states_buf)),   # (N, 24, 9, 9) uint8
        "next_globals": torch.from_numpy(np.stack(next_globals_buf)), # (N, 16) float32
        "dones": torch.tensor(dones_buf, dtype=torch.bool),
        "returns_to_go": torch.from_numpy(returns_to_go),
        "episode_ids": torch.tensor(episode_ids_buf, dtype=torch.long),
        "timesteps": torch.tensor(timesteps_buf, dtype=torch.long),
        "total_transitions": n_buf,
        "total_episodes": episode_id + 1,
        "description": "Dense Offline RL Replay Buffer from 10,295-level campaign conquest."
    }

    rl_output_path = DATA_DIR / "candycrush_offline_rl_buffer.pt"
    print(f"      Saving Offline RL Replay Buffer to {rl_output_path}...")
    torch.save(offline_rl_buffer, rl_output_path)
    print(f"      Saved {rl_output_path.name} ({rl_output_path.stat().st_size / (1024*1024):.2f} MB)")

    print(f"\n[4/4] Dataset Generation Completed in {time.time() - t0:.2f}s!")
    print(f"      - DPO Pairs:        {pairs_count:,}")
    print(f"      - RL Transitions:   {n_buf:,} across {episode_id + 1:,} won episodes")
    print("=" * 80)

if __name__ == "__main__":
    build_dpo_and_offline_rl_datasets()
