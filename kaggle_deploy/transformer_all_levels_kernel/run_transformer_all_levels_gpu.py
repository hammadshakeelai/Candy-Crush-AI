"""
Kaggle Production GPU Pipeline: AlphaCandy Spatial Attention Transformer Across All Levels.
Learns policy distributions P(a|s) and win probabilities P(Win|s) from gameplay across hundreds/thousands of King levels.
Features:
  - 20-channel spatial feature tensor representation
  - 4-Head Multi-Head Spatial Self-Attention (MHSA) treating 81 cells as interactive tokens
  - Expert grandmaster demonstration learning across diverse levels
  - Group Relative Policy Optimization (GRPO) reinforcement learning on complex topologies
  - Automated persistence to /kaggle/working/checkpoints/
"""

import os
import sys
import time
import json
import zipfile
import shutil
import random
import copy
import argparse
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

print("=================================================================")
print("  ALPHACANDY TRANSFORMER: ALL-LEVELS GPU LEARNING PIPELINE       ")
print("=================================================================")

# 1. Environment Discovery
if os.path.exists("/kaggle"):
    input_root = os.environ.get("KAGGLE_INPUT_ROOT", "/kaggle/input")
    work_dir = os.environ.get("KAGGLE_WORK_DIR", "/kaggle/working")
else:
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    input_root = os.environ.get("KAGGLE_INPUT_ROOT", os.path.join(repo_root, "kaggle_deploy/src"))
    work_dir = os.environ.get("KAGGLE_WORK_DIR", "/tmp/candy_trans_test")

os.makedirs(work_dir, exist_ok=True)
os.chdir(work_dir)

src_root = None
data_json_found = None
sqlite_found = None

for root, dirs, files in os.walk(input_root):
    if "candy_crush_pkg.zip" in files and src_root is None:
        zip_path = os.path.join(root, "candy_crush_pkg.zip")
        print(f"[INFO] Extracting {zip_path} into {work_dir}...")
        try:
            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                zip_ref.extractall(work_dir)
            src_root = work_dir
        except Exception as e:
            print(f"[WARN] Failed extracting zip: {e}")

    if "core" in dirs and "models" in dirs and src_root is None:
        src_root = root
        print(f"[INFO] Discovered source tree at: {src_root}")

    if "configs_v1.208.json" in files and not data_json_found:
        data_json_found = os.path.join(root, "configs_v1.208.json")
    if "candycrush_master_levels.sqlite" in files and not sqlite_found:
        sqlite_found = os.path.join(root, "candycrush_master_levels.sqlite")

# Fallback to current repo root if not found in input
if not src_root:
    candidate_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    if os.path.exists(os.path.join(candidate_root, "core", "king_level_loader.py")):
        src_root = candidate_root

# Copy files from src_root into work_dir if outside work_dir
if src_root and os.path.abspath(src_root) != os.path.abspath(work_dir):
    print(f"[INFO] Copying source files from {src_root} to {work_dir}...")
    for item in os.listdir(src_root):
        s_item = os.path.join(src_root, item)
        d_item = os.path.join(work_dir, item)
        if not os.path.exists(d_item):
            try:
                if os.path.isdir(s_item):
                    shutil.copytree(s_item, d_item)
                else:
                    shutil.copy2(s_item, d_item)
            except Exception as e:
                print(f"[WARN] Failed copying {item}: {e}")

if work_dir not in sys.path:
    sys.path.insert(0, work_dir)
if src_root and src_root not in sys.path:
    sys.path.insert(0, src_root)

if data_json_found:
    os.environ["KING_CONFIG_PATH"] = data_json_found
if sqlite_found:
    os.environ["KING_SQLITE_PATH"] = sqlite_found

# 2. Hardware Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Compute Device: {device}")
if torch.cuda.is_available():
    print(f"[INFO] GPU Name: {torch.cuda.get_device_name(0)}")
    print(f"[INFO] VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

from core.reverse_engine import ReverseEngineState, ReverseCandyEngine, Direction
from core.king_level_loader import KingLevelLoader
from models.spatial_transformer_net import (
    AlphaCandySpatialNet, encode_spatial_board_tensor,
    encode_action_index, decode_action_index
)
from models.candyrl_ensemble import GrandmasterCandyRLEnsemble


def collect_multi_level_demonstrations(
    loader: KingLevelLoader,
    engine: ReverseCandyEngine,
    ensemble: GrandmasterCandyRLEnsemble,
    num_levels: int = 150
) -> List[Tuple[np.ndarray, int, float, float]]:
    """
    Collects high-quality expert demonstrations across hundreds of diverse King levels.
    """
    print(f"\n[Expert Collection] Gathering gameplay trajectories across {num_levels} levels...")
    dataset = []
    level_ids = list(range(1, num_levels + 1))
    random.seed(42)
    random.shuffle(level_ids)

    wins = 0
    t0 = time.time()

    for idx, lvl_id in enumerate(level_ids):
        try:
            cfg = loader.load_level(lvl_id)
        except Exception:
            continue

        state = ReverseEngineState(cfg)
        engine.initialize_board(state, seed=42 + lvl_id * 31)

        episode_transitions = []
        step = 0
        max_steps = min(30, cfg.moves_limit)

        while not state.has_won() and not state.has_lost() and step < max_steps:
            legal = engine.get_legal_moves(state)
            if not legal:
                engine.initialize_board(state)
                legal = engine.get_legal_moves(state)
                if not legal:
                    break

            best_m = max(legal, key=lambda m: ensemble.evaluate_move(state, m[0], m[1], m[2]))
            r, c, direction = best_m
            a_idx = encode_action_index(r, c, direction)

            x_np = encode_spatial_board_tensor(state)
            score_gained, _ = engine.execute_move(state, r, c, direction)

            episode_transitions.append((x_np, a_idx))
            step += 1

        won = state.has_won()
        if won:
            wins += 1

        val_target = 1.0 if won else -0.5
        win_prob_target = 1.0 if won else 0.0

        for x_np, a_idx in episode_transitions:
            dataset.append((x_np, a_idx, val_target, win_prob_target))

        if (idx + 1) % max(1, num_levels // 5) == 0:
            print(f"  Processed {idx + 1}/{num_levels} levels | Total Transitions: {len(dataset)} | Won: {wins}")

    elapsed = time.time() - t0
    print(f"[Expert Collection] Complete: {len(dataset)} transitions from {wins}/{num_levels} levels in {elapsed:.1f}s")
    return dataset


def run_grpo_episode(
    model: AlphaCandySpatialNet,
    engine: ReverseCandyEngine,
    state: ReverseEngineState,
    max_moves: int = 25,
    temperature: float = 0.8
):
    states_list = []
    actions_list = []
    log_probs_list = []
    total_reward = 0.0
    step = 0
    moves_limit = min(max_moves, state.moves_left)

    while not state.has_won() and not state.has_lost() and step < moves_limit:
        legal = engine.get_legal_moves(state)
        if not legal:
            engine.initialize_board(state)
            legal = engine.get_legal_moves(state)
            if not legal:
                break

        x_np = encode_spatial_board_tensor(state)
        x_t = torch.from_numpy(x_np).unsqueeze(0).to(device)

        with torch.no_grad():
            masked_logits, _, _, _ = model.forward_with_mask(x_t, legal)

        scaled_logits = masked_logits / max(0.1, temperature)
        action_dist = torch.distributions.Categorical(logits=scaled_logits)
        a_idx = int(action_dist.sample().item())
        log_prob = float(action_dist.log_prob(torch.tensor(a_idx, device=device)).item())

        r, c, d = decode_action_index(a_idx)
        if (r, c, d) not in legal:
            r, c, d = legal[0]
            a_idx = encode_action_index(r, c, d)

        states_list.append(x_t.squeeze(0).cpu())
        actions_list.append(a_idx)
        log_probs_list.append(log_prob)

        prev_obj = state.remaining_objectives_count()
        score_gained, stats = engine.execute_move(state, r, c, d)
        new_obj = state.remaining_objectives_count()

        obj_cleared = max(0, prev_obj - new_obj)
        cascades = stats.get("cascades", 0)

        step_reward = (obj_cleared * 15.0) + (cascades * 2.0) + (score_gained / 500.0)
        total_reward += step_reward
        step += 1

    if state.has_won():
        total_reward += 200.0 + (state.moves_left * 5.0)
    elif state.has_lost():
        total_reward -= 50.0

    return states_list, actions_list, log_probs_list, total_reward


def train_transformer_all_levels(
    demonstration_levels: int = 150,
    pretrain_epochs: int = 8,
    grpo_episodes: int = 100,
    checkpoint_out: str = "checkpoints/alphacandy_spatial_transformer_master.pt"
) -> Dict[str, Any]:
    loader = KingLevelLoader()
    engine = ReverseCandyEngine()
    ensemble = GrandmasterCandyRLEnsemble(engine)

    model = AlphaCandySpatialNet(in_channels=20, hidden_dim=128, num_actions=144).to(device)

    # Load existing checkpoint if available
    for init_ckpt in [
        "checkpoints/alphacandy_spatial_transformer_local.pt",
        "checkpoints/alphacandy_spatial_transformer.pt",
        os.path.join(work_dir, "checkpoints/alphacandy_spatial_transformer_local.pt")
    ]:
        if os.path.exists(init_ckpt):
            try:
                model.load_state_dict(torch.load(init_ckpt, map_location=device))
                print(f"[INFO] Loaded pre-trained spatial transformer from {init_ckpt}")
                break
            except Exception:
                pass

    t0 = time.time()

    # Stage 1: Multi-Level Demonstrations
    print("\n>>> STAGE 1: MULTI-LEVEL SPATIAL DEMONSTRATION TRAINING")
    dataset = collect_multi_level_demonstrations(loader, engine, ensemble, num_levels=demonstration_levels)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    model.train()

    num_samples = len(dataset)
    batch_size = 64
    for epoch in range(1, pretrain_epochs + 1):
        random.shuffle(dataset)
        epoch_loss = 0.0
        correct_policy = 0
        total_batches = 0

        for b_start in range(0, num_samples, batch_size):
            batch = dataset[b_start : b_start + batch_size]
            if len(batch) < 4:
                continue

            x_batch = torch.from_numpy(np.array([item[0] for item in batch], dtype=np.float32)).to(device)
            a_batch = torch.tensor([item[1] for item in batch], dtype=torch.int64).to(device)
            v_batch = torch.tensor([item[2] for item in batch], dtype=torch.float32).unsqueeze(1).to(device)
            w_batch = torch.tensor([item[3] for item in batch], dtype=torch.float32).unsqueeze(1).to(device)

            optimizer.zero_grad()
            p_logits, v_scalar, w_prob = model(x_batch)

            loss_p = F.cross_entropy(p_logits, a_batch)
            loss_v = F.mse_loss(v_scalar, v_batch)
            loss_w = F.binary_cross_entropy(w_prob, w_batch)
            loss = loss_p + (0.5 * loss_v) + (0.5 * loss_w)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += loss.item()
            preds = p_logits.argmax(dim=-1)
            correct_policy += (preds == a_batch).sum().item()
            total_batches += 1

        avg_loss = epoch_loss / max(1, total_batches)
        acc = (correct_policy / float(num_samples)) * 100.0
        print(f"Epoch {epoch:2d}/{pretrain_epochs:2d} | Loss: {avg_loss:.4f} | Policy Match Acc: {acc:.2f}%")

    # Stage 2: GRPO Multi-Level Reinforcement Learning
    print("\n>>> STAGE 2: GROUP RELATIVE POLICY OPTIMIZATION (GRPO)")
    ref_model = copy.deepcopy(model)
    ref_model.eval()

    grpo_opt = torch.optim.AdamW(model.parameters(), lr=8e-5, weight_decay=1e-4)
    group_size = 4
    grpo_levels = [1, 5, 10, 15, 20, 23, 25, 28, 30, 35, 50, 61, 86, 100, 150, 250]

    grpo_losses = []
    grpo_rewards = []

    for ep in range(1, grpo_episodes + 1):
        lvl_id = grpo_levels[(ep - 1) % len(grpo_levels)]
        try:
            cfg = loader.load_level(lvl_id)
        except Exception:
            continue

        seed = 42 + ep * 193
        group_trajectories = []
        group_rewards_ep = []

        for _ in range(group_size):
            state = ReverseEngineState(cfg)
            engine.initialize_board(state, seed=seed)
            states_t, actions_t, old_lp, rew = run_grpo_episode(model, engine, state)
            group_trajectories.append((states_t, actions_t, old_lp))
            group_rewards_ep.append(rew)

        r_mean = float(np.mean(group_rewards_ep))
        r_std = float(np.std(group_rewards_ep)) + 1e-8
        advantages = [(r - r_mean) / r_std for r in group_rewards_ep]

        model.train()
        grpo_opt.zero_grad()
        total_grpo_loss = 0.0

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
            total_grpo_loss += loss.item()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        grpo_opt.step()

        grpo_losses.append(total_grpo_loss)
        grpo_rewards.append(r_mean)

        if ep % max(1, grpo_episodes // 5) == 0 or ep == grpo_episodes:
            print(f"GRPO Episode {ep:3d}/{grpo_episodes:3d} | Level: {lvl_id:3d} | Mean Reward: {r_mean:6.1f} | Loss: {total_grpo_loss:.4f}")

    total_time = time.time() - t0

    os.makedirs(os.path.dirname(os.path.abspath(checkpoint_out)), exist_ok=True)
    torch.save(model.state_dict(), checkpoint_out)
    print(f"\n[AlphaCandy Complete] Checkpoint saved successfully: {checkpoint_out}")

    summary = {
        "demonstration_levels": demonstration_levels,
        "demonstration_transitions": len(dataset),
        "pretrain_epochs": pretrain_epochs,
        "final_pretrain_accuracy": round(acc, 2),
        "grpo_episodes": grpo_episodes,
        "grpo_mean_reward": round(float(np.mean(grpo_rewards[-20:])), 2),
        "duration_minutes": round(total_time / 60.0, 2),
        "checkpoint_saved": checkpoint_out
    }

    with open("transformer_training_metrics.json", "w") as f:
        json.dump(summary, f, indent=2)

    report_md = f"""# AlphaCandy Spatial Transformer: Multi-Level GPU Training Report

## Performance Metrics
- **Demonstration Levels Sampled:** {summary['demonstration_levels']}
- **Transitions Learned:** {summary['demonstration_transitions']}
- **Final Policy Match Accuracy:** **{summary['final_pretrain_accuracy']}%**
- **GRPO Reinforcement Episodes:** {summary['grpo_episodes']}
- **Mean GRPO Reward:** {summary['grpo_mean_reward']}
- **Training Duration:** {summary['duration_minutes']} minutes
- **Final Checkpoint Artifact:** `{checkpoint_out}`
"""
    with open("TRANSFORMER_LEARNING_REPORT.md", "w") as f:
        f.write(report_md)

    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-Level Spatial Transformer GPU Pipeline")
    parser.add_argument("--demo-levels", type=int, default=int(os.environ.get("DEMO_LEVELS", "100")))
    parser.add_argument("--epochs", type=int, default=int(os.environ.get("PRETRAIN_EPOCHS", "8")))
    parser.add_argument("--grpo-episodes", type=int, default=int(os.environ.get("GRPO_EPISODES", "80")))
    args = parser.parse_args()

    train_transformer_all_levels(
        demonstration_levels=args.demo_levels,
        pretrain_epochs=args.epochs,
        grpo_episodes=args.grpo_episodes,
        checkpoint_out=os.path.join(work_dir, "checkpoints/alphacandy_spatial_transformer_master.pt")
    )
