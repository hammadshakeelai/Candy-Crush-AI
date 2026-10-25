"""
Kaggle Production GPU Pipeline: Double Dueling Deep Q-Learning (DQN) Across All Levels.
Learns optimal Q-values and action advantages A(s, a) from gameplay across hundreds/thousands of King levels.
Features:
  - Double Q-Learning Bellman target decoupling
  - Prioritized Experience Replay (PER) with importance-sampling correction
  - Action-masked Q-value maximization over 144 canonical swaps
  - Polyak target network stabilization (tau=0.005)
  - Broad level sampling across Levels 1 to 1,000+
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
from typing import Dict, Any, List, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

print("=================================================================")
print("  DOUBLE DUELING DQN: MULTI-LEVEL GPU LEARNING PIPELINE          ")
print("=================================================================")

# 1. Environment Discovery
if os.path.exists("/kaggle"):
    input_root = os.environ.get("KAGGLE_INPUT_ROOT", "/kaggle/input")
    work_dir = os.environ.get("KAGGLE_WORK_DIR", "/kaggle/working")
else:
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    input_root = os.environ.get("KAGGLE_INPUT_ROOT", os.path.join(repo_root, "kaggle_deploy/src"))
    work_dir = os.environ.get("KAGGLE_WORK_DIR", "/tmp/candy_dqn_test")

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
from models.spatial_transformer_net import encode_spatial_board_tensor, encode_action_index
from models.candy_dqn import DuelingQNetwork, PrioritizedReplayBuffer


class MultiLevelDQNTrainer:
    """Scales Double Dueling DQN training across hundreds/thousands of levels."""

    def __init__(
        self,
        checkpoint_init: Optional[str] = None,
        gamma: float = 0.96,
        lr: float = 3e-4,
        tau: float = 0.005,
        buffer_size: int = 50000,
        batch_size: int = 64
    ):
        self.device = device
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size

        self.online_net = DuelingQNetwork(in_channels=20, hidden_dim=128, num_actions=144).to(self.device)

        # Load existing weights if available
        for candidate_ckpt in [
            checkpoint_init,
            "checkpoints/candy_crush_dueling_dqn.pt",
            os.path.join(work_dir, "checkpoints/candy_crush_dueling_dqn.pt")
        ]:
            if candidate_ckpt and os.path.exists(candidate_ckpt):
                try:
                    self.online_net.load_state_dict(torch.load(candidate_ckpt, map_location=self.device))
                    print(f"[INFO] Loaded pre-trained weights from {candidate_ckpt}")
                    break
                except Exception:
                    pass

        self.target_net = copy.deepcopy(self.online_net).to(self.device)
        self.target_net.eval()

        self.optimizer = torch.optim.AdamW(self.online_net.parameters(), lr=lr, weight_decay=1e-4)
        self.memory = PrioritizedReplayBuffer(capacity=buffer_size)

    def update_step(self) -> float:
        if len(self.memory) < self.batch_size:
            return 0.0

        batch, indices, is_weights = self.memory.sample(self.batch_size)

        states_t = torch.from_numpy(batch["states"]).to(self.device)
        actions_t = torch.from_numpy(batch["actions"]).unsqueeze(1).to(self.device)
        rewards_t = torch.from_numpy(batch["rewards"]).unsqueeze(1).to(self.device)
        next_states_t = torch.from_numpy(batch["next_states"]).to(self.device)
        dones_t = torch.from_numpy(batch["dones"]).unsqueeze(1).to(self.device)
        weights_t = torch.from_numpy(is_weights).unsqueeze(1).to(self.device)

        # 1. Q(s, a) from Online Net
        q_eval = self.online_net(states_t).gather(1, actions_t)

        # 2. Double DQN target computation
        with torch.no_grad():
            next_q_online = self.online_net(next_states_t)
            next_masks = torch.zeros_like(next_q_online, dtype=torch.bool)
            for b_idx, legal_indices in enumerate(batch["legal_next"]):
                for a_idx in legal_indices:
                    if 0 <= a_idx < 144:
                        next_masks[b_idx, a_idx] = True

            next_q_online_masked = torch.where(next_masks, next_q_online, torch.tensor(-1e9, device=self.device))
            next_best_actions = next_q_online_masked.argmax(dim=-1, keepdim=True)
            next_q_target = self.target_net(next_states_t).gather(1, next_best_actions)
            q_target = rewards_t + (1.0 - dones_t) * self.gamma * next_q_target

        td_errors = (q_eval - q_target).detach().squeeze(-1).cpu().numpy()
        self.memory.update_priorities(indices, td_errors)

        loss = (weights_t * F.smooth_l1_loss(q_eval, q_target, reduction="none")).mean()

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.online_net.parameters(), max_norm=1.0)
        self.optimizer.step()

        # Polyak Target Update
        for target_param, online_param in zip(self.target_net.parameters(), self.online_net.parameters()):
            target_param.data.copy_(self.tau * online_param.data + (1.0 - self.tau) * target_param.data)

        return float(loss.item())

    def train_multi_level(
        self,
        total_episodes: int = 500,
        max_level: int = 500,
        eps_start: float = 0.40,
        eps_end: float = 0.05,
        checkpoint_path: str = "checkpoints/candy_crush_dueling_dqn_master.pt"
    ) -> Dict[str, Any]:
        loader = KingLevelLoader()
        engine = ReverseCandyEngine()

        print(f"\n>>> LAUNCHING DQN MULTI-LEVEL LEARNING ({total_episodes} Episodes across Levels 1 to {max_level})")
        t0 = time.time()
        total_wins = 0
        total_moves = 0
        losses = []

        # Available levels pool
        level_candidates = list(range(1, max_level + 1))

        for ep in range(1, total_episodes + 1):
            lvl_id = random.choice(level_candidates)
            try:
                cfg = loader.load_level(lvl_id)
            except Exception:
                continue

            state = ReverseEngineState(cfg)
            engine.initialize_board(state, seed=42 + ep * 79)

            epsilon = max(eps_end, eps_start - (ep / float(total_episodes)) * (eps_start - eps_end))
            step = 0
            ep_reward = 0.0
            ep_loss = 0.0
            loss_count = 0

            max_steps = min(30, cfg.moves_limit)

            while not state.has_won() and not state.has_lost() and step < max_steps:
                legal = engine.get_legal_moves(state)
                if not legal:
                    engine.initialize_board(state)
                    legal = engine.get_legal_moves(state)
                    if not legal:
                        break

                s_np = encode_spatial_board_tensor(state)
                s_t = torch.from_numpy(s_np).unsqueeze(0).to(self.device)

                a_idx, (r, c, d), _ = self.online_net.select_action(s_t, legal, epsilon=epsilon)

                prev_obj = state.remaining_objectives_count()
                score_gained, stats = engine.execute_move(state, r, c, d)
                new_obj = state.remaining_objectives_count()

                obj_cleared = max(0, prev_obj - new_obj)
                cascades = stats.get("cascades", 0)

                # Shaped multi-paradigm reward
                reward = (obj_cleared * 12.0) + (cascades * 3.0) + (score_gained / 800.0)
                done = state.has_won() or state.has_lost() or (step + 1 >= max_steps)

                if state.has_won():
                    reward += 150.0 + (state.moves_left * 4.0)
                elif state.has_lost():
                    reward -= 30.0

                s_next_np = encode_spatial_board_tensor(state)
                next_legal = engine.get_legal_moves(state)
                next_legal_indices = [encode_action_index(nr, nc, nd) for nr, nc, nd in next_legal]

                self.memory.push(s_np, a_idx, reward, s_next_np, done, next_legal_indices)

                # Train step
                l = self.update_step()
                if l > 0:
                    ep_loss += l
                    loss_count += 1

                ep_reward += reward
                step += 1
                total_moves += 1

            if state.has_won():
                total_wins += 1

            if loss_count > 0:
                losses.append(ep_loss / loss_count)

            if ep % max(1, total_episodes // 10) == 0 or ep == total_episodes:
                avg_l = float(np.mean(losses[-20:])) if losses else 0.0
                win_pct = (total_wins / float(ep)) * 100.0
                elapsed = time.time() - t0
                print(
                    f"[{ep:4d}/{total_episodes:4d}] Lvl {lvl_id:4d} | "
                    f"Win: {win_pct:5.1f}% | Loss: {avg_l:6.4f} | "
                    f"Buf: {len(self.memory):5d} | Eps: {epsilon:.2f} | "
                    f"Elapsed: {elapsed/60:4.1f}m"
                )

                # Periodically checkpoint
                os.makedirs(os.path.dirname(os.path.abspath(checkpoint_path)), exist_ok=True)
                torch.save(self.online_net.state_dict(), checkpoint_path)

        os.makedirs(os.path.dirname(os.path.abspath(checkpoint_path)), exist_ok=True)
        torch.save(self.online_net.state_dict(), checkpoint_path)
        total_time = time.time() - t0

        summary = {
            "total_episodes": total_episodes,
            "max_level_pool": max_level,
            "total_wins": total_wins,
            "win_rate_percent": round((total_wins / max(1, total_episodes)) * 100.0, 2),
            "total_transitions_trained": total_moves,
            "replay_buffer_size": len(self.memory),
            "duration_minutes": round(total_time / 60.0, 2),
            "final_loss": round(float(np.mean(losses[-50:])) if losses else 0.0, 4),
            "checkpoint_saved": checkpoint_path
        }

        with open("dqn_training_metrics.json", "w") as f:
            json.dump(summary, f, indent=2)

        report_md = f"""# Double Dueling DQN: Multi-Level GPU Training Report

## Performance Metrics
- **Total Episodes Trained:** {summary['total_episodes']}
- **Level Pool Explored:** Levels 1 to {summary['max_level_pool']}
- **Win Rate During Training:** **{summary['win_rate_percent']}%**
- **Total Transitions Sampled:** {summary['total_transitions_trained']}
- **Replay Buffer Capacity:** {summary['replay_buffer_size']} transitions
- **Training Duration:** {summary['duration_minutes']} minutes
- **Final Huber Loss:** {summary['final_loss']}
- **Checkpoint Artifact:** `{checkpoint_path}`
"""
        with open("DQN_LEARNING_REPORT.md", "w") as f:
            f.write(report_md)

        print("\n==============================================================")
        print("          DOUBLE DUELING DQN TRAINING COMPLETE                 ")
        print("==============================================================")
        print(f"Total Wins       : {total_wins}/{total_episodes} ({summary['win_rate_percent']}%)")
        print(f"Total Transitions: {total_moves}")
        print(f"Final Checkpoint : {checkpoint_path}")
        print(f"Report Generated : DQN_LEARNING_REPORT.md")
        print("==============================================================\n")
        return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-Level DQN GPU Pipeline")
    parser.add_argument("--episodes", type=int, default=int(os.environ.get("DQN_EPISODES", "300")))
    parser.add_argument("--max-level", type=int, default=int(os.environ.get("DQN_MAX_LEVEL", "500")))
    parser.add_argument("--lr", type=float, default=3e-4)
    args = parser.parse_args()

    trainer = MultiLevelDQNTrainer(lr=args.lr)
    trainer.train_multi_level(
        total_episodes=args.episodes,
        max_level=args.max_level,
        checkpoint_path=os.path.join(work_dir, "checkpoints/candy_crush_dueling_dqn_master.pt")
    )
