"""
Self-Play & Reinforcement Learning Training Pipeline for Candy Crush.
Trains the Dual-Head ResNet using simulated experience and MCTS targets (AlphaZero style).
"""

import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from core.candy_board import BoardState, Direction
from core.candy_env import CandyCrushGymEnv
from models.network import CandyCrushPolicyValueNet
from models.heuristics import CandyCrushHeuristics
from models.mcts import CandyCrushMCTS


def train_self_play(
    num_episodes: int = 50,
    mcts_sims: int = 20,
    batch_size: int = 32,
    lr: float = 0.001,
    device: str = "cpu"
):
    print(f"=== Starting Candy Crush AI Training Pipeline ({device}) ===")
    os.makedirs("checkpoints", exist_ok=True)

    env = CandyCrushGymEnv(height=9, width=9, max_moves=25)
    model = CandyCrushPolicyValueNet(in_channels=19, num_res_blocks=4, channels=48).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    policy_criterion = nn.CrossEntropyLoss()
    value_criterion = nn.MSELoss()

    heuristics = CandyCrushHeuristics(env.engine)
    mcts = CandyCrushMCTS(env.engine, heuristics, num_simulations=mcts_sims)

    replay_buffer = []

    for ep in range(1, num_episodes + 1):
        obs, info = env.reset()
        episode_data = []
        terminated = False
        steps = 0
        ep_score = 0

        t0 = time.time()
        while not terminated and steps < 30:
            legal_moves = env.engine.get_legal_moves(env.state)
            if not legal_moves:
                break

            # Run MCTS to get policy distribution and best move
            best_move = mcts.search(env.state, policy_net=model)
            r, c, direction = best_move
            dir_idx = 0 if direction == Direction.RIGHT else 1
            action_idx = (r * env.width + c) * 2 + dir_idx

            # Target policy (one-hot on MCTS selection)
            target_policy = action_idx

            # Store transition (obs, target_policy)
            episode_data.append((obs, target_policy))

            # Step environment
            obs, reward, terminated, _, step_info = env.step(action_idx)
            ep_score = step_info["score"]
            steps += 1

        duration = time.time() - t0
        won = step_info.get("won", False)
        # Outcome value z in [-1, 1]
        z = 1.0 if won else math_tanh_reward(ep_score)

        # Store transitions with terminal value
        for s, a_target in episode_data:
            replay_buffer.append((s, a_target, z))

        print(f"Episode {ep:3d}/{num_episodes} | Steps: {steps:2d} | Score: {ep_score:6d} | Won: {won!s:5s} | Time: {duration:.2f}s | Buffer: {len(replay_buffer)}")

        # Train step if enough buffer samples
        if len(replay_buffer) >= batch_size:
            indices = np.random.choice(len(replay_buffer), batch_size, replace=False)
            batch = [replay_buffer[i] for i in indices]

            b_obs = torch.tensor(np.array([item[0] for item in batch]), dtype=torch.float32).to(device)
            b_target_pi = torch.tensor(np.array([item[1] for item in batch]), dtype=torch.long).to(device)
            b_target_z = torch.tensor(np.array([item[2] for item in batch]), dtype=torch.float32).unsqueeze(1).to(device)

            model.train()
            optimizer.zero_grad()
            p_logits, v_pred = model(b_obs)

            loss_p = policy_criterion(p_logits, b_target_pi)
            loss_v = value_criterion(v_pred, b_target_z)
            total_loss = loss_p + loss_v

            total_loss.backward()
            optimizer.step()

        # Checkpoint every 25 episodes
        if ep % 25 == 0:
            ckpt_path = f"checkpoints/candy_crush_net_ep{ep}.pt"
            torch.save(model.state_dict(), ckpt_path)
            print(f"--> Saved checkpoint: {ckpt_path}")

    # Final save
    torch.save(model.state_dict(), "checkpoints/candy_crush_net_latest.pt")
    print("=== Training Complete. Model saved to checkpoints/candy_crush_net_latest.pt ===")


def math_tanh_reward(score: int) -> float:
    return float(np.tanh(score / 8000.0))


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_self_play(num_episodes=5, mcts_sims=10, batch_size=16, device=device)
