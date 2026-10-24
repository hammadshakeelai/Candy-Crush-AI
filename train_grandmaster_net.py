"""
Grandmaster Neural Policy & Value Network Training on Official King Levels.
Implements AlphaZero-style self-play training using expert trajectories from CANDYRL solver.
Trains policy head P(a|s) and value head V(s) to guide search in under 0.5ms.
"""

import os
import sys
import time
import argparse
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import List, Tuple, Dict, Optional, Any

from core.king_level_loader import KingLevelLoader
from core.reverse_engine import ReverseCandyEngine, ReverseEngineState, Direction
from models.network import CandyCrushPolicyValueNet
from models.superhuman_solver import SuperhumanSolver


def generate_expert_trajectories(
    loader: KingLevelLoader,
    engine: ReverseCandyEngine,
    solver: SuperhumanSolver,
    num_episodes: int = 40
) -> List[Tuple[np.ndarray, int, float]]:
    """
    Simulates expert gameplay across diverse King levels and records (s, a, z) data.
    """
    total_levels = loader.get_total_levels()
    dataset = []

    print(f"[Dataset Generator] Generating expert data across {num_episodes} random King levels...")

    for ep in range(1, num_episodes + 1):
        lvl_num = random.randint(1, min(500, total_levels))
        try:
            cfg = loader.load_level(lvl_num)
        except Exception:
            continue

        state = ReverseEngineState(cfg)
        engine.initialize_board(state)

        ep_transitions = []
        step = 0

        while not state.has_won() and not state.has_lost() and step < cfg.moves_limit:
            # Capture state tensor
            board_tensor = state.to_tensor() if hasattr(state, "to_tensor") else _encode_state_tensor(state)
            
            best_move = solver.select_best_move(state)
            if best_move is None:
                break

            r, c, direction, _ = best_move
            dir_idx = 0 if direction == Direction.RIGHT else 1
            action_idx = (r * state.width + c) * 2 + dir_idx

            ep_transitions.append((board_tensor, action_idx))
            engine.execute_move(state, r, c, direction)
            step += 1

        won = state.has_won()
        outcome_z = 1.0 if won else -0.5

        for s, a in ep_transitions:
            dataset.append((s, a, outcome_z))

        print(f"  Level {lvl_num:4d} | Steps: {step:2d} | Won: {won!s:5s} | Transitions Captured: {len(ep_transitions)} | Total Dataset: {len(dataset)}")

    return dataset


def _encode_state_tensor(state: ReverseEngineState) -> np.ndarray:
    """Encodes state to (19, H, W) tensor."""
    channels = 19
    h, w = state.height, state.width
    tensor = np.zeros((channels, h, w), dtype=np.float32)

    for color_val in range(7):
        tensor[color_val] = (state.colors == color_val).astype(np.float32)
    for s_idx in range(5):
        tensor[7 + s_idx] = (state.specials == s_idx).astype(np.float32)
    for j_val in range(3):
        tensor[12 + j_val] = (state.jellies == j_val).astype(np.float32)

    tensor[15] = (state.blockers == 1).astype(np.float32)  # Chocolate
    tensor[16] = (state.blockers == 2).astype(np.float32)  # Licorice
    tensor[17] = (state.blockers == 3).astype(np.float32)  # Frosting
    tensor[18].fill(min(1.0, max(0.0, state.moves_left / 50.0)))
    return tensor


def train_grandmaster_model(
    episodes: int = 30,
    epochs: int = 15,
    batch_size: int = 64,
    lr: float = 0.001,
    checkpoint_out: Optional[str] = None,
    output_model: Optional[str] = None,
):
    save_path = checkpoint_out or output_model or "checkpoints/candy_crush_grandmaster_v2.pt"
    print("=== LAUNCHING GRANDMASTER NEURAL TRAINING PIPELINE ===")
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)

    loader = KingLevelLoader()
    engine = ReverseCandyEngine()
    solver = SuperhumanSolver(engine=engine, search_depth=2, top_k_candidates=4)

    # 1. Generate / Gather Expert Dataset
    dataset = generate_expert_trajectories(loader, engine, solver, num_episodes=episodes)
    if not dataset:
        print("No expert trajectories collected.")
        return

    # 2. Setup PyTorch ResNet Model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Training on device: {device} | Total Samples: {len(dataset)}")

    model = CandyCrushPolicyValueNet(in_channels=19, num_res_blocks=4, channels=64).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    policy_loss_fn = nn.CrossEntropyLoss()
    value_loss_fn = nn.MSELoss()

    # 3. Training Loop
    all_states = np.array([item[0] for item in dataset], dtype=np.float32)
    all_actions = np.array([item[1] for item in dataset], dtype=np.int64)
    all_outcomes = np.array([item[2] for item in dataset], dtype=np.float32)

    dataset_size = len(all_states)

    for epoch in range(1, epochs + 1):
        indices = np.random.permutation(dataset_size)
        epoch_loss_p = 0.0
        epoch_loss_v = 0.0
        num_batches = 0

        for start_idx in range(0, dataset_size, batch_size):
            end_idx = min(start_idx + batch_size, dataset_size)
            batch_indices = indices[start_idx:end_idx]

            b_x = torch.tensor(all_states[batch_indices]).to(device)
            b_pi = torch.tensor(all_actions[batch_indices]).to(device)
            b_z = torch.tensor(all_outcomes[batch_indices]).unsqueeze(1).to(device)

            model.train()
            optimizer.zero_grad()
            p_logits, v_pred = model(b_x)

            loss_p = policy_loss_fn(p_logits, b_pi)
            loss_v = value_loss_fn(v_pred, b_z)
            total_loss = loss_p + (0.5 * loss_v)

            total_loss.backward()
            optimizer.step()

            epoch_loss_p += loss_p.item()
            epoch_loss_v += loss_v.item()
            num_batches += 1

        scheduler.step()
        avg_lp = epoch_loss_p / max(1, num_batches)
        avg_lv = epoch_loss_v / max(1, num_batches)
        print(f"Epoch {epoch:2d}/{epochs:2d} | Policy Loss: {avg_lp:.4f} | Value Loss: {avg_lv:.4f} | LR: {scheduler.get_last_lr()[0]:.6f}")

    # 4. Save Trained Checkpoint
    torch.save(model.state_dict(), save_path)
    print(f"\n--> Successfully saved Grandmaster model to: {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Grandmaster Neural Network")
    parser.add_argument("--episodes", type=int, default=25, help="Number of expert games to play")
    parser.add_argument("--epochs", type=int, default=10, help="Training epochs")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    args = parser.parse_args()

    train_grandmaster_model(episodes=args.episodes, epochs=args.epochs, batch_size=args.batch_size)
