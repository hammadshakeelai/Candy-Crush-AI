"""
Synthetic Tactical Dataset Generator.
Synthesizes 2,000 ground-truth tactical micro-puzzles and encodes them into
canonical 20-channel PyTorch demonstration tensors (s, a*, won=1.0)
for the 80/20 curriculum blend.
"""

import os
import sys
import time
import numpy as np
import torch

sys.path.insert(0, os.getcwd())
from core.tactical_puzzles import TacticalPuzzleGenerator
from core.reverse_engine import ReverseEngineState, Direction
from models.spatial_transformer_net import encode_spatial_board_tensor, encode_action_index


def generate_tactical_tensors(
    output_path: str = "data/synthetic_tactics_dataset.pt",
    count_per_archetype: int = 500
):
    print("=" * 65)
    print("   SYNTHETIC TACTICAL DATASET GENERATOR (GRANDMASTER CURRICULUM)")
    print("=" * 65)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    generator = TacticalPuzzleGenerator(seed=42)

    puzzles = generator.generate_batch(count_per_type=count_per_archetype)
    total_puzzles = len(puzzles)
    print(f"[INFO] Generated {total_puzzles:,} synthetic puzzles across 4 tactical archetypes.")

    states = []
    actions = []
    win_targets = []
    move_ratios = []
    archetypes = []

    t0 = time.time()
    for idx, p in enumerate(puzzles):
        cfg = p["config"]
        r, c, d_str = p["optimal_action"]
        trick = p["trick_name"]

        state = ReverseEngineState(cfg)
        tensor_np = encode_spatial_board_tensor(state)
        direction = Direction.RIGHT if d_str == "H" else Direction.DOWN
        act_idx = encode_action_index(r, c, direction)

        states.append(tensor_np)
        actions.append(act_idx)
        win_targets.append(1.0)
        move_ratios.append(1.0)  # Maximum move efficiency
        archetypes.append(trick)

        if (idx + 1) % 500 == 0:
            print(f"[{idx+1:4d}/{total_puzzles}] Encoded {len(states):,} tensors ({time.time()-t0:.1f}s)")

    print(f"\n[INFO] Saving {len(states):,} tactical demonstration tensors to {output_path}...")
    dataset = {
        "states": torch.from_numpy(np.array(states, dtype=np.float32)),
        "actions": torch.tensor(actions, dtype=torch.long),
        "win_targets": torch.tensor(win_targets, dtype=torch.float32).unsqueeze(1),
        "move_ratios": torch.tensor(move_ratios, dtype=torch.float32).unsqueeze(1),
        "archetypes": archetypes,
        "sample_count": len(states)
    }
    torch.save(dataset, output_path)
    file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"[SUCCESS] Dataset generated successfully: {output_path} ({file_size_mb:.2f} MB)")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    generate_tactical_tensors(count_per_archetype=500)
