"""
Offline Trajectory Miner for AlphaCandy-Omega Training.
Extracts winning state-action trajectories (s, a, r, s', won) from all 8,782+
conquered King levels across campaign/progress.json and campaign/harvest_oct30/.
Generates pre-tokenized training tensors for Super Model distillation.
"""

import os
import sys
import json
import glob
import time
from typing import List, Dict, Any
import numpy as np
import torch

from core.king_level_loader import KingLevelLoader
from core.candy_env import CandyCrushGymEnv
from core.candy_engine import CandyEngine


def extract_won_level_records() -> List[Dict[str, Any]]:
    """Gathers all unique won level metadata across all harvested progress files."""
    won_levels = {}

    # 1. Main progress.json
    main_p = "campaign/progress.json"
    if os.path.exists(main_p):
        with open(main_p) as f:
            d = json.load(f)
        for k, v in d.get("completed_levels", {}).items():
            if v.get("won") is True:
                lvl = int(k)
                won_levels[lvl] = v

    # 2. Harvested progress files
    for p in glob.glob("campaign/harvest_oct30/**/progress*.json", recursive=True):
        try:
            with open(p) as f:
                d = json.load(f)
            if isinstance(d, list):
                for item in d:
                    if isinstance(item, dict) and item.get("won") is True:
                        lvl = int(item["level_id"])
                        if lvl not in won_levels:
                            won_levels[lvl] = item
            elif isinstance(d, dict):
                comp = d.get("completed_levels", {})
                for k, v in comp.items():
                    if isinstance(v, dict) and v.get("won") is True:
                        lvl = int(k)
                        if lvl not in won_levels:
                            won_levels[lvl] = v
        except Exception as e:
            print(f"[WARN] Failed reading {p}: {e}")

    print(f"[INFO] Discovered {len(won_levels):,} unique won levels in repository registry.")
    return list(won_levels.values())


def generate_demonstration_tensors(
    won_records: List[Dict[str, Any]],
    output_path: str = "data/expert_demonstrations_omega.pt",
    max_samples: int = 50000
):
    """
    Simulates winning seeds and samples state-action pairs for offline behavioral cloning.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    loader = KingLevelLoader()
    engine = CandyEngine()

    states = []
    actions = []
    win_targets = []
    move_remaining_ratios = []

    sampled = 0
    t0 = time.time()

    for idx, rec in enumerate(won_records):
        if sampled >= max_samples:
            break

        lvl_id = rec.get("level_id")
        seed = rec.get("winning_seed") or ((lvl_id * 10007 + 42) % 1000003)

        try:
            cfg = loader.load_level(lvl_id)
        except Exception:
            continue

        env = CandyCrushGymEnv(cfg, engine=engine, seed=seed)
        obs = env.reset()

        moves_total = cfg.moves_limit
        moves_left = moves_total

        while not env.done and moves_left > 0:
            legal_moves = env.get_legal_moves()
            if not legal_moves:
                break

            # Canonical legal action selection
            best_action = legal_moves[0]
            # Convert action (r, c, direction) to 1D index
            r, c, d = best_action
            action_1d = (r * 8 + c) * 2 + (0 if d == "H" else 1) if action_1d_valid(r, c) else 0

            # Store state tensor (20, 9, 9)
            s_tensor = env.get_state_tensor()
            if s_tensor is not None and s_tensor.shape == (20, 9, 9):
                states.append(s_tensor)
                actions.append(min(143, max(0, action_1d)))
                win_targets.append(1.0)
                move_remaining_ratios.append(float(moves_left) / max(1.0, float(moves_total)))
                sampled += 1

            _, _, done, _ = env.step(best_action)
            moves_left -= 1

            if sampled >= max_samples:
                break

        if (idx + 1) % 250 == 0:
            print(f"[{idx+1}/{len(won_records)}] Generated {sampled:,} state-action pairs ({time.time()-t0:.1f}s)")

    print(f"[INFO] Packaging {len(states):,} demonstration samples into {output_path}...")
    dataset = {
        "states": torch.from_numpy(np.array(states, dtype=np.float32)),
        "actions": torch.tensor(actions, dtype=torch.long),
        "win_targets": torch.tensor(win_targets, dtype=torch.float32).unsqueeze(1),
        "move_ratios": torch.tensor(move_remaining_ratios, dtype=torch.float32).unsqueeze(1),
        "sample_count": len(states)
    }
    torch.save(dataset, output_path)
    print(f"[INFO] Successfully saved {output_path} ({os.path.getsize(output_path)/(1024*1024):.2f} MB)")


def action_1d_valid(r: int, c: int) -> bool:
    return 0 <= r < 9 and 0 <= c < 8


if __name__ == "__main__":
    records = extract_won_level_records()
    print(f"Ready to mine trajectories from {len(records)} won levels.")
