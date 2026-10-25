"""
Local Continuous Model Training Service for Candy Crush Saga.
Runs in parallel on the local machine:
  1. Continual Demonstration Imitation & Policy Refinement
  2. Group Relative Policy Optimization (GRPO) on real King levels
  3. Periodically updates checkpoints/alphacandy_spatial_transformer_local.pt
"""

import os
import sys
import time
import torch

from train_spatial_transformer_gpu import train_alphacandy_transformer_pipeline
from core.king_level_loader import KingLevelLoader

def run_local_training_daemon():
    print("=================================================================")
    print("  STARTING LOCAL CANDY CRUSH CONTINUOUS MODEL TRAINING SERVICE   ")
    print("=================================================================")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Local Trainer] Device: {device}")

    # Ensure level loader finds local data
    loader = KingLevelLoader()
    total_levels = loader.get_total_levels()
    print(f"[Local Trainer] Total accessible King levels: {total_levels}")

    round_idx = 1
    ckpt_local = "checkpoints/alphacandy_spatial_transformer_local.pt"

    while True:
        print(f"\n>>> [Round {round_idx}] Starting continuous training cycle (Episodes: 25)...")
        t0 = time.time()
        try:
            train_alphacandy_transformer_pipeline(
                episodes=25,
                pretrain_levels=20,
                epochs=3,
                group_size=4,
                lr=1e-4,
                checkpoint_out=ckpt_local
            )
            duration = time.time() - t0
            print(f">>> [Round {round_idx} Complete] Duration: {duration/60.0:.2f} min | Checkpoint: {ckpt_local}")
        except Exception as e:
            print(f"[Local Trainer] Exception in round {round_idx}: {e}")
            import traceback
            traceback.print_exc()

        round_idx += 1
        time.sleep(2)

if __name__ == "__main__":
    run_local_training_daemon()
