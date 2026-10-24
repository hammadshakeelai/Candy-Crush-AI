"""
Kaggle High-Performance GPU Training for Candy Crush Superhuman AI.
Executes on NVIDIA Tesla T4 / P100 GPU with CUDA acceleration.
Combines:
  1. Massive Expert Trajectory Collection (1,000 official King levels)
  2. Deep Dual-Head ResNet Training with Mixed Precision (fp16)
  3. Group Relative Policy Optimization (GRPO) Fine-Tuning
"""

import os
import sys
import time
import torch

# Ensure local path is accessible
sys.path.insert(0, os.path.abspath("."))

from train_grandmaster_net import train_grandmaster_model
from train_grpo import train_grpo


def main():
    print("=================================================================")
    print("  CANDY CRUSH SUPERHUMAN AI: KAGGLE GPU PRODUCTION PIPELINE      ")
    print("=================================================================")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Executing on Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU Name: {torch.cuda.get_device_name(0)}")
        print(f"Total VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

    out_dir = "/kaggle/working" if os.path.exists("/kaggle/working") else "checkpoints"
    os.makedirs(out_dir, exist_ok=True)

    ckpt_stage1 = os.path.join(out_dir, "candy_crush_stage1_imitation.pt")
    ckpt_final = os.path.join(out_dir, "candy_crush_grandmaster_superhuman.pt")

    t_start = time.time()

    # Stage 1: Expert Self-Play Imitation Learning (100 episodes, 15 epochs)
    print("\n>>> STAGE 1: EXPERT TRAJECTORY COLLECTION & RESNET PRE-TRAINING")
    train_grandmaster_model(
        episodes=100,
        epochs=15,
        batch_size=128,
        lr=1e-3,
        checkpoint_out=ckpt_stage1
    )

    # Stage 2: GRPO Fine-Tuning (Group Relative Policy Optimization)
    print("\n>>> STAGE 2: GRPO FINE-TUNING ON KING OFFICIAL LEVELS")
    train_grpo(
        episodes=100,
        group_size=6,
        lr=2e-4,
        eps_clip=0.2,
        beta_kl=0.015,
        checkpoint_out=ckpt_final,
        initial_checkpoint=ckpt_stage1
    )

    total_time = time.time() - t_start
    print("\n=================================================================")
    print("  KAGGLE GPU PIPELINE COMPLETE!                                  ")
    print(f"  Final Model Checkpoint: {ckpt_final}")
    print(f"  Total Duration: {total_time / 60.0:.2f} minutes")
    print("=================================================================")


if __name__ == "__main__":
    main()
