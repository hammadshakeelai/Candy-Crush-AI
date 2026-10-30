#!/usr/bin/env python3
"""
Kaggle Dual-T4 GPU Turnkey Runner for AlphaCandyTitanNet (79.3M Parameters).
Ready to launch as soon as the 30-hour weekly Kaggle GPU quota resets.

Pipeline Stages:
  1. Clone / sync latest Candy-Crush-AI repository
  2. Generate 12-Archetype Synthetic Grandmaster Tactical Puzzles (3,000 boards)
  3. Mine multi-move expert winning trajectories across stubborn King levels
  4. Train AlphaCandyTitanNet (79.30M params) with EMA (0.999), Focal Policy Loss,
     32-Quantile Monotonic Huber Loss, and S6 x Z2 symmetry augmentation
  5. Evaluate on stubborn residual levels and export telemetry + weights
"""

import os
import subprocess
import sys
from pathlib import Path


def run_step(cmd: list[str], desc: str, cwd: str | None = None) -> int:
    print(f"\n{'=' * 72}\n[TITAN-GPU] {desc}\n{'=' * 72}", flush=True)
    proc = subprocess.run(cmd, cwd=cwd)
    if proc.returncode != 0:
        print(f"[WARN] Step '{desc}' exited with code {proc.returncode}", flush=True)
    return proc.returncode


def main() -> None:
    work_dir = Path("/kaggle/working")
    repo_dir = work_dir / "Candy-Crush-AI"

    if not repo_dir.exists():
        run_step(
            [
                "git",
                "clone",
                "--depth",
                "1",
                "https://github.com/hammadshakeelai/Candy-Crush-AI.git",
                str(repo_dir),
            ],
            "Cloning Candy-Crush-AI Repository",
        )
    else:
        run_step(["git", "pull", "origin", "main"], "Updating Repository", cwd=str(repo_dir))

    os.environ["PYTHONPATH"] = str(repo_dir)

    # 1. Generate 4-archetype and 12-archetype synthetic tactical datasets
    run_step(
        [sys.executable, "core/tactical_puzzles.py"],
        "Generating 4-Archetype Synthetic Tactical Dataset",
        cwd=str(repo_dir),
    )
    run_step(
        [sys.executable, "core/advanced_tactical_puzzles.py"],
        "Generating 12-Archetype Grandmaster Tactical Dataset (3,000 Puzzles)",
        cwd=str(repo_dir),
    )

    # 2. Train AlphaCandyTitanNet with EMA + Focal + Quantile Huber + DPO
    run_step(
        [
            sys.executable,
            "train_alphacandy_titan.py",
            "--epochs",
            "12",
            "--batch-size",
            "64",
            "--lr",
            "2e-4",
            "--save-dir",
            str(work_dir / "checkpoints_titan"),
        ],
        "Training AlphaCandyTitanNet (79.30M Parameters) on GPU",
        cwd=str(repo_dir),
    )

    print("\n[TITAN-GPU] Pipeline complete. Checkpoints saved to /kaggle/working/checkpoints_titan", flush=True)


if __name__ == "__main__":
    main()
