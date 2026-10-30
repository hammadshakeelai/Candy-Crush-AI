---
name: alphacandy-zero-trainer
description: >-
  Train and evaluate AlphaCandyZeroNet using Gumbel AlphaZero Sequential Halving,
  Prioritized Level Replay (PLR), Go-Explore near-miss endgame restarts, and
  honest multi-seed evaluation (`alphacandy_zero.train` and `alphacandy_zero.evaluate`).
---

# AlphaCandy-Zero Expert Iteration & Honest Evaluation Skill

Use this skill to train `AlphaCandyZeroNet` (`tiny`, `cpu`, `base`, `large` presets) via self-play expert iteration or to run honest fixed-seed evaluations without seed-hunting inflation.

## 1. Honest Multi-Seed Evaluation (`alphacandy_zero.evaluate`)

Evaluate any checkpoint (or the legacy 20-channel `alphacandy_large_net_master.pt` via `--use-legacy-master`) across fixed seeds per level:

```bash
PYTHONPATH=. /home/hammad/Antigravity_Workspace/Candy-Crush-AI/venv/bin/python3 -m alphacandy_zero.evaluate \
  --levels 1,25,50,235,272,279 \
  --preset cpu \
  --sims 24 \
  --considered 8 \
  --seeds 3 \
  --flip-ensemble
```

## 2. Self-Play Expert Iteration with PLR & Near-Miss Restarts (`alphacandy_zero.train`)

- **Local Smoke Test (<15s)**:
  ```bash
  PYTHONPATH=. /home/hammad/Antigravity_Workspace/Candy-Crush-AI/venv/bin/python3 -m alphacandy_zero.train \
    --levels 1-10 --preset tiny --sims 8 --games-per-iter 6 --iterations 2 --workers 3 --max-moves-override 6 --out /tmp/zero_smoke
  ```
- **Kaggle 4-vCPU / GPU Production Run**:
  ```bash
  PYTHONPATH=. python3 -m alphacandy_zero.train \
    --levels 1-2500 \
    --focus-file campaign/unsolved_levels_1_2500.json \
    --preset cpu \
    --sims 32 \
    --games-per-iter 96 \
    --p-restart 0.30 \
    --time-budget-hours 11.0 \
    --out /kaggle/working/alphacandy_zero
  ```
