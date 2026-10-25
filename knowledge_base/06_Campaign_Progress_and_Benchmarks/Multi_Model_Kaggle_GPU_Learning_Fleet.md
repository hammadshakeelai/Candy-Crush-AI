---
title: Multi-Model Kaggle GPU Learning Fleet
tags:
  - benchmark
  - architecture
  - gpu-training
  - continual-learning
created: 2026-10-25
---

# Multi-Model Kaggle GPU Learning Fleet

To scale neural training across the entire 10,295+ King level blueprint database, we deployed a distributed multi-paradigm fleet across Kaggle GPU instances and local continuous compute daemons.

---

## 🚀 Active Kaggle GPU Fleet Overview

```
                                  ┌───────────────────────────────┐
                                  │   Official King Level Repo    │
                                  │   (10,295 Level Blueprints)   │
                                  └───────────────┬───────────────┘
                                                  │
                 ┌────────────────────────────────┴────────────────────────────────┐
                 ▼                                                                 ▼
┌─────────────────────────────────┐                             ┌──────────────────────────────────┐
│   Kaggle Tesla T4 GPU (Fleet 1) │                             │   Kaggle Tesla T4 GPU (Fleet 2)  │
│   candy-crush-dqn-learning-gpu  │                             │ alphacandy-transformer-all-levels│
├─────────────────────────────────┤                             ├──────────────────────────────────┤
│ • Double Dueling DQN + PER      │                             │ • 20-channel Spatial Transformer │
│ • 144 Canonical Swaps Action    │                             │ • Multi-Head Self-Attention      │
│ • Sharded Multi-Level Replay    │                             │ • Multi-Level Demonstration BC   │
│ • TD-Error Prioritization       │                             │ • Group Relative Policy Opt (GRPO│
└────────────────┬────────────────┘                             └─────────────────┬────────────────┘
                 │                                                                │
                 └───────────────────────────────┬────────────────────────────────┘
                                                 ▼
                                 ┌───────────────────────────────┐
                                 │   Grandmaster Unified Agent   │
                                 │   Ensemble Fusion (Inference) │
                                 └───────────────────────────────┘
```

---

## 🛠️ Kernel 1: Double Dueling DQN Multi-Level Learner
- **Kernel URL**: [kaggle.com/code/hammadshakeelai/candy-crush-dqn-learning-gpu](https://www.kaggle.com/code/hammadshakeelai/candy-crush-dqn-learning-gpu)
- **Deployment Path**: `kaggle_deploy/dqn_gpu_kernel/run_dqn_all_levels_gpu.py`
- **Architecture**:
  - Double Dueling Deep Q-Network ($Q(s, a) = V(s) + (A(s, a) - \frac{1}{|A|}\sum A(s, a'))$)
  - Prioritized Experience Replay (PER) with rank-based priority ($\alpha=0.6, \beta=0.4 \to 1.0$)
  - Multi-level episode rollouts across varying level geometries and blockers
- **Objective**: Learn universal value approximations across 144 action coordinates invariant to target objectives (Jelly, Ingredients, Drop, Score).

---

## 🧠 Kernel 2: AlphaCandy Spatial Transformer & GRPO All-Levels Trainer
- **Kernel URL**: [kaggle.com/code/hammadshakeelai/alphacandy-transformer-all-levels](https://www.kaggle.com/code/hammadshakeelai/alphacandy-transformer-all-levels)
- **Deployment Path**: `kaggle_deploy/transformer_all_levels_kernel/run_transformer_all_levels_gpu.py`
- **Two-Stage Curriculum**:
  1. **Spatial Self-Attention Demonstration Pre-Training**: Collects trajectories across diverse King levels and trains the spatial transformer using cross-entropy loss against high-quality moves.
  2. **Group Relative Policy Optimization (GRPO)**: Simulates groups of $G=4$ parallel rollouts per level state, computes relative advantage $A_i = \frac{r_i - \text{mean}(r)}{\text{std}(r) + \epsilon}$, and updates policy parameters without requiring a separate critic network.
- **Output Artifact**: `checkpoints/alphacandy_spatial_transformer_master.pt`

---

## ⚡ Kernel 3: Async 1,000 Games Parallel Benchmark
- **Kernel URL**: [kaggle.com/code/hammadshakeelai/candy-crush-async-1000-games](https://www.kaggle.com/code/hammadshakeelai/candy-crush-async-1000-games)
- **Completed Result**:
  - Levels Tested: **1,000 / 1,000**
  - Pass-2 Overall Win Rate: **74.4% (744 / 1,000)**
  - Attempt-1 First-Try Win Rate: **89.38% (665 / 744)**
  - Ingredients Mode Win Rate: **87.10% (243 / 279)**
  - Runtime: **9 minutes 24 seconds** (0 errors)

---

## 🔄 Local Continuous Training Daemon
- **Script**: `train_local_continuous.py`
- **Execution**: Continuous background daemon (`task-2283`)
- **Status**: Iterating through continuous GRPO loops (>62 rounds, >1,500 episodes) updating `checkpoints/alphacandy_spatial_transformer_local.pt`.

---

## 🔗 Related MOCs and Notes
- [[05_Campaign_and_Benchmarks_MOC]]
- [[01_Architecture_MOC]]
- [[Async_1000_Games_Kaggle_Benchmark]]
- [[Async_GPU_Parallelism_Architecture]]
- [[Zero_Error_Fault_Tolerance_Guards]]
- [[Double_Dueling_DQN_PER]]
- [[AlphaCandy_Spatial_Transformer]]
