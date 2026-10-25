---
title: Pure AlphaCandy Large Standalone Agent
tags:
  - architecture
  - pure-neural
  - spatial-transformer
  - high-throughput
created: 2026-10-25
---

# Pure AlphaCandy Large Standalone Agent

The **Pure AlphaCandy Large Standalone Agent** ([`models/alphacandy_large_agent.py`](file:///home/hammad/Antigravity_Workspace/Candy-Crush-AI/models/alphacandy_large_agent.py)) operates strictly via deep end-to-end neural inference over the 20-channel board tensor using the 14.4M-parameter [`AlphaCandyLargeNet`](file:///home/hammad/Antigravity_Workspace/Candy-Crush-AI/models/spatial_transformer_net.py).

---

## 🎯 Design Philosophy

Unlike ensemble methods or tree-search solvers that simulate future moves using an explicit game engine copy, the Pure Large Agent tests the hypothesis:
> *"Can a sufficiently deep, high-capacity spatial attention model solve complex match-3 boards in a single forward pass without lookahead?"*

### Key Advantages:
1. **Ultra-Low Latency**: Single forward pass takes **$< 2\text{ ms}$** on GPU Tensor Cores.
2. **Batched Throughput**: Can evaluate a batch of $B=64$ parallel board states simultaneously on a single GPU.
3. **High-Cascade Generation**: Deep multi-head self-attention naturally aligns multi-match explosive setups, averaging over 56,000 points per game.

---

## 🏗️ Technical Architecture

- **Trunk**: 6 Residual Convolutional Blocks with Mish activation ($C = 256$ channels).
- **Attention**: 2 separate Spatial Attention stages (8 heads each, dimension 32 per head) providing an infinite spatial receptive field across $9 \times 9$ discrete grids.
- **Action Selection**: Masked Softmax over 144 canonical swap indices:
  $$P(a_i | s) = \frac{\exp(z_i / \tau)}{\sum_{j \in \text{Legal}} \exp(z_j / \tau)}$$
- **Value & Win Probability**: Joint scalar heads predicting future utility $V(s) \in [-1, 1]$ and binary win likelihood $P(\text{Win}|s) \in [0, 1]$.

---

## 📈 Empirical Baseline

In the head-to-head milestone benchmark ([`campaign/benchmark_large_vs_consensus.py`](file:///home/hammad/Antigravity_Workspace/Candy-Crush-AI/campaign/benchmark_large_vs_consensus.py)):
- **Win Rate**: **60.0%** (Wins Levels 1, 5, 20 in 2–3 moves).
- **Move Latency**: **1.8 ms** per turn on GPU.
- **Score**: Averaged **56,561 points** per level.

---

## 🔗 Related Notes
- [[AlphaCandy_Large_Architecture]]
- [[Grandmaster_Consensus_Architecture]]
- [[01_Architecture_MOC]]
- [[Async_GPU_Parallelism_Architecture]]
