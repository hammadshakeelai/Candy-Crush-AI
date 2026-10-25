---
title: AlphaCandy Large Architecture (~14.4M Parameters)
tags:
  - architecture
  - neural-networks
  - spatial-transformer
  - deep-learning
created: 2026-10-25
---

# AlphaCandy Large Architecture (~14.4M Parameters)

The **AlphaCandy Large** architecture (`AlphaCandyLargeNet` in `models/spatial_transformer_net.py`) is a high-capacity, dual-head spatial attention network specifically engineered to conquer complex multi-hop board layouts, multi-layer chocolate spawners, magic mixers, and long cascade sequences across thousands of official King levels.

---

## 🏗️ Architectural Specifications

| Component | Standard AlphaCandy (`AlphaCandySpatialNet`) | **AlphaCandy Large (`AlphaCandyLargeNet`)** |
| :--- | :--- | :--- |
| **Total Parameters** | ~2.9 Million | **~14.42 Million** |
| **Hidden Channels ($C$)** | 128 | **256** |
| **Self-Attention Stages** | 1 stage (4 Attention Heads) | **2 stages (8 Attention Heads each)** |
| **Residual Conv Blocks** | 4 blocks (Mish + BatchNorm) | **6 blocks (Mish + BatchNorm)** |
| **Policy Head FC Layer** | Linear(5,184 $\to$ 256) $\to$ 144 | **Linear(10,368 $\to$ 512) $\to$ 144** |
| **Value Head FC Layer** | Linear(2,592 $\to$ 128) $\to$ 2 | **Linear(5,184 $\to$ 256) $\to$ 2** |
| **Dropout Regularization** | 0.10 | **0.15** |

---

## 🔄 Network Topology & Flow

```mermaid
flowchart TD
    In["Input Tensor (B, 20, 9, 9)"] --> Stem["Stem Conv 3x3 (256 ch, Mish, BN)"]
    Stem --> Res1["ResBlock 1 & 2 (256 ch)"]
    Res1 --> Attn1["Spatial Attention 1 (8 Heads, 256 ch)<br>Global Island Cross-Attention"]
    Attn1 --> Res2["ResBlock 3 & 4 (256 ch)"]
    Res2 --> Attn2["Spatial Attention 2 (8 Heads, 256 ch)<br>Multi-Hop Cascade Flow Context"]
    Attn2 --> Res3["ResBlock 5 & 6 (256 ch)"]
    
    Res3 --> PConv["Policy Conv 1x1 (128 ch)"]
    PConv --> PFC["FC Linear(10,368 -> 512 -> 144)"]
    PFC --> Policy["Masked Policy Distribution P(a|s)"]
    
    Res3 --> VConv["Value Conv 1x1 (64 ch)"]
    VConv --> VFC["FC Linear(5,184 -> 256 -> 2)"]
    VFC --> Value["Expected Value V(s) & Win Probability P(Win|s)"]
```

---

## ⚡ Deployment & Kaggle Async Scaling

`AlphaCandyLargeNet` is integrated into:
1. **[`GrandmasterUnifiedAgent`](file:///home/hammad/Antigravity_Workspace/Candy-Crush-AI/models/grandmaster_unified_agent.py)** via `model_size="large"`.
2. **Kaggle Kernel**: [`alphacandy-async-grandmaster-large-gpu`](https://www.kaggle.com/code/hammadshakeelai/alphacandy-async-grandmaster-large-gpu), which evaluates large-scale campaigns across 2,500+ official King levels in parallel.

---

## 🔗 Related Notes
- [[01_Architecture_MOC]]
- [[AlphaCandy_Spatial_Transformer]]
- [[Grandmaster_Unified_Agent]]
- [[Async_GPU_Parallelism_Architecture]]
- [[Multi_Model_Kaggle_GPU_Learning_Fleet]]
