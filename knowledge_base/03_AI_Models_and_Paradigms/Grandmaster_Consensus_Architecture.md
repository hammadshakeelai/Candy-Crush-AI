---
title: Grandmaster Consensus Architecture (Large + DQN Master + Lookahead)
tags:
  - architecture
  - grandmaster
  - ensemble
  - lookahead
  - expectation-search
created: 2026-10-25
---

# Grandmaster Consensus Architecture (Large + DQN Master + Lookahead)

The **Grandmaster Consensus Agent** ([`models/grandmaster_consensus_agent.py`](file:///home/hammad/Antigravity_Workspace/Candy-Crush-AI/models/grandmaster_consensus_agent.py)) unifies three orthogonal reinforcement learning and heuristic paradigms into an adaptive consensus engine with multi-depth expectimax lookahead.

---

## 🏛️ Tri-Paradigm Synthesis

```
                               ┌─────────────────────────────────┐
                               │     Board State Tensor s        │
                               └───────────────┬─────────────────┘
                                               │
               ┌───────────────────────────────┼───────────────────────────────┐
               ▼                               ▼                               ▼
┌─────────────────────────────┐ ┌─────────────────────────────┐ ┌─────────────────────────────┐
│    AlphaCandy Large (14.4M) │ │  Double Dueling DQN Master  │ │   CANDYRL Expert Ensemble   │
│  2-Stage Spatial Attention  │ │  Prioritized Replay (PER)   │ │  Tactical Blocker Heuristics│
├─────────────────────────────┤ ├─────────────────────────────┤ ├─────────────────────────────┤
│ • Policy Prior P(a|s)       │ │ • Universal Q-values Q(s, a)│ │ • Blocker Proximity H(s, a) │
│ • Win Probability P(Win|s)  │ │ • 144 Canonical Swaps       │ │ • Special Candy Combos      │
└──────────────┬──────────────┘ └──────────────┬──────────────┘ └──────────────┬──────────────┘
               │                               │                               │
               └───────────────────────┬───────┴───────────────────────────────┘
                                       │
                                       ▼
                       ┌───────────────────────────────┐
                       │ Adaptive Phase-Weighted Layer │
                       │    S = w_t·P + w_d·Q + w_h·H  │
                       └───────────────┬───────────────┘
                                       │ Top-K Candidates (k=6)
                                       ▼
                       ┌───────────────────────────────┐
                       │  Expectimax Tree Lookahead    │
                       │  (Depth 1–3 with Beam Pruning)│
                       └───────────────┬───────────────┘
                                       │
                                       ▼
                             Optimal Swap Action a*
```

---

## ⚖️ Dynamic Phase-Weighted Consensus

Consensus weights dynamically adjust according to game phase:

$$\text{Score}(s, a) = w_{\text{trans}} \cdot P(a|s) + w_{\text{dqn}} \cdot \tilde{Q}(s, a) + w_{\text{heur}} \cdot \tilde{H}(s, a)$$

| Game Phase | Move Criteria | $w_{\text{trans}}$ (Spatial Prior) | $w_{\text{dqn}}$ (Q-Values) | $w_{\text{heur}}$ (Tactical Blocker) | Strategic Objective |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **Opening** | $\text{moves} > 15$ ($>50\%$) | **0.40** | **0.35** | **0.25** | Broad board clearing, space creation, combo alignment |
| **Midgame** | $6 \le \text{moves} \le 15$ | **0.50** | **0.25** | **0.25** | Targeted objective reduction, multi-hop cascades |
| **Endgame** | $\text{moves} \le 5$ ($\le 20\%$) | **0.30** | **0.10** | **0.60** | Strict objective fulfillment, emergency moves |

---

## 🌲 Adaptive Multi-Depth Lookahead Engine

Instead of brute-force evaluation of all 144 swaps, the agent:
1. Extracts the top $k=6$ candidates from the consensus layer.
2. Evaluates immediate forward state transitions using `ReverseCandyEngine.execute_move`.
3. If $P(\text{Win}|s) \in [0.25, 0.65]$ (high tension) or $\text{moves} \le 4$, expands search to **Depth 2 or 3**.
4. Prunes inferior branches immediately if any candidate achieves an instant level win.

---

## 🏆 Head-to-Head Milestone Results

In empirical validation across milestone King levels ([`campaign/benchmark_large_vs_consensus.py`](file:///home/hammad/Antigravity_Workspace/Candy-Crush-AI/campaign/benchmark_large_vs_consensus.py)):
- **Win Rate**: **5/5 WON (100.0%)**
- **Move Efficiency**: **5.8 moves** average per level (over $2\times$ faster than pure neural).
- **Hard Level Mastery**: Effortlessly solved Level 10 (Jelly) and Level 25 (Ingredients with Licorice) where single-model baselines struggled.

---

## 🔗 Related Notes
- [[Pure_AlphaCandy_Large_Agent]]
- [[AlphaCandy_Large_Architecture]]
- [[01_Architecture_MOC]]
- [[Grandmaster_Unified_Agent]]
- [[Double_Dueling_DQN_PER]]
