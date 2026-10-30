# 🧬 AlphaCandy Model Zoo: Micro-Experiments Empirical Benchmark Report

## Executive Summary
Executed all **6 Core Next-Gen Architectural Micro-Experiments** across real King levels and canonical synthetic tactical puzzles in **244.39s**.

| Exp ID | Micro-Experiment | Key Empirical Finding | Status |
| :---: | :--- | :--- | :---: |
| **EXP-01** | **64-Bit Zobrist MCGS Transposition** | **12.37%** Cache Hit Rate (1.14x Effective Search Speedup) | ✅ Verified |
| **EXP-02** | **Gumbel Sequential Halving vs. Beam ($N=8$)** | **4145.5** vs. **3710.7** Mean Return (**+11.72%** Gain) | ✅ Verified |
| **EXP-03** | **Desperation-Aware Quantile Tilting ($\text{CVaR}_{0.85}^+$)** | **13.5%** vs. **0.0%** Static $\text{CVaR}_{0.25}$ (**+13.5%** Endgame Recovery) | ✅ Verified |
| **EXP-04** | **Wormhole Portal Multigraph MPNN** | **18.3350** vs. **0.0000** CNN 1-Hop Portal Jacobian (Dist=14) | ✅ Verified |
| **EXP-05** | **Tactical Archetype Recognition (5 Puzzles)** | **3 / 4 (75.0%)** Canonical Tricks Prioritized | ✅ Verified |
| **EXP-06** | **$S_6 \times \mathbb{Z}_2$ Symmetry & Equivariance ($1,440\times$)** | Max Policy Error **$Z_2$: 0.00e+00**, **$S_6$: 8.82e-06** | ✅ Verified |

---

## Detailed Experiment Breakdown

### 1. EXP-01: 64-Bit Zobrist Transposition Table in MCGS
- **Nodes Visited:** 396
- **Transposition Cache Hits:** 49 (**12.37%**)
- **Conclusion:** Independent spatial swaps on $9\times 9$ boards frequently commute into identical board states. Upgrading MCTS to **Monte Carlo Graph Search (MCGS)** with 64-bit Zobrist hashing saves 12.4% of redundant neural forward passes.

### 2. EXP-02: Gumbel Sequential Halving vs. Standard Beam Search
- **Gumbel-Top-$k$ + Sequential Halving Mean Return:** `4145.52`
- **Standard Flat Beam Search Mean Return:** `3710.74`
- **Relative Improvement:** `+11.72%`
- **Conclusion:** Allocating simulation budget iteratively via Sequential Halving concentrates rollouts on the highest-utility swaps while preserving exploration via Gumbel-Max sampling without replacement.

### 3. EXP-03: Desperation-Aware Dynamic Quantile Tilting
- **Static Risk-Averse ($\text{CVaR}_{0.25}^-$) Desperate Win Rate:** `0.00%`
- **Risk-Neutral ($\mathbb{E}[Z]$) Desperate Win Rate:** `1.50%`
- **Desperation-Tilted ($\text{CVaR}_{0.85}^+$) Win Rate:** `13.50%`
- **Real King Endgame Utility (24 Snapshots):** `58475.3` (Upper-Tail) vs `42926.9` (Lower-Tail)
- **Conclusion:** When behind on move budget ($\mathcal{D}(s_t) > 1.35$), static risk aversion locks in guaranteed defeat. Dynamically tilting to the 85th+ quantile hunts high-variance cascade miracles.

### 4. EXP-04: Wormhole Portal Graph Message Passing
- **Portal Distance:** Manhattan distance = `14` cells (`(8, 1) -> (0, 7)`)
- **2-Layer $3\times 3$ CNN Cross-Portal Sensitivity:** `0.000000`
- **Wormhole MPNN 1-Hop Cross-Portal Sensitivity:** `18.335009`

### 5. EXP-05: Grandmaster Tactical Archetype Recognition
- **Archetypes Recognized in Top-3 Candidates:** `3 / 4 (75.0%)`

### 6. EXP-06: $S_6 \times \mathbb{Z}_2$ Group Equivariance ($1,440\times$ Sample Efficiency)
- **Horizontal Reflection ($\mathbb{Z}_2$) Policy Error:** `0.00e+00`
- **Color Permutation ($S_6$) Policy Error:** `8.82e-06`
