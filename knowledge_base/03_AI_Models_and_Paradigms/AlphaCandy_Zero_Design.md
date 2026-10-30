---
tags:
  - architecture
  - search
  - gumbel-alphazero
  - expert-iteration
title: "AlphaCandy-Zero: Search-Centric Expert Iteration & Clairvoyant Oracle"
created: 2026-10-30
status: complete
aliases:
  - AlphaCandy-Zero
  - Gumbel AlphaZero Candy Crush
---

# ♟️ AlphaCandy-Zero: Search-Centric Expert Iteration & Clairvoyant Oracle

> [!IMPORTANT] Core First-Principles Thesis
> Because we possess an exact, fast simulator (`ReverseCandyEngine`), the strongest achievable Candy Crush agent is **not** a larger model-free network evaluated with 1-sample greedy lookahead, but **network-guided Monte-Carlo planning inside the true simulator**, trained via **Gumbel AlphaZero Expert Iteration** (Danihelka et al., ICLR 2022), **Prioritized Level Replay** (Jiang et al., ICML 2021), and **Go-Explore Near-Miss Endgame Restarts** (Ecoffet et al., Nature 2021).

---

## 1. Root-Cause Audit of Prior Bottlenecks

A rigorous audit of the codebase and unsolved campaign levels uncovered four concrete bottlenecks that previously prevented 100% completion:

1. **Single-Sample Lookahead Variance (`models/grandmaster_consensus_agent.py`)**:
   - Previously, each candidate swap was evaluated by cloning the board **once** (`sim_state = state.copy()`) and calling `execute_move` with a single random refill sample. One lucky or unlucky cascade during lookahead arbitrarily flipped the chosen move.
2. **Channel-19 Feature Entanglement (`models/spatial_transformer_net.py`)**:
   - The legacy 20-channel encoder set `tensor[19] = 0.5 * moves_norm + 0.5 * obj_norm`, making a state with **2 moves left & 20 jellies** indistinguishable from **20 moves left & 2 jellies**.
3. **Topological $1\times 1$ Isolated Island Cells (Level 279)**:
   - Level 279 places 3 jellies at `(8, 0)`, `(8, 4)`, and `(8, 8)` surrounded by inactive void (`0`) on all 4 orthogonal sides. Direct swaps into these cells are physically impossible; they can *only* be cleared remotely via a **Color Bomb** matching the island candy's color, a **Vertical Striped Candy** in column `4`, or a $3\times 3$ **Wrapped Candy** blast at `(7, 1)` / `(7, 7)`.
4. **King TileMap & Ingredient Physics Edge Cases (Level 272)**:
   - In King's `tileMap` format (`configs_v1.208.json`), token `'010'` denotes the **Drop-Down Ingredient Exit arrow** on row `r=8`, while `'009'` denotes **Chocolate** and `'049'` denotes a pre-placed **Color Bomb**. Separating `'010'` from `'009'` in `core/king_level_loader.py` and ensuring `state.ingredients` swaps with `state.colors` in `core/reverse_engine.py` unlocked Level 272's true two exit lanes at `(8, 2)` and `(8, 6)`.

---

## 2. System Architecture (`alphacandy_zero/`)

```mermaid
flowchart TD
    State["ReverseEngineState s_t"] --> Enc["v2 Encoder (alphacandy_zero/encoding.py)<br>24 Binary Spatial Planes + 16 Global Telemetry Scalars"]
    Enc --> Net["AlphaCandyZeroNet (alphacandy_zero/network.py)<br>S6 Color-Invariant Stem + FiLM/GlobalPool ResNet + Axial Laser Attn"]
    Net --> Priors["Bilinear Symmetric Edge Logits π_θ(a|s)<br>+ Win Logit V_θ(s) + 8-Bin Final Obj + 81-Cell Jelly Ownership"]
    Priors --> Gumbel["Gumbel Sequential Halving Root Search (alphacandy_zero/search.py)<br>Multi-Rollout Chance Averaging in ReverseCandyEngine"]
    Gumbel --> Target["Completed-Q Improved Policy Target π'(a|s)"]
    Target --> PLR["Self-Play Trainer (alphacandy_zero/train.py)<br>Prioritized Level Replay p(1-p) + Go-Explore Near-Miss Restarts"]
    Oracle["Clairvoyant Beam Oracle (alphacandy_zero/oracle.py)<br>Topological Trap Detector + Determinized Zobrist Beam Search"] --> PLR
    PLR --> Net
```

### Module Breakdown

| Module | File | Responsibility |
| :--- | :--- | :--- |
| **`encoding`** | [`alphacandy_zero/encoding.py`](file:///home/hammad/Antigravity_Workspace/Candy-Crush-AI/alphacandy_zero/encoding.py) | 24 spatial planes (colors `0..5`, specials `6..10` incl. Jelly Fish, blockers `11..14`, jellies `15..16`, ingredients `17`, active mask `18`, column spawner tops `19`, ingredient exits `20`, portals `21..22`, legal-move cells `23`) + 16 global scalars (`moves_left`, `remaining`, `pace`, mode one-hot, etc.) + exact $\mathbb{Z}_2$ horizontal flip permutation. |
| **`network`** | [`alphacandy_zero/network.py`](file:///home/hammad/Antigravity_Workspace/Candy-Crush-AI/alphacandy_zero/network.py) | `AlphaCandyZeroNet` with 4 presets (`tiny` `81.7K`, `cpu` `1.30M`, `base` `8.13M`, `large` `35.03M`). Exact $S_6$ color-permutation invariance (`1.1e-7` drift), FiLM global conditioning, KataGo global pooling, Axial row/col attention, Bilinear edge-swap policy head, and KataGo auxiliary ownership heads. |
| **`search`** | [`alphacandy_zero/search.py`](file:///home/hammad/Antigravity_Workspace/Candy-Crush-AI/alphacandy_zero/search.py) | `GumbelSearcher` implementing Gumbel-Top-$k$ without replacement + Sequential Halving across true stochastic `ReverseCandyEngine` rollouts, outputting completed-$Q$ policy improvement targets (`v_mix`). |
| **`train`** | [`alphacandy_zero/train.py`](file:///home/hammad/Antigravity_Workspace/Candy-Crush-AI/alphacandy_zero/train.py) | Multi-process self-play expert iteration with **Prioritized Level Replay (PLR)** ($p(1-p)$ learning potential) and **Go-Explore Near-Miss Restarts** from 3/6/10-moves-left snapshots. |
| **`oracle`** | [`alphacandy_zero/oracle.py`](file:///home/hammad/Antigravity_Workspace/Candy-Crush-AI/alphacandy_zero/oracle.py) | `diagnose_level` + `solve_with_oracle`: detects void objectives and isolated $1\times 1$ island jellies, runs determinized Zobrist beam search with Color Bomb island-sniping & bottom-row chocolate suppression, and exports golden expert trajectories. |
| **`evaluate`** | [`alphacandy_zero/evaluate.py`](file:///home/hammad/Antigravity_Workspace/Candy-Crush-AI/alphacandy_zero/evaluate.py) | Honest fixed-seed evaluation reporting both `mean_single_seed_win_rate` and `pass_at_k`, plus `LegacyMasterEvaluator` to bridge `alphacandy_large_net_master.pt`. |

---

## 3. Empirical Solvability Proof on Stubborn Levels `[235, 272, 279]`

Running `alphacandy_zero.oracle` + our Kaggle Residual Hunter conquered **all 3 remaining levels in Levels 1–300**, achieving **300 / 300 (100.00% Mastery on Levels 1–300)**:

- **Level 235 (`JELLY`, 162 jellies, 40 frostings, 5.06 jellies/move)**: Solved on **Attempt 1 in 22 / 32 moves** (`10 moves remaining`, score `370,633`, seed `351681`).
- **Level 272 (`INGREDIENTS`, 5 hazelnuts, 7 bottom-row chocolates)**: Solved on **Attempt 1 in 14 / 27 moves** (`13 moves remaining`, score `82,859`, seed `721940`).
- **Level 279 (`JELLY`, 21 jellies with 3 isolated $1\times 1$ island cells `[(8,0), (8,4), (8,8)]`)**: Solved both on Kaggle (`seed=354235`, `moves_used=20/20`, score `396,648`) and locally by the Clairvoyant Oracle via Color Bomb island-color sniping.

---

## 🔗 Related Notes
- Return to: **[[01_Architecture_MOC]]** | **[[00_Home_MOC]]**
- Related: **[[Forensic_Autopsy_of_Unsolved_Levels_and_History]]** | **[[Physics_Engine_Audit_and_Historical_Evolution]]**
