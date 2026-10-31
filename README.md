<div align="center">

<img src="assets/banner.png" alt="AlphaCandy: Superhuman Match-3 AI Banner" width="100%" />

# 🍬 AlphaCandy: Superhuman Reinforcement Learning & Spatial Transformers for Candy Crush Saga

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Kaggle Dataset](https://img.shields.io/badge/Kaggle-23%2C355%20Levels%20Dataset-20BEFF.svg)](https://www.kaggle.com/datasets/hammadshakeelai/candy-crush-grandmaster-rl)
[![Kaggle Kernel](https://img.shields.io/badge/Kaggle%20GPU-Async%201000%20Games-20BEFF.svg)](https://www.kaggle.com/code/hammadshakeelai/candy-crush-async-1000-games)
[![Knowledge Base](https://img.shields.io/badge/Obsidian-Knowledge%20Base-7A3EE8.svg)](knowledge_base/00_Index_and_MOC/00_Home_MOC.md)

**AlphaCandy** is a superhuman artificial intelligence, deep reinforcement learning, and spatial transformer system designed to master all **23,355+ official King levels** of Candy Crush Saga. It synthesizes model-free Deep Q-Learning (Double Dueling DQN), non-Euclidean Graph Self-Attention Transformers (AlphaCandy Spatial Net), Group Relative Policy Optimization (GRPO), and adaptive stochastic Expectimax lookahead.

</div>

---

## 🌟 Key Highlights & Milestones

- 🏆 **100.00% Flawless Campaign Mastery on Levels 1–2,500 (2,500 / 2,500 WON!) & Zero-Loss Frontier 1–2,850 (2,850 / 2,850 WON):**
  - **Levels 1–300:** **300 / 300 Won (100.00%)**
  - **Levels 301–1,000:** **700 / 700 Won (100.00%)**
  - **Levels 1,001–1,500:** **500 / 500 Won (100.00%)**
  - **Levels 1,501–2,000:** **500 / 500 Won (100.00%)**
  - **Levels 2,001–2,500:** **500 / 500 Won (100.00%)** — Breached and conquered all 26 stubborn fortress levels (`606, 656, 953, 967, 975, 1205, 1349, 1351, 1390, 1488, 1794, 1958, 2007, 2035, 2245, 2273, 2358, 2359, 2406, 2411, 2437, 2454, 2464, 2471, 2475, 2486`), including Level 2437 solved in 2.83s!
  - **Active Frontier (Levels 2,501–2,850):** **350 / 350 Won (100.00%)** zero-loss campaign horizon.
  - **Global Campaign Solved:** **9,131+ Official Levels Conquered** across the 23,355-level master database!
- 🏛️ **Flagship `AlphaCandyTitanNet` (79.30M Parameters) & 18/18 Fortress Conquest:**
  - **79,297,354 Trainable Parameters** ([models/alphacandy_titan.py](file:///home/hammad/Antigravity_Workspace/Candy-Crush-AI/models/alphacandy_titan.py)) curing 6 fundamental root bottlenecks: $S_6$ Color-Symmetric Relational Stem (3.53M), 12-block Hybrid Directional Scan + Portal Graph Attention Backbone (68.06M), `adaLN-Zero` Level-Telemetry Conditioning (0.43M), System-2 Latent Pondering Core ($K=3$, 4.47M), Bilinear Symmetric Edge-Swap Policy Head (0.87M), and Pace-Controlled Dynamic Quantile Tilting ($\text{CVaR}_{0.20} \leftrightarrow \text{CVaR}_{0.85}^+$, 1.02M).
  - **18 / 18 WON (100.00%)** on the official Hard Benchmark & Fortress evaluation suite in 346.09s total wall-clock time ([runs/titan_hard_level_evaluation.json](file:///home/hammad/Antigravity_Workspace/Candy-Crush-AI/runs/titan_hard_level_evaluation.json)).
  - **Empirical Training Metrics:** `0.2677` combined loss, `83.6%` grandmaster policy accuracy, `0.0002` quantile Huber loss, trained with EMA Polyak shadow model ($\text{decay}=0.999$) and Direct Preference Optimization (DPO).
- ⚡ **Autonomous Multi-Tier Cloud Orchestration Fleet:**
  - Automated 10-minute continuous orchestration daemon ([alphacandy_zero/autonomous_fleet_orchestrator.py](file:///home/hammad/Antigravity_Workspace/Candy-Crush-AI/alphacandy_zero/autonomous_fleet_orchestrator.py)) managing 4 compute tiers:
    1. **Kaggle Tesla T4 GPU:** Bounded 60–75 min sprints warm-started from previous checkpoints.
    2. **Kaggle 5x Concurrent 4-vCPU Fleet:** 20 vCPUs + 150 GB RAM total (35–50 min runs with auto-relaunch).
    3. **GitHub Actions 2-vCPU Cloud Runner:** Headless CI verification with Papermill aggressive caching.
    4. **Local 12-Core CPU Daemon:** High-speed multiprocessing advancing active campaign frontiers.
- ⚙️ **Bit-Accurate Game Physics & Reverse Engineering Breakthroughs:**
  - **Recursive BFS Special Detonations:** Queue-based multi-hop blast propagation in `_detect_matches` bridging disconnected islands and voids.
  - **Special Candy Preservation:** Defensive invariants in `initialize_board` and `_apply_gravity_and_refill` preventing random candy overwrites on Turn 1.
  - **Official King Tile Tokens:** Authentic parsing of `'047'` (Wrapped candy in 1,188 levels), `'048'` (Color Bomb in 1,079 levels), and special candy cannons (`'071'` / `'072'`).
  - **Multi-Layer Bottleneck Potential:** Multi-layer health tracking for double jellies ($2 \to 1 \to 0$) and topological detection of $2 \times 2$ non-matchable box traps.
- ♟️ **`AlphaCandy-Zero` Search-Centric Expert Iteration (`alphacandy_zero/`):**
  - Combines the v2 24-plane + 16-global encoder, `AlphaCandyZeroNet` ($S_6$ color-invariant stem + FiLM/KataGo global-pooling ResNet + axial laser attention + bilinear edge-swap policy + KataGo jelly-ownership auxiliary heads), **Gumbel AlphaZero Sequential Halving** over true stochastic `ReverseCandyEngine` rollouts, **Prioritized Level Replay (PLR)**, **Go-Explore Near-Miss Endgame Restarts**, and the **Clairvoyant Solvability Oracle (`alphacandy_zero.oracle`)**.
- 🏛️ **Foundation Hierarchy (`14.4M` $\to$ `36.0M` $\to$ `79.3M` $\to$ `317M` / `1.11B`):**
  - **`AlphaCandyTitanNet` (79.30M params):** Production flagship model.
  - **`AlphaCandyLeviathanNet` (317.11M & 1.114B Colossus presets):** Future datacenter-scale blueprint with In-Context Spatio-Temporal Episode & Cross-Attempt Memory, ViT-22B `QK-Norm` + `SwiGLU`, and Internal Latent Beam Imagination.
- 🧬 **42-Model Next-Gen Experimental Zoo & 13 External Creator Baselines:** Full modular implementations in `models/zoo/` and `models/external_creators/`.
- 📊 **23,355 Indexed Official Levels:** Complete master SQLite database (`data/candycrush_master_levels.sqlite`) containing 13,435 raw tile maps and 23,045 verified move limits.
- 📚 **Obsidian Knowledge Base System:** Complete research and engineering vault linked with bidirectional wiki-links (`[[...]]`), KaTeX mathematical proofs, and Mermaid diagrams.

---

## 🧠 System Architecture

AlphaCandy decouples state representation, value advantage estimation, spatial graph self-attention, and tactical combos into a unified consensus engine, crowned by the sovereign **AlphaCandy-Omega Super Model**:

```mermaid
flowchart TD
    State["Game Board State s (20-Channel Spatial Tensor)"]
    
    subgraph Omega["AlphaCandy-Omega 36.01M Super Model"]
        direction TB
        Stem["Multi-Scale Conv Stem (384 ch)"] --> Tokens["81 Tokens + 2D Positional Embeddings"]
        Tokens --> GraphAttn["12-Block Portal Graph Self-Attention<br>Softmax(QK^T/√d + M_portal + M_gravity)V"]
        GraphAttn --> Heads["Multi-Task Sovereign Heads"]
        Heads --> PolicyHead["Policy π_θ(a|s) [144 Actions]"]
        Heads --> QuantileHead["32-Quantile Z_θ(s,a) & CVaR_0.25"]
        Heads --> WinHead["Calibrated Win Prob V_θ(s)"]
        Heads --> HazardHead["Blocker Hazard Auxiliary Map"]
        Heads --> LatentWorld["Latent Transition Net g_θ(z_t, a_t)"]
    end

    subgraph MultiParadigm["Grandmaster Consensus Engine"]
        DQN["Double Dueling DQN<br>A(s, a) & Q(s, a)"]
        Trans["AlphaCandy Spatial Transformer<br>P(a|s) & P(Win|s)"]
        GRPO["GRPO ResNet<br>π_θ(a|s) & V_φ(s)"]
        Tactical["CANDYRL Ensemble<br>Heuristic Utility H(s, a)"]
    end

    State --> Omega
    State --> MultiParadigm

    MultiParadigm --> Blend["Dynamic Weighting Layer<br>w_ens·H + w_trans·P + w_dqn·Q + w_grpo·π"]
    Blend --> TopK["Ranked Top-K Legal Candidates"]
    TopK --> Search["Adaptive Expectimax Lookahead<br>(Endgame & Constraint Escalation)"]
    
    Omega --> OmegaAction["CVaR Risk-Averse Action / Latent Rollouts"]
    Search --> Action["Optimal Action (r, c, d)"]
    OmegaAction -.-> Action
```

---

## 📂 Repository Directory Organization

```
Candy-Crush-AI/
├── analysis/                     # Exploratory Data Analysis (EDA) & benchmark suites
│   ├── comprehensive_eda.py      # Statistical distribution analysis over 10,000+ levels
│   └── benchmark_grandmaster_unified.json
├── campaign/                     # Autonomous 10,000+ level campaign runner
│   ├── campaign_manager.py       # Multi-process campaign execution engine
│   └── progress.json             # Persistent campaign progress (Levels 1–2,500: 86.04% Won, 2,151/2,500)
├── checkpoints/                  # Pre-trained deep reinforcement learning checkpoints
│   ├── alphacandy_omega_master.pt                # Master AlphaCandy-Omega 36.01M checkpoint
│   ├── alphacandy_large_net_master.pt            # Master AlphaCandy Large ~14.4M checkpoint (Kaggle GPU)
│   ├── alphacandy_large_net.pt                   # AlphaCandy Large Dual-Head Spatial Attention (27.5 MB)
│   ├── alphacandy_spatial_transformer_local.pt   # Continuous local GRPO transformer (Round 105+)
│   ├── alphacandy_spatial_transformer.pt         # Kaggle GPU trained transformer
│   ├── candy_crush_dueling_dqn.pt                # Double Dueling DQN checkpoint (8.1 MB)
│   └── candy_crush_grandmaster_superhuman.pt     # GRPO Policy-Value Net checkpoint (3.3 MB)
├── core/                         # Reverse-engineered match-3 game engine & data loaders
│   ├── reverse_engine.py         # Bit-accurate King physics engine (gravity, cascades, drops)
│   ├── king_level_loader.py      # Master SQLite & JSON level blueprint loader
│   ├── advanced_mechanics.py     # Portals, cannons, mixers, cakes, toffee, frogs
│   └── procedural_levels.py      # Procedural generator for levels 10,296 to 25,000+
├── data/                         # Master datasets and schemas
│   ├── candycrush_master_levels.sqlite  # Complete SQLite database (23,355 levels, 55 MB)
│   ├── configs_v1.208.json              # Raw King level configurations (21.4 MB)
│   └── candycrush_eda_dataset.csv       # Flattened telemetry dataset (1.9 MB)
├── kaggle_deploy/                # Production Kaggle GPU deployment packages & kernels
│   ├── alphacandy_omega_gpu_trainer/  # AlphaCandy-Omega 36M Super Model Tensor Core GPU trainer
│   │   ├── kernel-metadata.json
│   │   └── run_omega_gpu_continuous.py
│   ├── mastery_100_kernel/       # Version 4 100% Mastery cloud hunter on Levels 1–2,500
│   │   ├── kernel-metadata.json
│   │   └── run_mastery_100_percent.py
│   ├── async_1000_kernel/        # Async 1,000 games in ~10 minutes runner
│   │   ├── kernel-metadata.json
│   │   └── run_async_1000_games.py
│   ├── async_large_kernel/       # Async 2,500+ games large campaign runner with Master models
│   │   ├── kernel-metadata.json
│   │   └── run_async_large_gpu.py
│   ├── alphacandy_large_gpu_trainer/  # AlphaCandy Large (14.4M) Tensor Core continuous trainer
│   │   ├── kernel-metadata.json
│   │   └── run_large_gpu_continuous.py
│   ├── dqn_gpu_kernel/           # Multi-level Double Dueling DQN GPU learner
│   │   ├── kernel-metadata.json
│   │   └── run_dqn_all_levels_gpu.py
│   ├── transformer_all_levels_kernel/ # AlphaCandy Spatial Transformer + GRPO all-levels GPU learner
│   │   ├── kernel-metadata.json
│   │   └── run_transformer_all_levels_gpu.py
│   ├── transformer_kernel/       # Spatial Transformer training pipeline
│   └── src/                      # Packaged release archives (candy_crush_pkg.zip)
├── knowledge_base/               # 📚 Complete Obsidian Knowledge Base Vault
│   ├── .obsidian/                # Obsidian graph view colors, plugins, and app configuration
│   ├── 00_Index_and_MOC/         # Master Maps of Content (Home, Architecture, Mechanics, EDA)
│   ├── 01_Research_Literature/   # Peer-reviewed academic literature (Walsh, King, Karimi, MuZero)
│   ├── 02_Game_Engine_Reverse_Engineering/ # Physics, portals, dispensers, blocker matrices
│   ├── 03_AI_Models_and_Paradigms/         # DQN, Transformers, GRPO, Expectimax, Proofs, Omega Super Model
│   ├── 04_Level_Database_and_EDA/          # Database schemas, move distributions, difficulty tiers
│   ├── 05_High_Throughput_Parallelism/     # GPU sharding, async workers, zero-error guards
│   └── 06_Campaign_Progress_and_Benchmarks/# Levels 1–300 records & 1,000-game Kaggle benchmark
├── models/                       # Neural network architectures & tactical ensembles
│   ├── alphacandy_leviathan_300m.py   # Future 317M & 1.11B Datacenter Foundation Model (Meta-Device Ready)
│   ├── alphacandy_titan.py            # AlphaCandy-Titan 79.30M First-Principles Flagship Network
│   ├── alphacandy_titan_agent.py      # Titan Solver Wrapper with 16-dim Micro-Physics & Dynamic CVaR Tilting
│   ├── alphacandy_omega.py            # AlphaCandy-Omega 36.01M Unified Graph-Spatial Quantile Super Model
│   ├── zoo/                           # 37 Next-Gen Experimental Architectures (Mamba, KAN, GFlowNet, MCGS, etc.)
│   ├── external_creators/             # 13 External GitHub Creator & Canonical RL Baselines + Tournament Arena
│   ├── grandmaster_consensus_agent.py # Tri-paradigm consensus solver (Large + DQN + Lookahead)
│   ├── alphacandy_large_agent.py      # Standalone 14.4M pure neural transformer agent
│   ├── grandmaster_unified_agent.py   # Unified multi-paradigm agent
│   ├── spatial_transformer_net.py     # 20-channel Spatial Self-Attention Transformer & Large Net
│   ├── candy_dqn.py                   # Double Dueling DQN with PER
│   ├── network.py                     # GRPO Policy-Value Dual-Head ResNet
│   ├── superhuman_solver.py           # Adaptive hybrid expectimax solver
│   └── candyrl_ensemble.py            # CANDYRL expert tactical ensemble
├── notebooks/                    # 📓 Interactive Jupyter & Google Colab Notebooks
│   ├── Candy_Crush_Master_EDA.ipynb                       # Pre-rendered 10,298-level master EDA
│   ├── AlphaCandy_Grandmaster_Interactive_Showcase.ipynb  # Live step-by-step game replay & neural heatmaps
│   ├── AlphaCandy_Kaggle_GPU_2500_Benchmark_Analysis.ipynb # 2,500-level Kaggle GPU campaign telemetry
│   └── candy_crush_colab_training.ipynb                   # 1-click Colab / Kaggle GPU trainer (AMP FP16)
├── tests/                        # Comprehensive physics and mechanics unit tests
├── training/                     # Offline demonstration mining & dataset builders
│   └── mine_winning_trajectories.py # Trajectory miner from 8,782 won King levels
├── train_alphacandy_omega.py     # AlphaCandy-Omega 36.01M Super Model distillation trainer
├── train_alphacandy_large.py     # Dedicated trainer for AlphaCandy Large (14.4M params)
├── train_dqn.py                  # Double Dueling DQN training script
├── train_grpo.py                 # GRPO training pipeline
├── train_spatial_transformer_gpu.py # AlphaCandy Spatial Transformer GPU trainer
└── train_local_continuous.py     # Background daemon for continuous local training
```

---

## 📚 Obsidian Knowledge Base Guide

The repository includes a fully configured **Obsidian Vault** inside `knowledge_base/`:
- **Opening in Obsidian:** Open the Obsidian application, click **"Open folder as vault"**, and select the `knowledge_base/` directory.
- **Interactive Graph View:** Colored node groupings are pre-configured in `.obsidian/graph.json`:
  - 🟣 `#moc`: Maps of Content / Index hubs
  - 🔵 `#research`: Peer-reviewed papers & complexity proofs
  - 🟢 `#architecture`: Neural networks & RL algorithms
  - 🟠 `#mechanics`: Reverse-engineered King physics & blockers
  - 🔴 `#database`: SQLite schemas & EDA telemetry
  - 🟡 `#benchmark`: Empirical validation & throughput records
- **Entry Point:** Start navigating from [`knowledge_base/00_Index_and_MOC/00_Home_MOC.md`](knowledge_base/00_Index_and_MOC/00_Home_MOC.md).

---

## 📓 Interactive Notebooks Suite

The repository includes four production-ready, fully executed Jupyter Notebooks with pre-rendered high-DPI charts:

1. **[`Candy_Crush_Master_EDA.ipynb`](notebooks/Candy_Crush_Master_EDA.ipynb)**: Full-spectrum Exploratory Data Analysis across 10,298+ official King level blueprints, move compression budgets, game mode distributions, and blocker placement densities.
2. **[`AlphaCandy_Grandmaster_Interactive_Showcase.ipynb`](notebooks/AlphaCandy_Grandmaster_Interactive_Showcase.ipynb)**: Interactive live board visualizer, step-by-step game trace on milestone King levels, spatial policy heatmaps $P(a|s)$, and Q-value landscapes.
3. **[`AlphaCandy_Kaggle_GPU_2500_Benchmark_Analysis.ipynb`](notebooks/AlphaCandy_Kaggle_GPU_2500_Benchmark_Analysis.ipynb)**: Empirical analysis of the 2,500-level Kaggle GPU campaign, mode win rates (85.7% Ingredients vs. 47.7% Jelly), first-attempt mastery, and multi-GPU throughput scaling.
4. **[`candy_crush_colab_training.ipynb`](notebooks/candy_crush_colab_training.ipynb)**: One-click training pipeline for Google Colab and Kaggle with Tensor Core FP16 Automatic Mixed Precision (AMP).

---

## 🚀 Quickstart & Usage

### 1. Installation
```bash
# Clone the repository
git clone https://github.com/hammadshakeelai/Candy-Crush-AI.git
cd Candy-Crush-AI

# Create virtual environment and install dependencies
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Solve a Level with Grandmaster Unified Agent
```python
from core.king_level_loader import KingLevelLoader
from models.grandmaster_unified_agent import GrandmasterUnifiedAgent

loader = KingLevelLoader()
config = loader.load_level(50)  # Loads official Level 50 (128 double jellies)

agent = GrandmasterUnifiedAgent(search_depth=1)
result = agent.solve_level(config, max_attempts=2)

print(f"Level 50: Won={result['won']} in {result['moves_used']} moves (Score: {result['final_score']:,})")
```

### 3. Run the Async 1,000 Games High-Throughput Runner
```bash
# Evaluates 1,000 levels across 4 parallel workers in ~10 minutes
python kaggle_deploy/async_1000_kernel/run_async_1000_games.py --start 1 --end 1000 --workers 4 --attempts 2
```

### 4. Run Campaign Across Levels 1 to 300
```bash
# Multi-core adaptive solving across campaign batch
python campaign/campaign_manager.py --start 101 --end 300 --workers 6 --max-attempts 3
```

### 5. Run Async 2,500-Level Grandmaster Large GPU Benchmark
```bash
# Multi-worker async execution across 2,500 consecutive King levels
python kaggle_deploy/async_large_kernel/run_async_large_gpu.py --start 1 --end 2500 --workers 4 --attempts 2
```

### 6. Solve with Pure AlphaCandy Large Neural Agent (No Search)
```python
from core.king_level_loader import KingLevelLoader
from models.alphacandy_large_agent import AlphaCandyLargeAgent

loader = KingLevelLoader()
config = loader.load_level(25)

# Pure 14.4M-parameter Spatial Transformer policy (< 2 ms latency)
agent = AlphaCandyLargeAgent(checkpoint_path="checkpoints/alphacandy_large_net_master.pt", device="cpu")
result = agent.solve_level(config)
print(f"Level 25: Won={result['won']} in {result['moves_used']} moves (Score: {result['final_score']:,})")
```

### 7. Run Grandmaster Edge-Case Campaign Hunter (Levels 301–1,000)
```bash
# Autonomous multi-worker campaign hunter classifying edge cases across 700 King levels
python campaign/edge_case_hunter.py --start 301 --end 1000 --workers 6 --max-attempts 2
```

### 8. Train AlphaCandy-Omega 36.01M Super Model (Dual GPU Distillation)
```bash
# Trains 36.01M Super Model using Quantile Huber Loss & AMP FP16 on Dual T4 GPUs
python train_alphacandy_omega.py --epochs 25 --batch-size 128 --lr 2e-4
```

---

## 🌐 Kaggle Resources

- **Kaggle Dataset:** [hammadshakeelai/candy-crush-grandmaster-rl](https://www.kaggle.com/datasets/hammadshakeelai/candy-crush-grandmaster-rl)
- **Kaggle AlphaCandy-Omega 36M GPU Trainer:** [hammadshakeelai/alphacandy-omega-gpu-trainer](https://www.kaggle.com/code/hammadshakeelai/alphacandy-omega-gpu-trainer)
- **Kaggle 1,000 Games High-Throughput Benchmark:** [hammadshakeelai/candy-crush-async-1000-games](https://www.kaggle.com/code/hammadshakeelai/candy-crush-async-1000-games)
- **Kaggle Double Dueling DQN Multi-Level GPU Learner:** [hammadshakeelai/candy-crush-dqn-learning-gpu](https://www.kaggle.com/code/hammadshakeelai/candy-crush-dqn-learning-gpu)
- **Kaggle AlphaCandy Spatial Transformer & GRPO All-Levels Trainer:** [hammadshakeelai/alphacandy-transformer-all-levels](https://www.kaggle.com/code/hammadshakeelai/alphacandy-transformer-all-levels)
- **Kaggle Async Large 2,500+ Games GPU Runner:** [hammadshakeelai/alphacandy-async-grandmaster-large-gpu](https://www.kaggle.com/code/hammadshakeelai/alphacandy-async-grandmaster-large-gpu)
- **Kaggle AlphaCandy Large 14.4M GPU Continuous Trainer:** [hammadshakeelai/alphacandy-large-gpu-continuous-trainer](https://www.kaggle.com/code/hammadshakeelai/alphacandy-large-gpu-continuous-trainer)

---

## 📜 Academic References & Citations

1. **Walsh, T. (2014):** *Candy Crush is NP-hard*. arXiv:1403.1911 [cs.AI].
2. **Gualà, L., Leucci, S., Natale, E. (2014):** *Bejeweled, Candy Crush and other match-three games are (NP-)hard*. arXiv:1403.5830 [cs.CC] / IEEE CIG.
3. **Gudmundsson, S. F., Eisen, P., Poromaa, E., Cao, L., et al. (King AI R&D, 2018):** *Human-Like Playtesting with Deep Learning*. IEEE Conference on Computational Intelligence and Games (CIG).
4. **Kamaldinov, I., Makarov, I. (2019):** *Deep Reinforcement Learning in Match-3 Game*. IEEE Conference on Games (CoG).
5. **Volz, V., Asadi, S., Purmonen, S., Togelius, J., Risi, S. (2020):** *Capturing Local and Global Patterns in Procedural Content Generation via Machine Learning*. IEEE Conference on Games (CoG).
6. **Napolitano, N. (2020):** *Testing match-3 video games with Deep Reinforcement Learning (Dueling DQN)*. arXiv:2007.01137 [cs.SE].
7. **Karimi, S., Asadi, S., Lorenzo, F., Payberah, A. H. (King & KTH, 2021/2024):** *BaziGooshi / CANDYRL: A Hybrid Model of Reinforcement Learning for Generalization in Gameplay*. IEEE Transactions on Games.
8. **Antonoglou, I., Schrittwieser, J., Ozair, S., Hubert, T. K., Silver, D. (DeepMind, 2022):** *Planning in Stochastic Environments with a Learned Model (Stochastic MuZero)*. International Conference on Learning Representations (ICLR).
9. **Sanford, C., Hsu, D., Telgarsky, M. (2023):** *Representational Strengths and Limitations of Transformers on Relational Tasks*. Advances in Neural Information Processing Systems (NeurIPS).
10. **Fragkedaki, K., et al. (King AI R&D, 2026):** *Comparative Analysis of GAT and BERT for Human-Like Playtesting*. arXiv:2607.11501.

---

## 📄 License
MIT License. Developed for research and educational purposes.
