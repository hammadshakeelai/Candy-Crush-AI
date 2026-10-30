# External GitHub Creators, Academic Literature, and RL Baselines Comparison

> **Tags**: `#LiteratureReview` `#ExternalCreators` `#Benchmarks` `#RLBaselines` `#ModelComparison`
> **Related Notes**: [[Project_History_Red_Team_Critique_and_Evolution]], [[Forensic_Autopsy_of_Unsolved_Levels_and_History]], [[AlphaCandy_Leviathan_300M_and_Colossus_1B_Future_Blueprint]]

---

## 1. Overview of the 13 External Creator & Literature Baselines

To rigorously benchmark AlphaCandy against the broader research community, open-source GitHub creators, and foundational DeepMind/OpenAI RL paradigms, we reconstructed **13 distinct external baselines** inside [`models/external_creators/`](file:///home/hammad/Antigravity_Workspace/Candy-Crush-AI/models/external_creators/__init__.py). Every model operates on our standardized `(B, 25, 9, 9)` spatial tensor and `146`-action interface so it can be evaluated head-to-head on the exact same 14,700 King levels.

| # | Model Class / Agent | Historical / Creator Origin | Paradigm | Params | Strengths & Fatal Flaws on Levels 1–2,500 |
| :-: | :--- | :--- | :--- | :---: | :--- |
| 1 | [`KingPlaytestCNN2018`](file:///home/hammad/Antigravity_Workspace/Candy-Crush-AI/models/external_creators/king_playtest_cnn_2018.py) | Gudmundsson et al. (King Digital, FDG 2018) | Supervised Imitation CNN | `~1.8M` | Fast playtesting bot; fails on complex portal & lock chains due to zero lookahead |
| 2 | [`MulliganaceousPolicyNet`](file:///home/hammad/Antigravity_Workspace/Candy-Crush-AI/models/external_creators/mulliganaceous_cnn.py) | `mulliganaceous/candy-crush-ai` (GitHub) | Lightweight Conv Policy | `~0.9M` | Good local match finder; blind to chocolate turn-lock suppression |
| 3 | [`KarimiCandyRL2020`](file:///home/hammad/Antigravity_Workspace/Candy-Crush-AI/models/external_creators/karimi_dqn_2020.py) | Karimi et al. / Open-Source Match-3 DQN | Dueling Double-DQN | `~2.4M` | Overestimates Q-values under high skyfall stochasticity |
| 4 | [`AlphaZeroVanilla2017Net`](file:///home/hammad/Antigravity_Workspace/Candy-Crush-AI/models/external_creators/alphazero_vanilla_2017.py) | Silver et al. (DeepMind, 2017) | ResNet + Scalar Value + MCTS | `~11.2M` | Scalar value $V \in [-1,1]$ collapses on low-probability miracle cascade boards |
| 5 | [`MuZeroVanilla2020Net`](file:///home/hammad/Antigravity_Workspace/Candy-Crush-AI/models/external_creators/muzero_vanilla_2020.py) | Schrittwieser et al. (DeepMind, 2020) | Implicit Latent Dynamics | `~14.8M` | Deterministic single-path latent transition blurs stochastic skyfall cascades |
| 6 | [`RainbowDQN2018Net`](file:///home/hammad/Antigravity_Workspace/Candy-Crush-AI/models/external_creators/rainbow_dqn_2018.py) | Hessel et al. (DeepMind, AAAI 2018) | C51 Distributional + NoisyNet | `~4.6M` | Fixed 51-atom support lacks action-delta cross-attention and portal graph edges |
| 7 | [`ImpalaResNetBaseline`](file:///home/hammad/Antigravity_Workspace/Candy-Crush-AI/models/external_creators/impala_resnet_baseline.py) | Espeholt et al. (DeepMind, ICML 2018) | V-Trace Actor-Critic ResNet | `~3.2M` | High throughput, shallow receptive field; fails on multi-step combo synthesis |
| 8 | [`DreamerV3DiscreteRSSMNet`](file:///home/hammad/Antigravity_Workspace/Candy-Crush-AI/models/external_creators/dreamerv3_rssm_baseline.py) | Hafner et al. (2023) | Discrete 32x32 Categorical RSSM | `~18.5M` | General world model, but lacks exact Zobrist MCGS and micro-physics priors |
| 9 | [`UCTCreatorAgent`](file:///home/hammad/Antigravity_Workspace/Candy-Crush-AI/models/external_creators/uct_creator_agent.py) | Classic Game AI / Monte Carlo UCT | Pure Rollout UCT Tree Search | `0` (CPU) | Horizon-limited by random rollout noise; misses coordinated special setups |
| 10 | [`VanillaPPOMatch3Net`](file:///home/hammad/Antigravity_Workspace/Candy-Crush-AI/models/external_creators/vanilla_ppo_match3.py) | Schulman et al. (OpenAI, 2017) | Clipped Surrogate Actor-Critic | `~2.8M` | Myopic credit assignment across 30-move horizons with sparse win/loss rewards |
| 11 | [`GeneticHeuristicCreatorBot`](file:///home/hammad/Antigravity_Workspace/Candy-Crush-AI/models/external_creators/genetic_heuristic_creator.py) | Evolutionary / CMA-ES Match-3 Bots | Evolved Linear/Rule Weights | `<0.1M` | Strong on open Jelly/Score boards; breaks down on non-Euclidean portals (`28.9%`) |
| 12 | [`GreedyImmediateScoreBot`](file:///home/hammad/Antigravity_Workspace/Candy-Crush-AI/models/external_creators/greedy_immediate_score.py) | Classic 1-Step Myopic Baseline | 1-Ply Immediate Reward Max | `0` (Rule) | Solves 100% of pure Score levels, but gets entombed by expanding chocolate |
| 13 | [`HumanPersonaBots`](file:///home/hammad/Antigravity_Workspace/Candy-Crush-AI/models/external_creators/human_persona_bots.py) | Casual / Combo-Chaser / Objective-Focused | Cognitive Behavioral Personas | `0` (Rule) | Models human playtest distributions; useful for difficulty calibration |

---

## 2. Deep-Dive Architectural Critique: Why External Baselines Fail on the 349 Hard Levels

Our forensic analysis of the **2,500-level campaign** (`2,151` won, `349` unsolved) explains why standard external architectures hit an asymptotic ceiling between **42% and 68% win rate** on Levels 1–2,500 compared to AlphaCandy's **86.04%**:

### 1. Industrial Playtesting CNNs (`KingPlaytestCNN2018`, `MulliganaceousPolicyNet`)
- **Design Goal**: Gudmundsson et al. (King, 2018) trained deep convolutional networks via behavioral cloning on millions of human playtest trajectories to predict level difficulty for game designers.
- **Why They Plateau**:
  1. **Euclidean Locality**: Standard $3\times 3$ conv stacks cannot propagate gradients across teleportation portals (`101 / 349` unsolved levels).
  2. **Move-Budget Invariance**: A static CNN applies the exact same convolutional filters whether `moves_left = 30` or `moves_left = 2`, whereas `AlphaCandyTitanNet` uses **`adaLN-Zero` Move-Budget Conditioning** to dynamically shift layer representations in the endgame.

### 2. Value-Based Model-Free RL (`KarimiCandyRL2020`, `RainbowDQN2018Net`)
- **Design Goal**: Estimate $Q(s, a)$ directly from replay buffers without tree search.
- **Why They Plateau**:
  1. **Flat Linear Action Head**: Mapping a pooled board vector $\mathbb{R}^d \to \mathbb{R}^{146}$ forces a massive linear layer to memorize the geometry of all 146 swaps independently. In contrast, `AlphaCandyTitanNet`'s **`SwapDeltaPolicyHead`** extracts exact cell pair embeddings $(u, v)$ and cross-attends over the board while injecting 16 deterministic micro-physics features.
  2. **Fixed vs. Continuous Quantiles**: While `RainbowDQN2018Net` uses C51 fixed support atoms, `AlphaCandyTitanNet` (`51` quantiles) and `AlphaCandyLeviathanNet` (`64` quantiles) use **Implicit Quantile Networks (IQN)** with dynamically sampled risk thresholds $\tau$, enabling targeted upper-tail CVaR optimization during desperation endgames.

### 3. Classic Model-Based RL (`AlphaZeroVanilla2017Net`, `MuZeroVanilla2020Net`, `DreamerV3DiscreteRSSMNet`)
- **Design Goal**: Combine neural representations with MCTS or latent world models.
- **Why They Plateau**:
  1. **Tree Redundancy vs. Zobrist MCGS**: Vanilla AlphaZero/MuZero use tree structures where commuting swaps ($A \to B$ vs. $B \to A$) expand duplicate subtrees. AlphaCandy's **Zobrist 64-bit MCGS Transposition Graph** merges identical states across branches.
  2. **Scalar Value Blindness**: Vanilla AlphaZero predicts a single scalar $V(s) \in [-1, 1]$, which averages out low-probability winning skyfall cascades on 1-move-left boards and abandons the only winning line.
  3. **Memoryless Across Attempts**: None of the 13 external baselines retain memory across failed attempts on the same level. `AlphaCandyLeviathanNet` (`310M`/`1.05B`) introduces **`CrossAttemptMemoryBank`** to explicitly avoid repeating opening traps across retries.

---

## 3. Feature-by-Feature Matrix: External Baselines vs. AlphaCandy

| Capability / Architectural Feature | `KingPlaytestCNN` (2018) | `AlphaZeroVanilla` (2017) | `MuZero` / `DreamerV3` | `RainbowDQN` (2018) | **`AlphaCandyTitanNet` (`79.30M`)** | **`AlphaCandyLeviathanNet` (`310M` / `1.05B`)** |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Non-Euclidean Portal Graph Attention** | ❌ | ❌ | ❌ | ❌ | ✅ (`NonEuclideanPortalGraphBlock`) | ✅ (`PortalGravityGraphAttention`) |
| **Move-Budget `adaLN-Zero` Conditioning** | ❌ | ❌ | ❌ | ❌ | ✅ (14 layers) | ✅ (20–24 layers + `QK-Norm`) |
| **Endpoint Swap-Delta Cross-Attention** | ❌ | ❌ | ❌ | ❌ | ✅ (+16 Micro-Physics Feats) | ✅ (+16 Micro-Physics Feats) |
| **Risk-Aware IQN Quantile Value Head** | ❌ | ❌ (Scalar) | ❌ | ⚠️ (Fixed C51) | ✅ (51-Quantile IQN CVaR) | ✅ (64-Quantile IQN CVaR) |
| **Latent Cascade World Model** | ❌ | ❌ | ⚠️ ($K=1$ path) | ❌ | ✅ ($H=3$ recurrent steps) | ✅ ($K=4\text{--}8$ Beams, $H=5\text{--}6$ steps) |
| **In-Context Episode Trajectory Memory** | ❌ | ❌ | ⚠️ (GRU state) | ❌ | ❌ | ✅ ($T=8\text{--}16$ Past Steps) |
| **Cross-Attempt Failure Memory Bank** | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ ($M=8\text{--}16$ Failed Attempts) |
| **Zobrist MCGS Graph Transposition** | ❌ | ❌ (Tree only) | ❌ (Tree only) | ❌ | ✅ (64-bit Zobrist DAG) | ✅ (64-bit Zobrist DAG) |
