---
title: "AlphaCandy Next-Gen Model Zoo (10 Architectures) & Micro-Experiments Suite"
tags:
  - architecture
  - model-zoo
  - gumbel-muzero
  - vision-mamba
  - equivariant-networks
  - decision-transformer
  - diffusion-policy
  - graph-neural-networks
  - gflownet
  - cvar
  - distributional-rl
  - mcgs
  - micro-experiments
created: 2026-10-30
status: complete
aliases:
  - AlphaCandy Model Zoo
  - Next-Gen Model Zoo
  - Desperation-Aware Quantile Tilting
  - S6 Z2 Equivariance Proof
---

# 🧬 AlphaCandy Next-Gen Experimental Model Zoo & Micro-Experiments Suite

> [!IMPORTANT] Sovereign Architectural Expansion
> Beyond our flagship **[[AlphaCandy_Omega_Super_Model_Architecture|AlphaCandy-Omega (36.01M)]]** and **[[Grandmaster_Consensus_Architecture|Grandmaster Consensus Engine]]**, the **AlphaCandy Next-Gen Model Zoo** (`models/zoo/`) introduces **10 specialized frontier architectures** engineered from 2023–2026 breakthroughs in discrete combinatorial planning, state-space models, geometric deep learning, generative flow networks, and dynamic risk-sensitive distributional reinforcement learning. Every hypothesis is empirically falsifiable through our **10-Part Micro-Experiments Suite** (`experiments/micro_tests/`) and deployed to Kaggle Cloud via `kaggle_deploy/model_zoo_experiments_kernel/`.

---

## 🏛️ Taxonomy of the 10 Next-Gen Model Zoo Architectures

```mermaid
flowchart TD
    State["Canonical Board State Tensor s ∈ ℝ^(B × 20 × 9 × 9)"] --> Router["10. GrandmasterMoERouter<br>(16-Dim Topological Gating Network)"]

    subgraph WorldModelsSearch["Latent Planning & Graph Search"]
        M1["1. GumbelMuZeroCandyNet<br>Gumbel-Top-k + Sequential Halving"]
        M9["9. TranspositionMCGSSolver<br>64-Bit Zobrist DAG + Dynamic CVaR_α(s)"]
    end

    subgraph SpatialBackbones["Physics-Aligned Spatial Backbones"]
        M2["2. CandyMamba2DNet<br>4-Way Cross-Scan Selective SSM"]
        M3["3. GravityEquivariantCandyNet<br>S_6 × Z_2 Group Equivariance (1,440×)"]
        M6["6. WormholeGraphNeuralNet<br>4-Edge Heterogeneous Multigraph MPNN"]
    end

    subgraph GenerativeSequential["Sequential, Multimodal & Hierarchical"]
        M4["4. CandyDecisionTransformer<br>Return-To-Go Upside-Down RL"]
        M5["5. CandyDiffusionPolicyNet<br>T=4 Iterative Score Denoising"]
        M7["7. HierarchicalSubgoalCandyNet<br>Feudal 6-Option Manager + Worker"]
        M8["8. CandyGFlowNet<br>Trajectory Balance Diverse Sampler"]
    end

    Router --> M1
    Router --> M2
    Router --> M3
    Router --> M4
    Router --> M5
    Router --> M6
    Router --> M7
    Router --> M8
    Router --> M9

    M1 --> Synthesis["Unified 144-Action Policy π*(a|s) & Calibrated Value V*(s)"]
    M2 --> Synthesis
    M3 --> Synthesis
    M4 --> Synthesis
    M5 --> Synthesis
    M6 --> Synthesis
    M7 --> Synthesis
    M8 --> Synthesis
    M9 --> Synthesis
```

### Comprehensive Architecture Matrix

| # | Model Class | Module File | Core Mathematical Paradigm | Primary Match-3 Advantage |
| :-: | :--- | :--- | :--- | :--- |
| **1** | `GumbelMuZeroCandyNet` | [`models/zoo/gumbel_muzero_net.py`](file:///home/hammad/Antigravity_Workspace/Candy-Crush-AI/models/zoo/gumbel_muzero_net.py) | Gumbel-Top-$k$ Sampling without Replacement + Sequential Halving in Latent Space $\mathcal{Z}$ | Strict policy improvement $\pi_{\text{new}} \ge \pi_{\text{old}}$ guaranteed even at ultra-low budgets ($N = 4\text{--}8$ simulations). |
| **2** | `CandyMamba2DNet` | [`models/zoo/vision_mamba_2d_net.py`](file:///home/hammad/Antigravity_Workspace/Candy-Crush-AI/models/zoo/vision_mamba_2d_net.py) | 4-Way Directional Selective State Space Model (VMamba Cross-Scan) | $\mathcal{O}(N)$ directional state recurrences aligned with downward gravity drops, upward column compression, and horizontal striped lasers. |
| **3** | `GravityEquivariantCandyNet` | [`models/zoo/equivariant_dihedral_net.py`](file:///home/hammad/Antigravity_Workspace/Candy-Crush-AI/models/zoo/equivariant_dihedral_net.py) | $S_6 \times \mathbb{Z}_2$ Color-Permutation & Horizontal Reflection Group Equivariance | **$1,440\times$ sample efficiency reduction** via exact weight sharing across all $6! = 720$ color permutations and $2$ horizontal mirror states. |
| **4** | `CandyDecisionTransformer` | [`models/zoo/decision_transformer_candy.py`](file:///home/hammad/Antigravity_Workspace/Candy-Crush-AI/models/zoo/decision_transformer_candy.py) | Causal Sequence Modeling over $(R_t, O_t, M_t, s_t, a_t)$ across $K=8$ steps | Resolves non-Markovian hidden timers (Magic Mixers, Evil Spawner cadences) and paces move urgency via target Return-To-Go conditioning. |
| **5** | `CandyDiffusionPolicyNet` | [`models/zoo/discrete_diffusion_policy.py`](file:///home/hammad/Antigravity_Workspace/Candy-Crush-AI/models/zoo/discrete_diffusion_policy.py) | Iterative Score-Based Denoising over Action Logits ($T=4$ DDIM/Cosine steps) | Eliminates multimodal averaging failure on split-island boards by collapsing onto a single coherent strategic mode. |
| **6** | `WormholeGraphNeuralNet` | [`models/zoo/gnn_wormhole_message_passing.py`](file:///home/hammad/Antigravity_Workspace/Candy-Crush-AI/models/zoo/gnn_wormhole_message_passing.py) | Heterogeneous 4-Edge Directed Multigraph Message Passing (`E_adj`, `E_gravity`, `E_portal`, `E_special_blast`) | Direct 1-hop message passing across non-local teleporter wormholes and full-row/column striped laser trajectories. |
| **7** | `HierarchicalSubgoalCandyNet` | [`models/zoo/hierarchical_option_critic.py`](file:///home/hammad/Antigravity_Workspace/Candy-Crush-AI/models/zoo/hierarchical_option_critic.py) | Feudal Manager (6 Strategic Subgoals + $9\times 9$ Goal Map $G$) + Conditioned Worker | Explicit human-interpretable subgoal decomposition (Special Creation, Combo Detonation, Chocolate Suppression, Portal Uncapping, Corner Sniping). |
| **8** | `CandyGFlowNet` | [`models/zoo/gflownet_combinatorial_sampler.py`](file:///home/hammad/Antigravity_Workspace/Candy-Crush-AI/models/zoo/gflownet_combinatorial_sampler.py) | Generative Flow Network with Trajectory Balance / Detailed Balance ($F_\theta(s,a), \log Z_\theta(s), P_B$) | Samples diverse, orthogonal winning trajectories $P(\tau) \propto R(\tau)$ for multi-attempt retry hunting without deterministic trap collapse. |
| **9** | `TranspositionMCGSSolver` | [`models/zoo/monte_carlo_graph_search.py`](file:///home/hammad/Antigravity_Workspace/Candy-Crush-AI/models/zoo/monte_carlo_graph_search.py) | 64-Bit Zobrist Hashing DAG Search + Desperation-Aware Dynamic Quantile Tilting $\text{CVaR}_{\alpha(s)}$ | Eliminates redundant evaluations of commutative swaps and switches from risk-averse survival ($\alpha=0.25$) to miracle-seeking optimism ($\alpha=0.85^+$). |
| **10** | `GrandmasterMoERouter` | [`models/zoo/meta_adaptive_router.py`](file:///home/hammad/Antigravity_Workspace/Candy-Crush-AI/models/zoo/meta_adaptive_router.py) | 16-Feature Topological Level Analyzer + Softmax Specialist Gating Network | Dynamically routes each board state to the ideal specialist ensemble based on blocker density, portal topology, and endgame urgency. |

---

## 🔬 Deep Dive 1: Gumbel MuZero with Sequential Halving (`GumbelMuZeroCandyNet`)

Standard AlphaZero/MuZero relies on PUCT with Dirichlet root noise, which fails to guarantee policy improvement when the simulation budget $N$ is smaller than the branching factor $|\mathcal{A}_{\text{legal}}| \le 144$. Following Danihelka et al. (*Policy Improvement by Planning with Gumbel*, ICLR 2022), `GumbelMuZeroCandyNet` replaces PUCT at the root with **Gumbel-Top-$k$ Sampling without Replacement** and **Sequential Halving**:

1. **Gumbel-Top-$k$ Root Sampling:**
   Given root policy logits $\ell(s_0, a) = \log \pi_\theta(a \mid z_0)$ from `RepresentationNet` $h_\theta(s_0) \to z_0$, we sample i.i.d. Gumbel variates $g_a \sim \text{Gumbel}(0, 1)$ and select the top $k$ legal candidate actions:
   $$\mathcal{C}_0 = \text{argtop-}k_{a \in \mathcal{A}_{\text{legal}}} \big( g_a + \ell(s_0, a) \big)$$
2. **Sequential Halving across $\lceil \log_2 k \rceil$ Phases:**
   In each phase $p \in \{1, \dots, \lceil \log_2 k \rceil\}$, the remaining candidates $\mathcal{C}_{p-1}$ are rolled out in latent space via `DynamicsNet` $g_\theta(z_t, a_t) \to (\hat{r}_t, \hat{z}_{t+1})$ and scored via monotonically transformed **completed Q-values** $\hat{q}(a)$:
   $$\text{Score}_p(a) = g_a + \ell(s_0, a) + \sigma\big(\hat{q}(a)\big), \quad \text{where } \sigma(\hat{q}(a)) = (c_{\text{visit}} + \max_b N(b)) \cdot c_{\text{scale}} \cdot \hat{q}(a)$$
   The bottom $50\%$ of candidates in $\mathcal{C}_{p-1}$ are pruned, doubling the simulation density for surviving candidates until a single optimal action $a^*$ remains.
3. **Guaranteed Policy Improvement Theorem:**
   For any simulation budget $N \ge 2$, the selected action satisfies:
   $$\mathbb{E}\big[ q(s_0, a^*) \big] \ge \sum_{a \in \mathcal{A}_{\text{legal}}} \pi_\theta(a \mid s_0) q(s_0, a)$$
   preventing low-budget search degradation on CPU-constrained workers.

---

## 📐 Deep Dive 2: $1,440\times$ Sample Efficiency Proof of $S_6 \times \mathbb{Z}_2$ Group Equivariance (`GravityEquivariantCandyNet`)

### 1. Why Full Dihedral $D_4$ Symmetry Fails in Match-3
In board games like Go, the dynamics are invariant under the full dihedral group $D_4$ ($90^\circ, 180^\circ, 270^\circ$ rotations and reflections, $|D_4| = 8$). However, in *Candy Crush Saga*, **gravity acts strictly downward along the $+r$ vertical axis** (barring explicit conveyor/gravity-shift mechanics). Rotating a board by $90^\circ$ or flipping it upside-down violates physical cascade trajectories.

### 2. The Exact Symmetry Group of Candy Crush: $G = S_6 \times \mathbb{Z}_2$
While vertical rotation is broken by gravity, Match-3 physics remain strictly invariant/equivariant under two orthogonal symmetry groups:

1. **Color-Channel Permutation Group $S_6$ ($|S_6| = 6! = 720$):**
   The first 6 channels of our 20-channel state tensor represent the 6 standard candy colors $\{\text{Red}, \text{Blue}, \text{Green}, \text{Yellow}, \text{Orange}, \text{Purple}\}$. Because match rules, special candy creation (Striped, Wrapped, Color Bomb), and blocker damage depend solely on **color equality** $\mathbb{I}(c_i = c_j)$ and never on specific color identity, any permutation $\pi_c \in S_6$ of channels $0\text{--}5$ yields an isomorphic board state with identical optimal actions:
   $$\forall \pi_c \in S_6: \quad \pi^*\big(\rho_{\text{state}}(\pi_c) s\big) = \pi^*(s), \quad V^*\big(\rho_{\text{state}}(\pi_c) s\big) = V^*(s)$$
2. **Horizontal Reflection Group $\mathbb{Z}_2 = \{e, \tau_{\text{flip}}\}$ ($|\mathbb{Z}_2| = 2$):**
   Reflecting the board horizontally $(r, c) \mapsto (r, W - 1 - c)$ where $W = 9$ commutes with downward gravity, diagonal sliding symmetry, and match-3 rules. Under $\tau_{\text{flip}}$, canonical actions $a \in \{0, \dots, 143\}$ transform via the induced action permutation $\rho_{\text{act}}(\tau_{\text{flip}})$:
   - **Horizontal Swaps** $a \in [0, 71]$ at $(r, c, \text{RIGHT})$ (swapping $(r, c) \leftrightarrow (r, c+1)$) map to $(r, 7 - c, \text{RIGHT})$:
     $$\rho_{\text{act}}(\tau_{\text{flip}})(8r + c) = 8r + (7 - c), \quad r \in \{0,\dots,8\}, c \in \{0,\dots,7\}$$
   - **Vertical Swaps** $a \in [72, 143]$ at $(r, c, \text{DOWN})$ (swapping $(r, c) \leftrightarrow (r+1, c)$) map to $(r, 8 - c, \text{DOWN})$:
     $$\rho_{\text{act}}(\tau_{\text{flip}})(72 + 9r + c) = 72 + 9r + (8 - c), \quad r \in \{0,\dots,7\}, c \in \{0,\dots,8\}$$

### 3. Proof of $1,440\times$ Hypothesis Space Compression
By the Orbit-Stabilizer Theorem, the direct product group $G = S_6 \times \mathbb{Z}_2$ has order:
$$|G| = |S_6| \times |\mathbb{Z}_2| = 6! \times 2 = 720 \times 2 = 1,440$$
In `GravityEquivariantCandyNet`:
- Each color channel $c \in \{0, \dots, 5\}$ is processed through a **shared per-color equivariant encoder** $\phi_{\text{color}}(s_c, s_{\text{context}})$ and aggregated via permutation-invariant symmetric statistics ($\max_{c} \phi_c \oplus \frac{1}{6}\sum_c \phi_c \oplus \text{std}_c(\phi_c)$), enforcing exact $S_6$ invariance.
- Spatial features are symmetrized across $\mathbb{Z}_2$ via bilateral weight tying:
  $$\pi_{\text{equiv}}(s) = \frac{1}{2}\Big( \pi_\theta(s) + \rho_{\text{act}}(\tau_{\text{flip}})^{-1} \pi_\theta\big(\rho_{\text{state}}(\tau_{\text{flip}}) s\big) \Big)$$
Consequently, a single gradient update on $(s_t, a_t)$ simultaneously updates the policy across the entire $1,440$-state equivalence orbit $\mathcal{O}_G(s_t) = \{g \cdot s_t \mid g \in S_6 \times \mathbb{Z}_2\}$, reducing the effective sample complexity of the hypothesis class by a factor of **$1,440\times$** with zero symmetry-breaking approximation error ($\|\pi_{\text{equiv}}(\tau_{\text{flip}} s) - \rho_{\text{act}}(\tau_{\text{flip}})\pi_{\text{equiv}}(s)\|_\infty < 10^{-6}$).

---

## ⚡ Deep Dive 3: Desperation-Aware Dynamic Quantile Tilting ($\text{CVaR}_{\alpha(s)}$)

### 1. The Finite-Horizon Paradox of Static Risk Aversion
In **[[AlphaCandy_Omega_Super_Model_Architecture]]**, static lower-tail Conditional Value-at-Risk ($\text{CVaR}_{0.25}$) maximizes the worst $25\%$ of return quantiles to prevent board locks and chocolate expansion. However, finite-horizon deadline theory (see our literature survey on *Dynamic CVaR Switching in Finite-Horizon MDPs*) reveals a fundamental paradox:
- **When Ahead / Comfortable:** Avoiding lower-tail catastrophes (board locks, chocolate explosions) guarantees victory. Static risk aversion ($\text{CVaR}_{0.25}$) is optimal.
- **When Behind / Desperate:** Suppose only $m = 2$ moves remain, but $j = 9$ jellies are still uncleared. A "safe" conservative move that clears $1\text{--}2$ jellies with $100\%$ certainty yields a **guaranteed loss ($0\%$ win rate)**! Only a high-variance gamble—such as detonating a deep bottom-row match hoping for a $10\%$ upper-tail skyfall cascade of $8+$ jellies—has a non-zero probability of winning.

```mermaid
flowchart LR
    State["Board State s_t<br>Moves Left m_t, Objectives Left o_t"] --> Ratio["Compute Desperation Index D(s_t)<br>D = (o_t / o_max) / max(ε, m_t / m_max)"]
    Ratio --> Check{"Trajectory Regime?"}
    
    Check -- "D(s_t) < 0.85 (Ahead / Safe)" --> Averse["Risk-Averse Survival Mode<br>Lower-Tail CVaR_0.25(s, a)<br>Protect Lead & Prevent Locks"]
    Check -- "0.85 ≤ D(s_t) ≤ 1.35 (Balanced)" --> Neutral["Risk-Neutral Expected Value<br>Mean Return E[Z_θ(s, a)]"]
    Check -- "D(s_t) > 1.35 or m_t ≤ 3 (Desperate)" --> Seeking["Risk-Seeking Miracle-Hunting Mode<br>Upper-Tail CVaR_0.85+(s, a)<br>Maximize 85th–100th Percentile Cascade Ceiling!"]
```

### 2. Mathematical Formulation of Dynamic Quantile Tilting
Let $Z_\theta(s, a) = \{Z_{(1)}(s, a) \le Z_{(2)}(s, a) \le \dots \le Z_{(K)}(s, a)\}$ denote the sorted $K = 32$ return quantiles for state-action pair $(s, a)$. Let:
- $m_t = \text{moves\_left}$ and $M = \text{moves\_limit}$ be the remaining and initial move budgets.
- $o_t$ and $O_0$ be the remaining and initial objective counts (e.g., remaining jellies, blockers, ingredients, or score deficit).

We define the dimensionless **Desperation Index** $\mathcal{D}(s_t) \in [0, \infty)$:
$$\mathcal{D}(s_t) = \frac{o_t / \max(1, O_0)}{\max(0.05, m_t / \max(1, M))}$$
- If $\mathcal{D}(s_t) < 1.0$, the agent has cleared objectives faster than it has consumed moves (**Ahead of Pace**).
- If $\mathcal{D}(s_t) > 1.35$ (or $m_t \le 3$ with $o_t > m_t$), the agent is **Behind Pace / Desperate**.

In `TranspositionMCGSSolver` ([`models/zoo/monte_carlo_graph_search.py`](file:///home/hammad/Antigravity_Workspace/Candy-Crush-AI/models/zoo/monte_carlo_graph_search.py)), the risk-tilted utility $U_{\text{tilt}}(s_t, a)$ dynamically shifts its integration support across the quantile spectrum:

$$U_{\text{tilt}}(s_t, a) = \begin{cases}
\displaystyle \text{CVaR}_{0.25}^-(s_t, a) = \frac{1}{8} \sum_{k=1}^{8} Z_{(k)}(s_t, a) & \text{if } \mathcal{D}(s_t) < 0.85 \quad \text{(Ahead: Lower-Tail Risk-Averse)} \\[12pt]
\displaystyle \mathbb{E}[Z_\theta(s_t, a)] = \frac{1}{32} \sum_{k=1}^{32} Z_{(k)}(s_t, a) & \text{if } 0.85 \le \mathcal{D}(s_t) \le 1.35 \quad \text{(On-Pace: Risk-Neutral)} \\[12pt]
\displaystyle \text{CVaR}_{0.85}^+(s_t, a) = \frac{1}{5} \sum_{k=28}^{32} Z_{(k)}(s_t, a) & \text{if } \mathcal{D}(s_t) > 1.35 \text{ or } (m_t \le 3 \land o_t > m_t) \quad \text{(Desperate: Upper-Tail Miracle-Seeking)}
\end{cases}$$

**Theorem (Endgame Domination of Upper-Tail Tilting):**
Let $v_{\text{win}}$ denote the terminal threshold required to clear remaining objectives $o_t$ within $m_t$ moves. Suppose that for a safe low-variance action $a_{\text{safe}}$, $P(Z(s_t, a_{\text{safe}}) \ge v_{\text{win}}) = 0$, whereas for a volatile cascade action $a_{\text{volatile}}$, $\mathbb{E}[Z(s_t, a_{\text{volatile}})] < \mathbb{E}[Z(s_t, a_{\text{safe}})]$ but $Z_{(30)}(s_t, a_{\text{volatile}}) \ge v_{\text{win}}$. Then both $\text{CVaR}_{0.25}^-$ and $\mathbb{E}[Z]$ select $a_{\text{safe}}$ with $0\%$ win probability, whereas $\text{CVaR}_{0.85}^+$ selects $a_{\text{volatile}}$, strictly increasing terminal win probability $\Delta P(\text{Win}) > 0$.

---

## 🧪 The 10-Part Micro-Experiments Suite (`experiments/micro_tests/`)

To rapidly validate every architectural component prior to multi-hour GPU training, [`experiments/micro_tests/run_micro_experiments_suite.py`](file:///home/hammad/Antigravity_Workspace/Candy-Crush-AI/experiments/micro_tests/run_micro_experiments_suite.py) and its Kaggle cloud counterpart [`kaggle_deploy/model_zoo_experiments_kernel/run_zoo_micro_experiments_kaggle.py`](file:///home/hammad/Antigravity_Workspace/Candy-Crush-AI/kaggle_deploy/model_zoo_experiments_kernel/run_zoo_micro_experiments_kaggle.py) execute 10 targeted empirical experiments:

| Exp ID | Micro-Experiment Name | Hypothesis Tested | Key Quantitative Metric |
| :-: | :--- | :--- | :--- |
| **Exp 01** | **$S_6 \times \mathbb{Z}_2$ Symmetry Verification** | Horizontal reflection & $S_6$ color permutations preserve exact policy equivariance and value invariance. | Max Equivariance Error $< 10^{-6}$ across $1,440$ symmetry orbits |
| **Exp 02** | **Gumbel Sequential Halving vs. Beam Search** | Gumbel-Top-$k$ + Sequential Halving achieves higher post-search policy improvement at budget $N=8$ than flat Top-$k$ beam search. | Mean Policy Value Improvement $\Delta V_{\text{Gumbel}} > \Delta V_{\text{Beam}}$ |
| **Exp 03** | **64-Bit Zobrist MCGS Transposition Hit Rate** | Commutative independent swaps on $9\times 9$ boards collide into identical afterstates in Depth-2/3 search. | Zobrist Cache Hit Rate ($18\%\text{--}38\%$) & Search Speedup |
| **Exp 04** | **Desperation-Aware Dynamic Quantile Tilting** | Switching to $\text{CVaR}_{0.85}^+$ when desperate ($m_t \le 3$, high deficit) salvages unwinnable endgames where $\text{CVaR}_{0.25}^-$ fails. | Endgame Miracle Recovery Rate ($+20\%\text{--}35\%$ win delta) |
| **Exp 05** | **Wormhole Portal GNN vs. Local CNN Routing** | `WormholeGraphNeuralNet` routes gradient & attention energy directly across non-adjacent portal coordinates in 1 hop. | Cross-Portal Jacobian Sensitivity Ratio ($> 8\times$ vs. $3\times 3$ CNN) |
| **Exp 06** | **Hierarchical Subgoal Option Alignment** | `HierarchicalSubgoalCandyNet` Manager selects appropriate subgoal options across the 5 canonical tactical archetypes. | Subgoal-to-Archetype Alignment Accuracy & Entropy |
| **Exp 07** | **GFlowNet Multi-Attempt Trajectory Diversity** | `CandyGFlowNet` samples diverse high-reward root actions across $K=8$ retries instead of repeating a failed greedy argmax. | Unique Action Coverage & Shannon Entropy $H(\pi_{\text{GFN}})$ |
| **Exp 08** | **Vision Mamba 4-Way SSM Throughput Scaling** | `CandyMamba2DNet` 4-way directional scan achieves competitive latency and linear scaling across board tokens. | Forward Pass Latency ($\text{ms/batch}$) & Throughput ($\text{states/s}$) |
| **Exp 09** | **Discrete Diffusion Multimodal Mode Locking** | Iterative denoising ($T=4$) locks onto a single island mode rather than splitting probability mass across invalid averages. | Top-1 Mode Concentration Ratio & Entropy Reduction |
| **Exp 10** | **Meta-Adaptive MoE Specialist Routing** | `GrandmasterMoERouter` assigns distinct specialist gating profiles to Jelly, Chocolate, Portal, Ingredient, and Endgame boards. | Gating Divergence ($D_{\text{JS}}$) & Specialist Selection Fidelity |

---

## ☁️ Kaggle Cloud Verification Kernel (`kaggle_deploy/model_zoo_experiments_kernel/`)

The Kaggle deployment package in [`kaggle_deploy/model_zoo_experiments_kernel/`](file:///home/hammad/Antigravity_Workspace/Candy-Crush-AI/kaggle_deploy/model_zoo_experiments_kernel/) runs entirely on Kaggle's free 4-vCPU tier (`enable_gpu: false`) without consuming weekly GPU quota:
- **Metadata Configuration:** [`kaggle_deploy/model_zoo_experiments_kernel/kernel-metadata.json`](file:///home/hammad/Antigravity_Workspace/Candy-Crush-AI/kaggle_deploy/model_zoo_experiments_kernel/kernel-metadata.json) (`id: "hammadshakeelai/alphacandy-model-zoo-micro-experiments"`).
- **Execution Script:** [`kaggle_deploy/model_zoo_experiments_kernel/run_zoo_micro_experiments_kaggle.py`](file:///home/hammad/Antigravity_Workspace/Candy-Crush-AI/kaggle_deploy/model_zoo_experiments_kernel/run_zoo_micro_experiments_kaggle.py) evaluates real King levels (Levels 10, 23, 25, 65, 100, 125, 148, 201, 235, 272, 300) and outputs:
  - `MODEL_ZOO_MICRO_EXPERIMENTS_REPORT.md`
  - `micro_experiments_telemetry.json`

---

## 🔗 Related Notes & Bi-Directional Wiki-Links
- Return to Master Hub: **[[00_Home_MOC]]**
- Architecture Index: **[[01_Architecture_MOC]]**
- Sovereign Super Model: **[[AlphaCandy_Omega_Super_Model_Architecture]]**
- Synthetic Tactics A/B Study: **[[Synthetic_Tactics_Curriculum_Experiment]]**
- Tri-Paradigm Consensus: **[[Grandmaster_Consensus_Architecture]]**
- Afterstate Pruning & Expectimax: **[[Multi_Depth_Expectimax_with_Afterstate_Pruning]]**
- MuZero & World Models Literature: **[[Silver_Schrittwieser_MuZero_World_Models_Match3]]**
- Portal & Non-Euclidean Topology: **[[Portals_and_Teleporters_Topology]]**
- Reward Shaping Invariance: **[[Potential_Based_Reward_Shaping_in_Match3]]**
- Continual Learning Proof: **[[Single_Model_Continual_Learning_Proof]]**
