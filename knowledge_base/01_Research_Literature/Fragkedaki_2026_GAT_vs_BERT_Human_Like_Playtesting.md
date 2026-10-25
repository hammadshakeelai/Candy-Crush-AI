# Comparative Analysis of Graph Attention Networks (GAT) and BERT for Human-Like Playtesting in Match-3

**Authors:** Kleio Fragkedaki et al. (King Digital Entertainment / AI R&D)  
**Publication:** arXiv:2607.11501 (2026)  
**Domain:** Deep Learning, Graph Neural Networks, Spatial Transformers, Automated Playtesting, Match-3 Puzzle Games  

---

## 1. Executive Summary & Core Motivation

In commercial Match-3 games like *Candy Crush Saga* and *Candy Crush Soda Saga*, automated level playtesting is critical for difficulty balancing and churn mitigation. Historically, King's AI systems utilized Convolutional Neural Networks (CNNs) trained on human telemetry (Gudmundsson et al., 2018). However, CNNs suffer from a fundamental architectural flaw in modern Match-3 design:
- Standard 2D convolution assumes **local, Euclidean, grid-planar spatial adjacency**.
- Modern levels introduce **teleporters (portals)**, **irregular board cuts**, **unconnected sub-boards**, and **gravity-defying diagonal dispenser streams**.
- When candy matches at `(r1, c1)` trigger falls from a disconnected teleporter entry at `(r2, c2)`, Euclidean convolution cannot model the causal dependency without stacking 10+ layers or massive feature engineering.

Fragkedaki et al. evaluate two generalized architectures that overcome Euclidean constraints:
1. **Graph Attention Networks (GAT)**: Boards are represented as topological graphs where nodes are grid cells and edges represent physical adjacency, portal links, and gravity paths.
2. **BERT / Spatial Transformers**: Boards are tokenized into 2D positional sequences with bidirectional multi-head self-attention, allowing global receptive fields in layer 1.

---

## 2. Graph Formulation vs. Spatial Tokenization

### Graph Representation (GAT Formulation)
The Match-3 board is formulated as a directed graph $\mathcal{G} = (\mathcal{V}, \mathcal{E})$:
- **Vertices $\mathcal{V}$**: Each playable tile $(i, j)$ is a node with feature vector $x_i \in \mathbb{R}^d$ encoding candy color, special candy status (striped, wrapped, color bomb), blocker layer (jelly, frosting, licorice), and active objectives.
- **Edges $\mathcal{E}$**: Formed by three distinct relations:
  $$\mathcal{E} = \mathcal{E}_{\text{swap}} \cup \mathcal{E}_{\text{gravity}} \cup \mathcal{E}_{\text{portal}}$$
  - $\mathcal{E}_{\text{swap}}$: Undirected edges between orthogonally adjacent cells eligible for player swaps.
  - $\mathcal{E}_{\text{gravity}}$: Directed edges representing vertical fall paths ($u \to v$ if candy in $u$ drops into $v$).
  - $\mathcal{E}_{\text{portal}}$: Directed non-local edges connecting portal exits to portal entries.

The attention coefficient between nodes $i$ and $j$ in attention head $k$ is computed as:
$$\alpha_{ij}^k = \frac{\exp\left(\text{LeakyReLU}\left(\mathbf{a}^T [\mathbf{W} x_i \,\|\, \mathbf{W} x_j]\right)\right)}{\sum_{l \in \mathcal{N}_i} \exp\left(\text{LeakyReLU}\left(\mathbf{a}^T [\mathbf{W} x_i \,\|\, \mathbf{W} x_l]\right)\right)}$$

### Spatial Transformer (BERT Formulation)
In the Transformer formulation:
- Board cells are flattened into a token sequence $T = (t_1, t_2, \dots, t_{H \times W})$.
- Each token receives a learned 2D positional embedding:
  $$E_i = \text{Embed}_{\text{tile}}(x_i) + \text{PE}_{2D}(r_i, c_i)$$
- Multi-head self-attention computes dense pairwise interactions across all $H \times W$ tokens simultaneously:
  $$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{Q K^T}{\sqrt{d_k}}\right) V$$

---

## 3. Key Findings & Empirical Comparisons

| Architecture | Inductive Bias | Portal / Non-Local Handling | Inference Latency | Human Move Alignment (Top-1 Acc) |
|---|---|---|---|---|
| **Standard CNN (Baseline)** | Strict 2D locality | Poor (requires artificial channels) | **0.8 ms** | 48.2% |
| **GAT (Graph Attention)** | Explicit relational topology | Excellent (exact portal edges) | 4.6 ms | 61.7% |
| **BERT / Spatial Transformer** | Dense global attention | Superior (learns non-local dependencies) | 2.1 ms | **64.3%** |

### Critical Takeaways:
1. **Global Receptive Field is Essential**: The Spatial Transformer achieves the highest top-1 move agreement with human grandmasters (64.3%), because cascading cascades and color bomb setups depend on whole-board color parity rather than just local neighborhoods.
2. **Computational Advantage of Transformers**: On modern GPUs, dense tensor matrix multiplications in Transformer self-attention run significantly faster than sparse message-passing operations in GATs (2.1 ms vs. 4.6 ms per board state).
3. **Robustness to Unseen Mechanics**: When evaluated on newly released level mechanics not seen during training, Transformer attention maps smoothly shifted focus to high-entropy regions, whereas CNNs collapsed.

---

## 4. Integration into AlphaCandy Architecture

In the AlphaCandy engine:
- We chose the **Spatial Transformer (AlphaCandy Large / AlphaCandyLargeNet)** with 2-stage 8-head self-attention and 2D learned positional embeddings, perfectly echoing the superior performance and GPU parallel efficiency demonstrated by King's research.
- For portal-heavy levels, AlphaCandy's self-attention layers naturally route attention weights between disconnected portal coordinates without requiring explicit graph adjacency re-indexing.

---

## 5. References & Cross-Links
- **Related Papers**:
  - [[King_2018_Human_Like_Playtesting_CNN|King 2018: Human-Like Playtesting with Deep Learning]]
  - [[Sanford_2023_Higher_Order_Attention_Match3_Complexity|Sanford et al. 2023: Higher-Order Attention in Match-3 Complexity]]
  - [[Karimi_2021_CANDYRL_BaziGooshi_Hybrid_RL|Karimi et al. 2021/2024: CANDYRL & BaziGooshi]]
- **Engine Systems**:
  - [[Pure_AlphaCandy_Large_Agent|Pure AlphaCandy Large Model]]
  - [[Grandmaster_Consensus_Architecture|Grandmaster Consensus Architecture]]
