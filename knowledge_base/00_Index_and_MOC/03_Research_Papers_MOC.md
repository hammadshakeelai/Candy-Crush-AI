---
tags:
  - moc
  - research
title: "Academic Research & Literature Map of Content"
created: 2026-10-25
status: complete
aliases:
  - Research Papers MOC
  - Literature MOC
---

# 📚 Academic Research & Literature Map of Content

> [!NOTE] Literature Anchors
> Candy Crush and Match-3 puzzles have been analyzed extensively across computational complexity theory, telemetry-driven deep learning, and hybrid reinforcement learning.

```mermaid
graph LR
    subgraph Theory["Computational Complexity"]
        W1["Toby Walsh (2014)<br>[[Walsh_2014_Candy_Crush_NP_Hard]]"]
        D1["Erik Demaine et al.<br>[[Demaine_Combinatorial_Game_Complexity]]"]
    end

    subgraph Industry["King AI & Playtesting"]
        G1["Gudmundsson et al. (2018)<br>[[King_2018_Human_Like_Playtesting_CNN]]"]
        K1["Karimi et al. (2021)<br>[[Karimi_2021_CANDYRL_BaziGooshi_Hybrid_RL]]"]
    end

    subgraph DeepRL["Modern RL & World Models"]
        M1["Silver & Schrittwieser<br>[[Silver_Schrittwieser_MuZero_World_Models_Match3]]"]
        T1["Spatial Self-Attention<br>[[Transformers_and_Spatial_Attention_in_Grid_Worlds]]"]
    end

    W1 --> G1
    G1 --> K1
    K1 --> M1
    M1 --> T1
```

---

## 📑 Core Research Notes

- **Theoretical Complexity & NP-Hardness:**
  - **[[Walsh_2014_Candy_Crush_NP_Hard]]**: *Candy Crush is NP-hard* (arXiv:1403.5830). Reduction from 3-SAT to match-3 grid gadgets (Variable, Clause, and Wire gadgets).
  - **[[Demaine_Combinatorial_Game_Complexity]]**: MIT combinatorial game theory, tile-matching gadgets, and computational lower bounds across discrete puzzle games.
- **King AI Playtesting & Supervised Learning:**
  - **[[King_2018_Human_Like_Playtesting_CNN]]**: *Human-Like Playtesting with Deep Learning* (IEEE CIG 2018). Supervised CNNs trained on telemetry data for difficulty prediction and player churn prevention.
- **Hybrid Reinforcement Learning & Generalization:**
  - **[[Karimi_2021_CANDYRL_BaziGooshi_Hybrid_RL]]**: *CANDYRL / BaziGooshi: A Hybrid Reinforcement Learning Model for Generalization in Gameplay* (IEEE ToG 2021). Intrinsic vs. extrinsic reward shaping and multi-agent ensemble mixtures.
- **World Models & Modern Architectures:**
  - **[[DeepMind_Stochastic_MuZero_Afterstates_Match3]]**: *Planning in Stochastic Environments with a Learned Model* (Antonoglou et al., DeepMind 2022). Factoring deterministic afterstates $s^{\text{after}}$ from stochastic dispenser refills for match-3 tree search.
  - **[[Sanford_2023_Higher_Order_Attention_Match3_Complexity]]**: *Representational Strengths and Limitations of Transformers on Relational Tasks* (Sanford et al., 2023). Proving why standard single-layer attention fails on Match-3 triplets and why multi-stage attention is mathematically required.
  - **[[Silver_Schrittwieser_MuZero_World_Models_Match3]]**: Adapting DeepMind's MuZero latent dynamics function $g_\theta(s_t, a_t) \to s_{t+1}$ and value estimation to stochastic puzzle manifolds.
  - **[[Transformers_and_Spatial_Attention_in_Grid_Worlds]]**: Multi-Head Spatial Self-Attention on discrete 2D manifolds with voids, non-Euclidean topologies, and teleporters.

---

## 🔗 Related Notes
- Return to: **[[00_Home_MOC]]**
- Jump to: **[[01_Architecture_MOC]]** | **[[05_Campaign_and_Benchmarks_MOC]]**
