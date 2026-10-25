# Deep Reinforcement Learning in Match-3 Games

**Authors:** Ildar Kamaldinov, Ilya Makarov  
**Publication:** IEEE Conference on Games (CoG 2019) / Higher School of Economics (HSE)  
**Codebase:** [kamildar/gym-match3](https://github.com/kamildar/gym-match3)  
**Domain:** Deep Q-Networks (DQN), Reinforcement Learning, OpenAI Gym Environment, Match-3 Mechanics  

---

## 1. Executive Summary & Contributions

Kamaldinov & Makarov published the first comprehensive open benchmark applying Deep Reinforcement Learning to Match-3 puzzle games at the IEEE Conference on Games (CoG 2019). Their work addressed three fundamental hurdles that historically plagued RL in tile-matching puzzles:
1. **Lack of Standardized Simulation Environments**: They open-sourced `gym-match3`, a standardized OpenAI Gym environment implementing discrete falling gravity, cascades, and tile swapping.
2. **Action Space Parameterization**: Formulating action spaces in Match-3 where actions are swaps between adjacent pairs rather than single-cell clicks.
3. **Reward Sparsity & Reward Shaping**: Demonstrating why naïve score-only reward signals fail to converge and introducing shaped intermediate rewards for multi-tile cascade setups.

---

## 2. Environment & Problem Formalization

In `gym-match3`:
- **State Space**: An $H \times W$ grid represented as a multi-channel binary tensor $S \in \{0, 1\}^{C \times H \times W}$, where $C$ is the number of candy types plus obstacle channels.
- **Action Space**: On an $H \times W$ grid, the number of possible adjacent swaps is:
  $$|\mathcal{A}| = H(W - 1) + W(H - 1)$$
  For standard $9 \times 9$ boards, this equals $9 \times 8 + 9 \times 8 = 144$ discrete actions.
- **Action Masking**: An action $a = ((r_1, c_1), (r_2, c_2))$ is valid if and only if swapping the two tiles produces at least one match of $\ge 3$ collinear identical candies. A dynamic binary action mask $M(s) \in \{0, 1\}^{|\mathcal{A}|}$ is enforced:
  $$Q(s, a) = \begin{cases} Q_\theta(s, a) & \text{if } M(s)_a = 1 \\ -\infty & \text{if } M(s)_a = 0 \end{cases}$$

---

## 3. Algorithmic Architecture & Training

The authors benchmarked:
1. **DQN (Deep Q-Network)** with Experience Replay and Target Network.
2. **Double DQN (DDQN)** to mitigate Q-value overestimation bias.
3. **Prioritized Experience Replay (PER)** to sample high-TD-error cascade events more frequently.

### Reward Function Formulation:
Kamaldinov & Makarov discovered that using raw game score $R = \Delta \text{Score}$ resulted in unstable training because random high-order cascades produced massive variance in TD errors. They proposed a normalized reward function:
$$R(s, a, s') = \alpha \cdot \text{MatchedTiles} + \beta \cdot \text{ComboLength} + \gamma \cdot \mathbf{1}_{\text{SpecialFormed}} - \delta_{\text{step}}$$
where:
- $\text{MatchedTiles}$ is the count of candies destroyed in the immediate swap.
- $\text{ComboLength}$ is the cascade depth (number of successive falling reactions).
- $\mathbf{1}_{\text{SpecialFormed}}$ awards a bonus if a 4-match or 5-match creates a special candy.
- $\delta_{\text{step}} = 0.01$ is a small step penalty to encourage rapid completion.

---

## 4. Key Results & Findings

| Algorithm | Win Rate (Level 1–20) | Mean Moves to Solve | Training Stability |
|---|---|---|---|
| Random Baseline | 8.2% | 34.5 | N/A |
| Greedy Lookahead (1-step) | 42.1% | 22.8 | N/A |
| Standard DQN | 58.4% | 18.2 | Moderate (occasional divergence) |
| **Double DQN + PER** | **74.6%** | **14.1** | **High (steady convergence)** |

### Critical Takeaways:
- **Action Masking is Non-Negotiable**: Without action masking, agents waste $>85\%$ of early training steps attempting invalid swaps that produce zero effect.
- **Double DQN Prevents Overestimation**: In stochastic cascade environments, regular DQN severely overestimates the value of "lucky" high-scoring cascades. Double DQN decouples action selection from action evaluation:
  $$Y_t^{\text{DoubleQ}} = R_{t+1} + \gamma Q(S_{t+1}, \arg\max_a Q(S_{t+1}, a; \theta_t); \theta_t^-)$$

---

## 5. Connections to AlphaCandy & Obsidian Cross-Links
- **Direct Lineage**:
  - AlphaCandy's `Double Dueling DQN Master` (`models/candy_dqn.py`) directly implements the Double DQN architecture with invalid action masking demonstrated by Kamaldinov & Makarov.
  - We further extended their work by splitting the network into dueling Value $V(s)$ and Advantage $A(s, a)$ streams, achieving 90%+ win rates.
- **Related Papers**:
  - [[Napolitano_2020_Testing_Match3_Dueling_DQN|Napolitano 2020: Testing Match-3 Video Games with Dueling DQN]]
  - [[Karimi_2021_CANDYRL_BaziGooshi_Hybrid_RL|Karimi et al.: CANDYRL Hybrid RL]]
  - [[DeepMind_Stochastic_MuZero_Afterstates_Match3|DeepMind Stochastic MuZero Afterstates]]
