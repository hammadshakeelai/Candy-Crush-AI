### 🏆 Creator Benchmark Arena Leaderboard (Official King Levels: [1, 5, 10, 15, 20, 25])

| Rank | Model / Agent | Creator / Origin | Architecture / Paradigm | Params | Win Rate | Avg Moves | Avg Score | Avg Obj Left | Time/Lvl |
| :---: | :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | **GreedyImmediateScoreBot** | AlexEne/CCrush-Bot & OpenCV Bots | 1-Step Immediate Point & Cascade Maximizer | 0 (Search/Rule) | **100.0%** (6/6) | 5.0 | 42,771 | 0.0 | 0.541s |
| 2 | **KarimiCandyRL2020** | Karimi et al. (KTH & King, 2020) | 4-Skill Hybrid Intrinsic RL (HSH + Avg Bagging) | 1,561,396 | **100.0%** (6/6) | 5.0 | 42,594 | 0.0 | 0.550s |
| 3 | **GeneticHeuristicCreatorBot** | Open-Source GA/NEAT Match-3 Bots | 8-Gene Evolved Chromosome + NEAT MLP | 169 | **100.0%** (6/6) | 3.7 | 39,390 | 0.0 | 0.417s |
| 4 | **UCTCreatorAgent** | Poromaa (2017) / Shin et al. | Classic UCT MCTS (28 Rollouts, No Afterstates) | 0 (Search/Rule) | **100.0%** (6/6) | 7.8 | 39,090 | 0.0 | 2.545s |
| 5 | **AlphaCandy Grandmaster Unified** | Our AlphaCandy Suite | Spatial Transformer + DQN + Macro-Combo Expectimax | 14,412,821 | **100.0%** (6/6) | 3.8 | 33,285 | 0.0 | 4.613s |
| 6 | **KingPlaytestCNN2018** | Gudmundsson et al. (King, 2018) | 7-Layer 3x3 ELU ConvNet (Supervised Playtester) | 1,846,529 | **83.3%** (5/6) | 10.0 | 31,967 | 0.2 | 0.147s |
| 7 | **MulliganaceousPolicyNet** | mulliganaceous / Community RL | 4-Block ResNet + Pairwise Swap Policy Head | 344,642 | **83.3%** (5/6) | 7.0 | 30,271 | 0.2 | 0.080s |
| 8 | **BottomRowGreedyBot** | daviddoran/bejeweled-bot Baseline | Bottom-Row Gravity Cascade Heuristic | 0 (Search/Rule) | **66.7%** (4/6) | 10.2 | 26,103 | 1.5 | 0.078s |
| 9 | **AlphaCandy Dueling DQN Master** | Our AlphaCandy Suite | Double Dueling DQN + PER + Action Masking | 2,027,697 | **66.7%** (4/6) | 11.5 | 27,066 | 2.0 | 0.148s |
| 10 | **VanillaPPOMatch3Net** | kamildar/gym-match3 / CleanRL | Shared 4-Layer CNN Actor-Critic PPO | 804,977 | **66.7%** (4/6) | 13.2 | 35,875 | 257.0 | 0.114s |

*Total Tournament Runtime: 55.40s*
