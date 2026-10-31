### 🏆 Creator Benchmark Arena Leaderboard (Official King Levels: [1, 5, 10, 15, 20, 25])

| Rank | Model / Agent | Creator / Origin | Architecture / Paradigm | Params | Win Rate | Avg Moves | Avg Score | Avg Obj Left | Time/Lvl |
| :---: | :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | **KingPlaytestCNN2018** | Gudmundsson et al. (King, 2018) | 7-Layer 3x3 ELU ConvNet (Supervised Playtester) | 1,846,529 | **100.0%** (6/6) | 4.0 | 53,591 | 0.0 | 0.073s |
| 2 | **BottomRowGreedyBot** | daviddoran/bejeweled-bot Baseline | Bottom-Row Gravity Cascade Heuristic | 0 (Search/Rule) | **100.0%** (6/6) | 5.5 | 50,889 | 0.0 | 0.063s |
| 3 | **KarimiCandyRL2020** | Karimi et al. (KTH & King, 2020) | 4-Skill Hybrid Intrinsic RL (HSH + Avg Bagging) | 1,561,396 | **100.0%** (6/6) | 2.2 | 45,537 | 0.0 | 0.260s |
| 4 | **AlphaCandy Grandmaster Unified** | Our AlphaCandy Suite | Spatial Transformer + DQN + Macro-Combo Expectimax | 14,412,821 | **100.0%** (6/6) | 3.7 | 45,496 | 0.0 | 3.699s |
| 5 | **UCTCreatorAgent** | Poromaa (2017) / Shin et al. | Classic UCT MCTS (28 Rollouts, No Afterstates) | 0 (Search/Rule) | **100.0%** (6/6) | 4.2 | 41,968 | 0.0 | 1.307s |
| 6 | **VanillaPPOMatch3Net** | kamildar/gym-match3 / CleanRL | Shared 4-Layer CNN Actor-Critic PPO | 804,977 | **100.0%** (6/6) | 3.8 | 36,145 | 0.0 | 0.042s |
| 7 | **GeneticHeuristicCreatorBot** | Open-Source GA/NEAT Match-3 Bots | 8-Gene Evolved Chromosome + NEAT MLP | 169 | **100.0%** (6/6) | 2.2 | 34,414 | 0.0 | 0.238s |
| 8 | **GreedyImmediateScoreBot** | AlexEne/CCrush-Bot & OpenCV Bots | 1-Step Immediate Point & Cascade Maximizer | 0 (Search/Rule) | **83.3%** (5/6) | 4.8 | 91,674 | 0.2 | 0.648s |
| 9 | **MulliganaceousPolicyNet** | mulliganaceous / Community RL | 4-Block ResNet + Pairwise Swap Policy Head | 344,642 | **83.3%** (5/6) | 7.0 | 77,348 | 0.2 | 0.092s |
| 10 | **AlphaCandy Dueling DQN Master** | Our AlphaCandy Suite | Double Dueling DQN + PER + Action Masking | 2,027,697 | **83.3%** (5/6) | 8.7 | 58,661 | 0.2 | 0.124s |

*Total Tournament Runtime: 39.28s*
