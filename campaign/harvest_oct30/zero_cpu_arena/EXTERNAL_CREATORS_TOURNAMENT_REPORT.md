### 🏆 Creator Benchmark Arena Leaderboard (Official King Levels: [1, 5, 10, 15, 20, 25])

| Rank | Model / Agent | Creator / Origin | Architecture / Paradigm | Params | Win Rate | Avg Moves | Avg Score | Avg Obj Left | Time/Lvl |
| :---: | :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | **GreedyImmediateScoreBot** | AlexEne/CCrush-Bot & OpenCV Bots | 1-Step Immediate Point & Cascade Maximizer | 0 (Search/Rule) | **100.0%** (6/6) | 5.0 | 42,771 | 0.0 | 0.376s |
| 2 | **KarimiCandyRL2020** | Karimi et al. (KTH & King, 2020) | 4-Skill Hybrid Intrinsic RL (HSH + Avg Bagging) | 1,561,396 | **100.0%** (6/6) | 5.0 | 42,594 | 0.0 | 0.408s |
| 3 | **GeneticHeuristicCreatorBot** | Open-Source GA/NEAT Match-3 Bots | 8-Gene Evolved Chromosome + NEAT MLP | 169 | **100.0%** (6/6) | 3.8 | 39,648 | 0.0 | 0.298s |
| 4 | **UCTCreatorAgent** | Poromaa (2017) / Shin et al. | Classic UCT MCTS (28 Rollouts, No Afterstates) | 0 (Search/Rule) | **100.0%** (6/6) | 6.0 | 38,932 | 0.0 | 1.424s |
| 5 | **AlphaCandy Grandmaster Unified** | Our AlphaCandy Suite | Spatial Transformer + DQN + Macro-Combo Expectimax | 14,412,821 | **100.0%** (6/6) | 5.7 | 34,587 | 0.0 | 5.237s |
| 6 | **VanillaPPOMatch3Net** | kamildar/gym-match3 / CleanRL | Shared 4-Layer CNN Actor-Critic PPO | 804,977 | **66.7%** (4/6) | 10.3 | 40,114 | 0.2 | 0.076s |
| 7 | **KingPlaytestCNN2018** | Gudmundsson et al. (King, 2018) | 7-Layer 3x3 ELU ConvNet (Supervised Playtester) | 1,846,529 | **66.7%** (4/6) | 12.3 | 37,507 | 0.2 | 0.132s |
| 8 | **BottomRowGreedyBot** | daviddoran/bejeweled-bot Baseline | Bottom-Row Gravity Cascade Heuristic | 0 (Search/Rule) | **66.7%** (4/6) | 11.2 | 27,404 | 1.5 | 0.063s |
| 9 | **AlphaCandy Dueling DQN Master** | Our AlphaCandy Suite | Double Dueling DQN + PER + Action Masking | 2,027,697 | **66.7%** (4/6) | 11.7 | 28,082 | 2.0 | 0.108s |
| 10 | **MulliganaceousPolicyNet** | mulliganaceous / Community RL | 4-Block ResNet + Pairwise Swap Policy Head | 344,642 | **50.0%** (3/6) | 14.8 | 10,167 | 9.2 | 0.127s |

*Total Tournament Runtime: 49.50s*
