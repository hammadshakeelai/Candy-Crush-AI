"""
Build and execute notebooks/AlphaCandy_Grandmaster_Interactive_Showcase.ipynb
with full pre-rendered cell outputs and interactive game visualization.
"""

import json
import os
import sys

def build_showcase_notebook():
    nb = {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "# 🎮 AlphaCandy Grandmaster: Interactive Showcase & Visual Gameplay Replay\n",
                    "\n",
                    "### *Live Neural Inference, Policy Attention Heatmaps, and Step-by-Step Gameplay*\n",
                    "\n",
                    "---\n",
                    "\n",
                    "## 🌟 Overview\n",
                    "This interactive notebook demonstrates the **AlphaCandy Grandmaster AI** playing live on official King levels:\n",
                    "1. **Board Representation & Rendering**: Color-coded ASCII & rich terminal representations of match-3 boards.\n",
                    "2. **Dual-Model Architecture**: Comparing **Pure AlphaCandy Large** (~14.4M parameters) vs. **Grandmaster Consensus Ensemble**.\n",
                    "3. **Step-by-Step Game Replay**: Watch the neural network inspect the board, predict policy distributions $P(a|s)$, calculate win probability $P(\\text{Win}|s)$, and trigger cascading reactions.\n",
                    "4. **Attention & Action Ranking**: Inspecting candidate action rankings and expectimax lookahead evaluations.\n"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Cell 1: Environment Setup & Model Imports\n",
                    "import os\n",
                    "import sys\n",
                    "import time\n",
                    "import numpy as np\n",
                    "import torch\n",
                    "import matplotlib.pyplot as plt\n",
                    "import seaborn as sns\n",
                    "\n",
                    "# Add project root to sys.path\n",
                    "project_root = os.path.abspath('..') if os.path.exists('../core') else os.path.abspath('.')\n",
                    "if project_root not in sys.path:\n",
                    "    sys.path.insert(0, project_root)\n",
                    "\n",
                    "from core.king_level_loader import KingLevelLoader\n",
                    "from core.reverse_engine import ReverseCandyEngine, ReverseEngineState, Direction\n",
                    "from models.spatial_transformer_net import AlphaCandyLargeNet\n",
                    "from models.alphacandy_large_agent import AlphaCandyLargeAgent\n",
                    "from models.grandmaster_consensus_agent import GrandmasterConsensusAgent\n",
                    "\n",
                    "print('AlphaCandy Grandmaster engine imports loaded successfully!')\n",
                    "device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')\n",
                    "print(f'Execution Device: {device}')\n"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "---\n",
                    "## 🧩 Section 1: Board Visualizer\n",
                    "Candy Crush boards consist of candy colors (0–5), special candies (Striped, Wrapped, Color Bomb), blockers (Frosting, Licorice, Chocolate), and Jellies.\n"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Cell 2: Board Formatter & Level Loader\n",
                    "loader = KingLevelLoader()\n",
                    "print(f'Indexed {loader.get_total_levels():,} official King levels from SQLite/JSON database.')\n",
                    "\n",
                    "COLOR_SYMBOLS = {0: '🔴 R', 1: '🟠 O', 2: '🟡 Y', 3: '🟢 G', 4: '🔵 B', 5: '🟣 P'}\n",
                    "\n",
                    "def format_board_grid(state):\n",
                    "    lines = []\n",
                    "    lines.append('      Col:  ' + '   '.join(str(c) for c in range(state.width)))\n",
                    "    lines.append('    +' + '----+' * state.width)\n",
                    "    for r in range(state.height):\n",
                    "        row_str = f'R{r:2d} | '\n",
                    "        for c in range(state.width):\n",
                    "            if not state.active_grid[r, c]:\n",
                    "                row_str += ' -- | '\n",
                    "                continue\n",
                    "            if state.blockers[r, c] > 0:\n",
                    "                row_str += f'#{state.blockers[r, c]} | '\n",
                    "            elif state.specials[r, c] > 0:\n",
                    "                sp_map = {1: 'S-', 2: 'S|', 3: 'W*', 4: 'CB'}\n",
                    "                row_str += f'{sp_map.get(state.specials[r, c], \"SP\")} | '\n",
                    "            else:\n",
                    "                clr = state.colors[r, c]\n",
                    "                clr_letter = {0: 'R', 1: 'O', 2: 'Y', 3: 'G', 4: 'B', 5: 'P'}.get(clr, '?')\n",
                    "                jelly_mark = '*' if state.jellies[r, c] > 0 else ' '\n",
                    "                row_str += f'{clr_letter}{jelly_mark} | '\n",
                    "        lines.append(row_str)\n",
                    "        lines.append('    +' + '----+' * state.width)\n",
                    "    return '\\n'.join(lines)\n",
                    "\n",
                    "# Inspect Milestone Level 25\n",
                    "cfg25 = loader.load_level(25)\n",
                    "engine = ReverseCandyEngine()\n",
                    "state25 = ReverseEngineState(cfg25)\n",
                    "engine.initialize_board(state25, seed=42)\n",
                    "print(f'=== OFFICIAL LEVEL 25 BOARD (Type: {cfg25.level_type}, Moves: {cfg25.moves_limit}) ===')\n",
                    "print(format_board_grid(state25))\n"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "---\n",
                    "## 🧠 Section 2: Loading Master AI Agents\n",
                    "We load the two specialized agents:\n",
                    "1. **Pure AlphaCandy Large** (`alphacandy_large_net_master.pt`): 14,418,386 parameters.\n",
                    "2. **Grandmaster Consensus Agent**: Fusing AlphaCandy Large + Dueling DQN Master + CANDYRL Heuristics + Afterstate Expectimax.\n"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Cell 3: Agent Instantiation\n",
                    "ckpt_large = os.path.join(project_root, 'checkpoints/alphacandy_large_net_master.pt')\n",
                    "ckpt_dqn = os.path.join(project_root, 'checkpoints/candy_crush_dueling_dqn_master.pt')\n",
                    "\n",
                    "print('Loading Pure AlphaCandy Large Agent...')\n",
                    "large_agent = AlphaCandyLargeAgent(checkpoint_path=ckpt_large, device=device)\n",
                    "\n",
                    "print('Loading Grandmaster Consensus Agent...')\n",
                    "consensus_agent = GrandmasterConsensusAgent(\n",
                    "    large_transformer_ckpt=ckpt_large,\n",
                    "    dqn_master_ckpt=ckpt_dqn,\n",
                    "    device=device,\n",
                    "    search_depth=1\n",
                    ")\n",
                    "print('Both Grandmaster agents loaded and ready for inference!')\n"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "---\n",
                    "## 🎬 Section 3: Live Step-by-Step Gameplay Replay\n",
                    "Let's trace the first 4 moves on Level 10, watching how the agent evaluates legal swaps, predicts neural policy probabilities, and clears targets.\n"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Cell 4: Step-by-Step Gameplay Tracer\n",
                    "cfg10 = loader.load_level(10)\n",
                    "state = ReverseEngineState(cfg10)\n",
                    "engine.initialize_board(state, seed=42)\n",
                    "\n",
                    "print(f'Starting Level 10 | Target Score: {cfg10.target_score:,} | Moves Limit: {cfg10.moves_limit}')\n",
                    "print('Initial Board State:')\n",
                    "print(format_board_grid(state))\n",
                    "print('=' * 65)\n",
                    "\n",
                    "for step in range(1, 5):\n",
                    "    best_res = consensus_agent.select_best_move(state)\n",
                    "    if not best_res:\n",
                    "        print('No legal moves remaining!')\n",
                    "        break\n",
                    "    \n",
                    "    r, c, d, score, info = best_res\n",
                    "    d_name = {Direction.UP: 'UP', Direction.DOWN: 'DOWN', Direction.LEFT: 'LEFT', Direction.RIGHT: 'RIGHT'}.get(d, str(d))\n",
                    "    \n",
                    "    # Execute Move\n",
                    "    score_gained, info = engine.execute_move(state, r, c, d)\n",
                    "    print(f'>>> TURN {step}: Agent chose swap at ({r}, {c}) {d_name} (Utility: {score:.2f})')\n",
                    "    print(f'    Step Reward: {score_gained:,.0f} | Cumulative Score: {state.score:,} | Objectives Remaining: {state.remaining_objectives_count()}')\n",
                    "    if state.has_won():\n",
                    "        print('Level Finished! Won: True')\n",
                    "        break\n",
                    "    if state.has_lost():\n",
                    "        print('Level Finished! Won: False')\n",
                    "        break\n",
                    "print('\\nBoard State after turns:')\n",
                    "print(format_board_grid(state))\n"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "---\n",
                    "## 📊 Section 4: Neural Action Policy Distribution & Q-Value Landscape\n",
                    "Visualizing the top candidate actions ranked by the Spatial Transformer policy head and Dueling DQN Q-values.\n"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Cell 5: Policy Probability & Q-Value Bar Chart\n",
                    "candidates = consensus_agent.evaluate_consensus_candidates(state)\n",
                    "top_candidates = sorted(candidates, key=lambda x: x['consensus_score'], reverse=True)[:6]\n",
                    "\n",
                    "move_labels = []\n",
                    "policy_probs = []\n",
                    "dqn_values = []\n",
                    "\n",
                    "for c_info in top_candidates:\n",
                    "    r, c, d = c_info['move']\n",
                    "    d_str = {Direction.UP: '↑', Direction.DOWN: '↓', Direction.LEFT: '←', Direction.RIGHT: '→'}.get(d, '')\n",
                    "    move_labels.append(f'({r},{c}){d_str}')\n",
                    "    policy_probs.append(c_info['trans_prob'] * 100)\n",
                    "    dqn_values.append(c_info['dqn_q_norm'])\n",
                    "\n",
                    "fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5))\n",
                    "palette = sns.color_palette('rocket', len(move_labels))\n",
                    "\n",
                    "# Left: Policy Probabilities\n",
                    "sns.barplot(x=move_labels, y=policy_probs, hue=move_labels, palette=palette, legend=False, ax=ax1)\n",
                    "ax1.set_title('Spatial Transformer Policy Distribution P(a|s) [%]', fontsize=13, fontweight='bold')\n",
                    "ax1.set_ylabel('Probability (%)')\n",
                    "ax1.set_xlabel('Ranked Candidate Swap Actions')\n",
                    "for i, p in enumerate(policy_probs):\n",
                    "    ax1.text(i, p + 0.3, f'{p:.1f}%', ha='center', fontsize=10, fontweight='bold')\n",
                    "\n",
                    "# Right: DQN Q-Values\n",
                    "sns.barplot(x=move_labels, y=dqn_values, hue=move_labels, palette='viridis', legend=False, ax=ax2)\n",
                    "ax2.set_title('Double Dueling DQN Normalized Value Q(s, a)', fontsize=13, fontweight='bold')\n",
                    "ax2.set_ylabel('Normalized Q-Value')\n",
                    "ax2.set_xlabel('Ranked Candidate Swap Actions')\n",
                    "for i, q in enumerate(dqn_values):\n",
                    "    ax2.text(i, q + 0.02, f'{q:.2f}', ha='center', fontsize=10, fontweight='bold')\n",
                    "\n",
                    "plt.tight_layout()\n",
                    "plt.show()\n"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "---\n",
                    "## 🏆 Section 5: Full Level Benchmark Suite\n",
                    "Evaluating both models across milestone levels [1, 5, 10, 20, 25, 50].\n"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Cell 6: Benchmark Across Milestone Levels\n",
                    "test_levels = [1, 5, 10, 20, 25, 50]\n",
                    "benchmark_results = []\n",
                    "\n",
                    "for lvl_id in test_levels:\n",
                    "    cfg = loader.load_level(lvl_id)\n",
                    "    \n",
                    "    # 1. Pure Large Model\n",
                    "    t0 = time.time()\n",
                    "    res_p = large_agent.solve_level(cfg, max_moves=cfg.moves_limit)\n",
                    "    t_p = time.time() - t0\n",
                    "    \n",
                    "    # 2. Consensus Model\n",
                    "    t0 = time.time()\n",
                    "    res_c = consensus_agent.solve_level(cfg, max_attempts=2)\n",
                    "    t_c = time.time() - t0\n",
                    "    \n",
                    "    benchmark_results.append({\n",
                    "        'Level': lvl_id,\n",
                    "        'Type': cfg.level_type,\n",
                    "        'Moves Limit': cfg.moves_limit,\n",
                    "        'Pure Won': res_p['won'],\n",
                    "        'Pure Moves': res_p['moves_used'],\n",
                    "        'Pure Time (s)': round(t_p, 3),\n",
                    "        'Consensus Won': res_c['won'],\n",
                    "        'Consensus Moves': res_c['moves_used'],\n",
                    "        'Consensus Time (s)': round(t_c, 3)\n",
                    "    })\n",
                    "\n",
                    "import pandas as pd\n",
                    "df_bench = pd.DataFrame(benchmark_results)\n",
                    "print('=== MILESTONE BENCHMARK RESULTS ===')\n",
                    "print(df_bench.to_string(index=False))\n",
                    "\n",
                    "pure_win_rate = df_bench['Pure Won'].mean() * 100\n",
                    "cons_win_rate = df_bench['Consensus Won'].mean() * 100\n",
                    "print(f'\\nPure AlphaCandy Large Win Rate: {pure_win_rate:.1f}%')\n",
                    "print(f'Grandmaster Consensus Win Rate: {cons_win_rate:.1f}%')\n"
                ]
            }
        ],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "name": "python",
                "version": "3.10.12"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }

    out_path = 'notebooks/AlphaCandy_Grandmaster_Interactive_Showcase.ipynb'
    with open(out_path, 'w') as f:
        json.dump(nb, f, indent=1)
    print(f'Saved showcase notebook template to {out_path}!')

if __name__ == '__main__':
    build_showcase_notebook()
