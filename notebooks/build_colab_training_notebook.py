"""
Generate a modern, fully-featured Google Colab / Kaggle GPU training notebook
for AlphaCandy Large (~14.4M parameters) and Grandmaster Consensus.
"""

import json

nb = {
    "cells": [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# 🍭 AlphaCandy Grandmaster: GPU Training on Google Colab & Kaggle\n",
                "\n",
                "**Architecture:** Pure AlphaCandy Large (~14.4M Parameters) & Grandmaster Consensus Ensemble  \n",
                "**Hardware Accelerated:** Tensor Core FP16 Automatic Mixed Precision (AMP)  \n",
                "**Environment:** Official King Match-3 Physics Engine across 23,355 levels  \n",
                "\n",
                "---\n",
                "\n",
                "This production notebook enables one-click GPU training, policy evaluation, and checkpoint generation for:\n",
                "1. **AlphaCandy Large** (`AlphaCandyLargeNet` ~14.4M): 2-stage 8-head Spatial Self-Attention + 6 Residual Blocks.\n",
                "2. **Double Dueling DQN Master**: State-action advantage decoupling on 144 canonical swaps.\n",
                "3. **Grandmaster Consensus Ensemble**: Policy prior + Q-value consensus + Afterstate Expectimax lookahead.\n"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# 1. Verify GPU Acceleration & Tensor Cores\n",
                "import torch\n",
                "print(f\"PyTorch Version: {torch.__version__}\")\n",
                "print(f\"CUDA Available : {torch.cuda.is_available()}\")\n",
                "if torch.cuda.is_available():\n",
                "    print(f\"GPU Device     : {torch.cuda.get_device_name(0)}\")\n",
                "    print(f\"Memory Allocated: {torch.cuda.memory_allocated(0)/(1024**2):.2f} MB\")\n",
                "!nvidia-smi\n"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# 2. Clone Repository or Set Up Environment\n",
                "import os, sys\n",
                "if not os.path.exists('Candy-Crush-AI'):\n",
                "    !git clone https://github.com/hammadshakeelai/Candy-Crush-AI.git\n",
                "    %cd Candy-Crush-AI\n",
                "else:\n",
                "    %cd Candy-Crush-AI\n",
                "    !git pull origin main\n",
                "\n",
                "!pip install -q torch numpy pandas matplotlib seaborn\n",
                "sys.path.insert(0, os.path.abspath('.'))\n"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# 3. Inspect Master Level Database & Physics Engine\n",
                "from core.king_level_loader import KingLevelLoader\n",
                "from core.reverse_engine import ReverseCandyEngine\n",
                "\n",
                "loader = KingLevelLoader()\n",
                "print(f\"Indexed {len(loader.indexed_levels)} King levels!\")\n",
                "\n",
                "# Sample Milestone Level 25\n",
                "lvl25 = loader.load_level(25)\n",
                "print(f\"Level 25: Type={lvl25.level_type}, Moves={lvl25.moves_limit}, Dimensions={lvl25.height}x{lvl25.width}\")\n"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# 4. Initialize AlphaCandy Large (~14.4M Parameters)\n",
                "from models.spatial_transformer_net import AlphaCandyLargeNet\n",
                "\n",
                "device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')\n",
                "model = AlphaCandyLargeNet(in_channels=20, hidden_dim=256, num_actions=144).to(device)\n",
                "param_count = sum(p.numel() for p in model.parameters())\n",
                "print(f\"AlphaCandy Large Net Initialized on {device} with {param_count:,} parameters!\")\n"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# 5. Run Demonstration Imitation + GRPO Policy Training Cycle\n",
                "# This executes 1 full cycle of self-play demonstration + group relative policy optimization\n",
                "!python train_alphacandy_large.py --levels 1 5 10 20 25 --demo-epochs 2 --grpo-episodes 10\n"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# 6. Benchmark Pure AlphaCandy Large vs. Grandmaster Consensus Ensemble\n",
                "from models.alphacandy_large_agent import AlphaCandyLargeAgent\n",
                "from models.grandmaster_consensus_agent import GrandmasterConsensusAgent\n",
                "\n",
                "# Load Agents\n",
                "large_agent = AlphaCandyLargeAgent(checkpoint_path='checkpoints/alphacandy_large_net_master.pt', device=device)\n",
                "consensus_agent = GrandmasterConsensusAgent(\n",
                "    large_transformer_ckpt='checkpoints/alphacandy_large_net_master.pt',\n",
                "    dqn_master_ckpt='checkpoints/candy_crush_dueling_dqn_master.pt',\n",
                "    device=device,\n",
                "    search_depth=1\n",
                ")\n",
                "\n",
                "# Solve Level 10\n",
                "config10 = loader.load_level(10)\n",
                "res_pure = large_agent.solve_level(config10)\n",
                "res_cons = consensus_agent.solve_level(config10)\n",
                "\n",
                "print(f\"Level 10 Pure Large: Won={res_pure['won']} in {res_pure['moves_used']} moves (Score: {res_pure['final_score']:,})\")\n",
                "print(f\"Level 10 Consensus : Won={res_cons['won']} in {res_cons['moves_used']} moves (Score: {res_cons['final_score']:,})\")\n"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# 7. Download Master Checkpoints to Local Machine (Colab)\n",
                "try:\n",
                "    from google.colab import files\n",
                "    files.download('checkpoints/alphacandy_large_net_master.pt')\n",
                "    print(\"Triggered checkpoint download successfully!\")\n",
                "except Exception as e:\n",
                "    print(f\"Not in Colab or download skipped: {e}\")\n"
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
            "codemirror_mode": {"name": "ipython", "version": 3},
            "file_extension": ".py",
            "mimetype": "text/x-python",
            "name": "python",
            "nbconvert_exporter": "python",
            "pygments_lexer": "ipython3",
            "version": "3.10.12"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 4
}

with open('notebooks/candy_crush_colab_training.ipynb', 'w') as f:
    json.dump(nb, f, indent=1)

print("Successfully generated updated notebooks/candy_crush_colab_training.ipynb!")
