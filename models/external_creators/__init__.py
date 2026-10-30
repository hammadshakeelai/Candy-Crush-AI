"""
External Creator & Academic Baseline Models Suite (`models.external_creators`).
Provides faithful PyTorch / Python reproductions of published and open-source Match-3 / Candy Crush AI models
for head-to-head benchmarking against AlphaCandy architectures.
"""

from models.external_creators.king_gudmundsson_cnn_2018 import (
    KingPlaytestCNN2018, KingGudmundssonAgent2018
)
from models.external_creators.mulliganaceous_policy_net import (
    MulliganaceousPolicyNet, MulliganaceousPolicyAgent
)
from models.external_creators.karimi_candyrl_hybrid import (
    KarimiCandyRL2020
)
from models.external_creators.vanilla_ppo_actor_critic import (
    VanillaPPOMatch3Net, VanillaPPOAgent
)
from models.external_creators.uct_mcts_creator_bot import (
    UCTCreatorAgent
)
from models.external_creators.neat_genetic_heuristic_bot import (
    GeneticHeuristicCreatorBot
)
from models.external_creators.greedy_cascade_bot import (
    GreedyImmediateScoreBot, BottomRowGreedyBot
)

__all__ = [
    "KingPlaytestCNN2018",
    "KingGudmundssonAgent2018",
    "MulliganaceousPolicyNet",
    "MulliganaceousPolicyAgent",
    "KarimiCandyRL2020",
    "VanillaPPOMatch3Net",
    "VanillaPPOAgent",
    "UCTCreatorAgent",
    "GeneticHeuristicCreatorBot",
    "GreedyImmediateScoreBot",
    "BottomRowGreedyBot",
]
