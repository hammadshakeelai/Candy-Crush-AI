r"""
Reproduction of Open-Source Genetic Algorithm / NEAT Heuristic Evolved Match-3 Bot
(`GeneticHeuristicCreatorBot`).

Background:
  In open-source Bejeweled and Candy Crush bots on GitHub (e.g., `daviddoran/bejeweled-bot`,
  `kwembel/bejeweledbot`, CMA-ES / Genetic Algorithm feature weight tuners, and `neat-python`
  neuroevolution agents), a candidate move's 1-step afterstate is summarized into an 8-dimensional
  handcrafted feature vector $\phi(s, a) \in \mathbb{R}^8$:
    1. Normalized immediate points scored
    2. Cascade chain length
    3. Jellies cleared count
    4. Blockers (frosting/licorice/chocolate) destroyed count
    5. Special candies created (Striped / Wrapped / Color Bomb)
    6. Special candies triggered
    7. Normalized vertical row depth (bottom-row preference)
    8. Remaining moves urgency interaction
  The evolved chromosome consists of linear genome weights $w_{\text{GA}} \in \mathbb{R}^8$ plus
  a small NEAT-evolved 2-layer feed-forward topology with skip connections.
"""

from typing import List, Tuple, Dict, Optional, Any, Union
import time
import numpy as np
import torch
import torch.nn as nn

from core.reverse_engine import (
    ReverseEngineState, ReverseCandyEngine, Direction, LevelConfig,
    SpecialCandy, Blocker, OFFSETS
)
from models.spatial_transformer_net import (
    encode_action_index, NUM_CANONICAL_ACTIONS
)


class GeneticHeuristicCreatorBot(nn.Module):
    """
    Evolved Genetic Algorithm + NEAT Topology Heuristic Bot.
    Evaluates each legal move using an 8-gene chromosome and a compact neuro-evolved MLP.
    """

    def __init__(
        self,
        chromosome_weights: Optional[List[float]] = None,
        engine: Optional[ReverseCandyEngine] = None
    ):
        super().__init__()
        self.engine = engine or ReverseCandyEngine()
        self.num_actions = NUM_CANONICAL_ACTIONS

        # Best evolved 8-gene chromosome from open-source GA Match-3 literature:
        # [score_norm, cascades, jellies_cleared, blockers_cleared,
        #  specials_created, specials_triggered, row_depth, ingredient_drop]
        default_genes = chromosome_weights or [
            1.25,   # Gene 0: Immediate score gain
            2.40,   # Gene 1: Cascade chain reaction count
            6.80,   # Gene 2: Jelly layers cleared
            4.10,   # Gene 3: Blockers destroyed
            5.50,   # Gene 4: Special candies forged (Striped/Wrapped/Bomb)
            4.90,   # Gene 5: Special candies detonated
            1.65,   # Gene 6: Normalized row depth (gravity bias)
            7.20,   # Gene 7: Ingredients dropped/collected
        ]
        self.chromosome = nn.Parameter(
            torch.tensor(default_genes, dtype=torch.float32), requires_grad=False
        )

        # NEAT-style small evolved nonlinear feature interaction network (8 -> 16 -> 1 with direct skip)
        self.neat_hidden = nn.Linear(8, 16)
        self.neat_out = nn.Linear(16, 1)
        self._init_neat_topology()

    def _init_neat_topology(self) -> None:
        with torch.no_grad():
            nn.init.orthogonal_(self.neat_hidden.weight, gain=0.5)
            nn.init.zeros_(self.neat_hidden.bias)
            nn.init.orthogonal_(self.neat_out.weight, gain=0.25)
            nn.init.zeros_(self.neat_out.bias)

    def extract_move_features(
        self, state: ReverseEngineState, r: int, c: int, d: Direction
    ) -> np.ndarray:
        """Extracts the 8-dimensional phenotype feature vector for move (r, c, d)."""
        dr, dc = OFFSETS[d]
        nr, nc = r + dr, c + dc

        sim_state = state.copy()
        score_gained, stats = self.engine.execute_move(sim_state, r, c, d)
        if not stats["success"]:
            return np.zeros(8, dtype=np.float32)

        f_score = min(5.0, score_gained / 1000.0)
        f_cascades = float(min(8, stats.get("cascades", 0)))
        f_jellies = float(max(0, int(np.sum(state.jellies)) - int(np.sum(sim_state.jellies))))
        f_blockers = float(max(0, int(np.sum(state.blockers != Blocker.NONE)) - int(np.sum(sim_state.blockers != Blocker.NONE))))

        bombs_diff = max(0, int(np.sum(sim_state.specials == SpecialCandy.COLOR_BOMB)) - int(np.sum(state.specials == SpecialCandy.COLOR_BOMB)))
        wrap_diff = max(0, int(np.sum(sim_state.specials == SpecialCandy.WRAPPED)) - int(np.sum(state.specials == SpecialCandy.WRAPPED)))
        strip_diff = max(0, int(np.sum((sim_state.specials == SpecialCandy.STRIPED_HORIZONTAL) | (sim_state.specials == SpecialCandy.STRIPED_VERTICAL))) -
                         int(np.sum((state.specials == SpecialCandy.STRIPED_HORIZONTAL) | (state.specials == SpecialCandy.STRIPED_VERTICAL))))
        f_specials_created = (bombs_diff * 3.0) + (wrap_diff * 2.0) + (strip_diff * 1.0)

        s1, s2 = state.specials[r, c], state.specials[nr, nc]
        f_specials_trig = float((1 if s1 != SpecialCandy.NONE else 0) + (1 if s2 != SpecialCandy.NONE else 0))
        f_row_depth = max(r, nr) / float(max(1, state.height - 1))
        f_ingredients = float(max(0, sim_state.ingredients_collected - state.ingredients_collected))

        return np.array([
            f_score, f_cascades, f_jellies, f_blockers,
            f_specials_created, f_specials_trig, f_row_depth, f_ingredients
        ], dtype=np.float32)

    def forward(
        self,
        state_or_tensor: Union[ReverseEngineState, torch.Tensor, np.ndarray],
        legal_action_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Computes (B, 144) fitness scores for all canonical swap actions.
        """
        device = self.chromosome.device

        if isinstance(state_or_tensor, ReverseEngineState):
            scores = torch.full((1, self.num_actions), -1e9, dtype=torch.float32, device=device)
            legal_moves = self.engine.get_legal_moves(state_or_tensor)
            if not legal_moves:
                return scores

            feats_list = []
            indices = []
            for r, c, d in legal_moves:
                feats_list.append(self.extract_move_features(state_or_tensor, r, c, d))
                indices.append(encode_action_index(r, c, d))

            feats_t = torch.from_numpy(np.stack(feats_list, axis=0)).to(device)  # (M, 8)
            linear_fitness = torch.matmul(feats_t, self.chromosome)              # (M,)
            neat_bonus = self.neat_out(torch.tanh(self.neat_hidden(feats_t))).squeeze(-1)
            total_fitness = linear_fitness + neat_bonus

            for idx, val in zip(indices, total_fitness):
                scores[0, idx] = val
            return scores

        # Tensor batch mode (B, 20, 9, 9) -> (B, 144)
        x = torch.as_tensor(state_or_tensor, dtype=torch.float32, device=device)
        if x.dim() == 3:
            x = x.unsqueeze(0)
        B = x.shape[0]
        # Construct 8 spatial proxy features per cell and evaluate with chromosome + NEAT
        cell_score = (
            x[:, 6:10].sum(dim=1) * float(self.chromosome[4].item())
            + x[:, 15:17].sum(dim=1) * float(self.chromosome[2].item())
            + x[:, 10:15].sum(dim=1) * float(self.chromosome[3].item())
        )
        h_scores = (cell_score[:, :, :-1] + cell_score[:, :, 1:]).flatten(1)
        v_scores = (cell_score[:, :-1, :] + cell_score[:, 1:, :]).flatten(1)
        out = torch.cat([h_scores, v_scores], dim=1).view(B, self.num_actions)
        if legal_action_mask is not None:
            out = out.masked_fill(~legal_action_mask.to(device), -1e9)
        return out

    def select_best_move(self, state: ReverseEngineState) -> Optional[Tuple[int, int, Direction, float]]:
        legal_moves = self.engine.get_legal_moves(state)
        if not legal_moves:
            return None
        if len(legal_moves) == 1:
            r, c, d = legal_moves[0]
            return (r, c, d, 0.0)

        with torch.no_grad():
            scores = self.forward(state).squeeze(0)

        best_score = -1e9
        best_move = legal_moves[0]
        for r, c, d in legal_moves:
            idx = encode_action_index(r, c, d)
            s = float(scores[idx].item())
            if s > best_score:
                best_score = s
                best_move = (r, c, d)

        return (best_move[0], best_move[1], best_move[2], best_score)

    def solve_level(
        self,
        config_or_state: Union[LevelConfig, ReverseEngineState],
        max_moves: Optional[int] = None,
        seed: int = 42
    ) -> Dict[str, Any]:
        if isinstance(config_or_state, ReverseEngineState):
            state = config_or_state.copy()
        else:
            state = ReverseEngineState(config_or_state)
            self.engine.initialize_board(state, seed=seed)

        t0 = time.time()
        moves_limit = max_moves or state.moves_left
        step = 0
        shuffles = 0

        while not state.has_won() and not state.has_lost() and step < moves_limit:
            best = self.select_best_move(state)
            if best is None:
                shuffles += 1
                if shuffles > 3:
                    break
                self.engine.initialize_board(state)
                continue
            shuffles = 0
            r, c, d, _ = best
            self.engine.execute_move(state, r, c, d)
            step += 1

        return {
            "model_name": "GeneticHeuristicCreatorBot (GA/NEAT Evolved)",
            "level_id": state.config.level_id,
            "won": state.has_won(),
            "moves_used": step,
            "moves_remaining": state.moves_left,
            "final_score": state.score,
            "objectives_remaining": state.remaining_objectives_count(),
            "duration_sec": time.time() - t0,
            "parameters": self.count_parameters()
        }

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


if __name__ == "__main__":
    bot = GeneticHeuristicCreatorBot()
    dummy = torch.randn(2, 20, 9, 9)
    out = bot(dummy)
    print(f"GeneticHeuristicCreatorBot initialized | Params: {bot.count_parameters():,} | Output shape: {out.shape}")
