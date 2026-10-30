r"""
Reproduction of Classic Open-Source Computer-Vision / Screen-Scraping Greedy Bots
(`GreedyImmediateScoreBot` & `BottomRowGreedyBot`).

Background:
  The most widespread open-source Candy Crush / Bejeweled bots on GitHub
  (e.g., `AlexEne/CCrush-Bot`, `ahsodhi/candy-crush-bot`, `daviddoran/bejeweled-bot`)
  combine OpenCV screen capture with one of two deterministic 1-step greedy rules:
    1. `GreedyImmediateScoreBot`: Simulates each legal swap for 1 turn and picks the swap
       that yields the largest immediate point/cascade gain, completely ignoring jellies,
       disconnected islands, or multi-turn blocker mechanics.
    2. `BottomRowGreedyBot`: Standard folklore "always match at the bottom of the board"
       bot that selects the legal move with the largest row index $\max(r_1, r_2)$ to
       maximize gravity-induced random cascades from top dispensers.
"""

from typing import List, Tuple, Dict, Optional, Any, Union
import time
import numpy as np
import torch

from core.reverse_engine import (
    ReverseEngineState, ReverseCandyEngine, Direction, LevelConfig, OFFSETS
)
from models.spatial_transformer_net import (
    encode_action_index, decode_action_index, NUM_CANONICAL_ACTIONS
)


class GreedyImmediateScoreBot:
    """
    Open-Source Greedy 1-Step Immediate Score & Cascade Maximizer Bot.
    Evaluates immediate points gained from a 1-ply swap without objective awareness.
    """

    def __init__(self, engine: Optional[ReverseCandyEngine] = None):
        self.engine = engine or ReverseCandyEngine()

    def get_action_scores(
        self,
        state_or_tensor: Union[ReverseEngineState, torch.Tensor, np.ndarray]
    ) -> torch.Tensor:
        if isinstance(state_or_tensor, ReverseEngineState):
            scores = torch.full((1, NUM_CANONICAL_ACTIONS), -1e9, dtype=torch.float32)
            legal_moves = self.engine.get_legal_moves(state_or_tensor)
            for r, c, d in legal_moves:
                sim_state = state_or_tensor.copy()
                pts, stats = self.engine.execute_move(sim_state, r, c, d)
                if stats["success"]:
                    idx = encode_action_index(r, c, d)
                    scores[0, idx] = float(pts) + float(stats.get("cascades", 0) * 100.0)
            return scores

        x = torch.as_tensor(state_or_tensor, dtype=torch.float32)
        if x.dim() == 3:
            x = x.unsqueeze(0)
        B = x.shape[0]
        cell_density = x[:, :10].sum(dim=1)
        h_s = (cell_density[:, :, :-1] + cell_density[:, :, 1:]).flatten(1)
        v_s = (cell_density[:, :-1, :] + cell_density[:, 1:, :]).flatten(1)
        return torch.cat([h_s, v_s], dim=1).view(B, NUM_CANONICAL_ACTIONS)

    def __call__(
        self,
        state_or_tensor: Union[ReverseEngineState, torch.Tensor, np.ndarray]
    ) -> torch.Tensor:
        return self.get_action_scores(state_or_tensor)

    def select_best_move(self, state: ReverseEngineState) -> Optional[Tuple[int, int, Direction, float]]:
        legal_moves = self.engine.get_legal_moves(state)
        if not legal_moves:
            return None
        if len(legal_moves) == 1:
            r, c, d = legal_moves[0]
            return (r, c, d, 0.0)

        best_score = -1e9
        best_move = legal_moves[0]
        for r, c, d in legal_moves:
            sim_state = state.copy()
            pts, stats = self.engine.execute_move(sim_state, r, c, d)
            if not stats["success"]:
                continue
            val = float(pts) + float(stats.get("cascades", 0) * 100.0)
            if val > best_score:
                best_score = val
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
            "model_name": "GreedyImmediateScoreBot (Open-Source CV Bot)",
            "level_id": state.config.level_id,
            "won": state.has_won(),
            "moves_used": step,
            "moves_remaining": state.moves_left,
            "final_score": state.score,
            "objectives_remaining": state.remaining_objectives_count(),
            "duration_sec": time.time() - t0,
            "parameters": 0
        }

    def count_parameters(self) -> int:
        return 0


class BottomRowGreedyBot:
    """
    Classic Open-Source Bottom-Row Cascade Heuristic Bot.
    Always selects the legal move closest to the bottom of the grid (largest row index),
    breaking ties by horizontal orientation (which disturbs more columns).
    """

    def __init__(self, engine: Optional[ReverseCandyEngine] = None):
        self.engine = engine or ReverseCandyEngine()

    def get_action_scores(
        self,
        state_or_tensor: Union[ReverseEngineState, torch.Tensor, np.ndarray]
    ) -> torch.Tensor:
        if isinstance(state_or_tensor, ReverseEngineState):
            scores = torch.full((1, NUM_CANONICAL_ACTIONS), -1e9, dtype=torch.float32)
            legal_moves = self.engine.get_legal_moves(state_or_tensor)
            for r, c, d in legal_moves:
                dr, dc = OFFSETS[d]
                nr, nc = r + dr, c + dc
                idx = encode_action_index(r, c, d)
                horiz_bonus = 0.25 if d in (Direction.RIGHT, Direction.LEFT) else 0.0
                scores[0, idx] = float(max(r, nr)) + horiz_bonus
            return scores

        x = torch.as_tensor(state_or_tensor, dtype=torch.float32)
        if x.dim() == 3:
            x = x.unsqueeze(0)
        B = x.shape[0]
        scores = torch.zeros((B, NUM_CANONICAL_ACTIONS), dtype=torch.float32)
        for a_idx in range(NUM_CANONICAL_ACTIONS):
            r, c, d = decode_action_index(a_idx)
            scores[:, a_idx] = float(r) + (0.25 if d == Direction.RIGHT else 0.0)
        return scores

    def __call__(
        self,
        state_or_tensor: Union[ReverseEngineState, torch.Tensor, np.ndarray]
    ) -> torch.Tensor:
        return self.get_action_scores(state_or_tensor)

    def select_best_move(self, state: ReverseEngineState) -> Optional[Tuple[int, int, Direction, float]]:
        legal_moves = self.engine.get_legal_moves(state)
        if not legal_moves:
            return None
        if len(legal_moves) == 1:
            r, c, d = legal_moves[0]
            return (r, c, d, float(r))

        best_score = -1e9
        best_move = legal_moves[0]
        for r, c, d in legal_moves:
            dr, dc = OFFSETS[d]
            nr, _ = r + dr, c + dc
            horiz_bonus = 0.25 if d in (Direction.RIGHT, Direction.LEFT) else 0.0
            s = float(max(r, nr)) + horiz_bonus
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
            "model_name": "BottomRowGreedyBot (Bottom-Row Heuristic)",
            "level_id": state.config.level_id,
            "won": state.has_won(),
            "moves_used": step,
            "moves_remaining": state.moves_left,
            "final_score": state.score,
            "objectives_remaining": state.remaining_objectives_count(),
            "duration_sec": time.time() - t0,
            "parameters": 0
        }

    def count_parameters(self) -> int:
        return 0


if __name__ == "__main__":
    g_bot = GreedyImmediateScoreBot()
    b_bot = BottomRowGreedyBot()
    dummy = torch.randn(2, 20, 9, 9)
    print(f"GreedyImmediateScoreBot output: {g_bot(dummy).shape} | BottomRowGreedyBot output: {b_bot(dummy).shape}")
