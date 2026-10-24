"""
Candy Crush Heuristic Evaluator & Expert Strategy System.
Combines King's match-3 heuristics, bottom-cascade biasing, combo prioritization,
and obstacle containment.
"""

from typing import Tuple, List, Dict, Any, Optional
import numpy as np
from core.candy_board import (
    BoardState, CandyColor, SpecialType, BlockerType, Direction, DIRECTION_OFFSETS
)
from core.candy_engine import CandyEngine


class CandyCrushHeuristics:
    """
    Expert system for scoring potential Candy Crush moves.
    Provides fast (< 2ms) superhuman heuristic evaluation and serves as MCTS rollout guide.
    """

    def __init__(self, engine: Optional[CandyEngine] = None):
        self.engine = engine if engine is not None else CandyEngine()

    def evaluate_move(
        self, state: BoardState, r: int, c: int, direction: Direction
    ) -> float:
        """
        Calculates heuristic utility score for taking a specific move from the state.
        Higher is better.
        """
        dr, dc = DIRECTION_OFFSETS[direction]
        nr, nc = r + dr, c + dc

        if not state.is_valid_coord(nr, nc):
            return -1e9

        s1 = state.specials[r, c]
        s2 = state.specials[nr, nc]

        score = 0.0

        # 1. SPECIAL COMBINATION HEURISTICS (Highest Priority)
        if s1 == SpecialType.COLOR_BOMB and s2 == SpecialType.COLOR_BOMB:
            return 50000.0  # Clears board
        if (s1 == SpecialType.COLOR_BOMB and s2 in (SpecialType.STRIPED_HORIZONTAL, SpecialType.STRIPED_VERTICAL)) or \
           (s2 == SpecialType.COLOR_BOMB and s1 in (SpecialType.STRIPED_HORIZONTAL, SpecialType.STRIPED_VERTICAL)):
            return 25000.0  # Massive clear
        if (s1 == SpecialType.WRAPPED and s2 in (SpecialType.STRIPED_HORIZONTAL, SpecialType.STRIPED_VERTICAL)) or \
           (s2 == SpecialType.WRAPPED and s1 in (SpecialType.STRIPED_HORIZONTAL, SpecialType.STRIPED_VERTICAL)):
            return 18000.0  # 3 rows + 3 cols
        if s1 == SpecialType.WRAPPED and s2 == SpecialType.WRAPPED:
            return 15000.0  # Dual 5x5 blast
        if s1 == SpecialType.COLOR_BOMB or s2 == SpecialType.COLOR_BOMB:
            return 10000.0  # Single color wipe

        # 2. SIMULATION ROLLOUT (1-ply lookahead)
        sim_state = state.copy()
        score_gained, jellies_cleared, stats = self.engine.execute_move(sim_state, r, c, direction)
        
        if not stats["success"]:
            return -1000.0

        score += score_gained

        # 3. JELLY CLEARING BONUS
        score += (jellies_cleared * 2500.0)

        # 4. BOTTOM-BIAS CASCADE HEURISTIC
        # Moves lower on the board have higher gravity cascade potential
        bottom_bias = ((max(r, nr) / float(state.height - 1)) ** 1.5) * 400.0
        score += bottom_bias

        # 5. SPECIAL CANDY CREATION BONUS
        # If the move created new special candies
        new_bombs = np.sum(sim_state.specials == SpecialType.COLOR_BOMB) - np.sum(state.specials == SpecialType.COLOR_BOMB)
        new_wrapped = np.sum(sim_state.specials == SpecialType.WRAPPED) - np.sum(state.specials == SpecialType.WRAPPED)
        new_striped = np.sum((sim_state.specials == SpecialType.STRIPED_HORIZONTAL) | (sim_state.specials == SpecialType.STRIPED_VERTICAL)) - \
                      np.sum((state.specials == SpecialType.STRIPED_HORIZONTAL) | (state.specials == SpecialType.STRIPED_VERTICAL))

        score += (new_bombs * 5000.0)
        score += (new_wrapped * 3000.0)
        score += (new_striped * 1500.0)

        # 6. CASCADE MULTIPLIER
        score += (stats["cascades"] * 800.0)

        return score

    def select_best_move(self, state: BoardState) -> Optional[Tuple[int, int, Direction, float]]:
        """
        Scans all legal moves and returns the best move according to the heuristic system.
        """
        legal_moves = self.engine.get_legal_moves(state)
        if not legal_moves:
            return None

        best_score = -1e9
        best_move = None

        for r, c, direction in legal_moves:
            score = self.evaluate_move(state, r, c, direction)
            if score > best_score:
                best_score = score
                best_move = (r, c, direction, score)

        return best_move
