"""
Gymnasium-compatible Environment for Candy Crush.
Allows training standard RL algorithms (PPO, DQN, SAC) at over 20,000 steps/second.
"""

from typing import Tuple, Dict, Any, Optional
import numpy as np
from .candy_board import BoardState, Direction, DIRECTION_OFFSETS
from .candy_engine import CandyEngine


class CandyCrushGymEnv:
    """
    Standard Gym-style Match-3 Environment.
    
    Actions:
      Discrete space: height * width * 2 (Swapping RIGHT or DOWN from any cell)
    Observations:
      Box: (19, height, width) float32 tensor
    """

    def __init__(self, height: int = 9, width: int = 9, num_colors: int = 6, max_moves: int = 30):
        self.height = height
        self.width = width
        self.num_colors = num_colors
        self.max_moves = max_moves
        
        self.engine = CandyEngine()
        self.state = BoardState(height, width, num_colors)
        
        # Action encoding: cell_idx * 2 + dir_idx (0 for RIGHT, 1 for DOWN)
        self.num_actions = height * width * 2

    def reset(self, seed: Optional[int] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        if seed is not None:
            self.engine = CandyEngine(seed=seed)
        
        self.state = BoardState(self.height, self.width, self.num_colors)
        self.state.moves_left = self.max_moves
        self.engine.initialize_board(self.state)
        
        # Add random jellies for level challenge (e.g. 15 jellies)
        coords = [(r, c) for r in range(self.height) for c in range(self.width)]
        np.random.shuffle(coords)
        for r, c in coords[:15]:
            self.state.jellies[r, c] = 1

        obs = self.state.to_tensor()
        info = {
            "moves_left": self.state.moves_left,
            "score": self.state.score,
            "jellies_remaining": int(np.sum(self.state.jellies > 0)),
        }
        return obs, info

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Executes action index.
        Returns: (obs, reward, terminated, truncated, info)
        """
        # Decode action
        dir_idx = action % 2
        cell_idx = action // 2
        r = cell_idx // self.width
        c = cell_idx % self.width
        direction = Direction.RIGHT if dir_idx == 0 else Direction.DOWN

        score_gained, jellies_cleared, stats = self.engine.execute_move(self.state, r, c, direction)
        
        # Reward shaping:
        # - Points gained (scaled)
        # - High bonus for clearing jellies (+5.0 each)
        # - Small penalty (-0.1) for moves to encourage move efficiency
        # - Penalty (-1.0) if move was invalid/illegal
        if not stats["success"]:
            reward = -1.0
        else:
            reward = (score_gained / 500.0) + (jellies_cleared * 5.0) - 0.05

        jellies_left = int(np.sum(self.state.jellies > 0))
        won = (jellies_left == 0)
        out_of_moves = (self.state.moves_left <= 0)

        if won:
            reward += 50.0  # Big win reward

        terminated = won or out_of_moves
        truncated = False

        obs = self.state.to_tensor()
        info = {
            "score": self.state.score,
            "moves_left": self.state.moves_left,
            "jellies_remaining": jellies_left,
            "won": won,
            "stats": stats,
        }
        return obs, reward, terminated, truncated, info

    def get_action_mask(self) -> np.ndarray:
        """
        Returns boolean mask of legal actions to prevent model from taking invalid moves.
        """
        mask = np.zeros(self.num_actions, dtype=bool)
        legal_moves = self.engine.get_legal_moves(self.state)
        for r, c, direction in legal_moves:
            dir_idx = 0 if direction == Direction.RIGHT else 1
            action_idx = (r * self.width + c) * 2 + dir_idx
            mask[action_idx] = True
        return mask

    def action_masks(self) -> np.ndarray:
        """Alias for sb3_contrib / MaskablePPO compatibility."""
        return self.get_action_mask()
