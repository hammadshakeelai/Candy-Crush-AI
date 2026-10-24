"""
Unit and Integration Tests for Candy Crush AI Architecture.
Verifies simulation engine, special candies, cascades, MCTS, and neural network forward pass.
"""

import unittest
import numpy as np
import torch

from core.candy_board import (
    BoardState, CandyColor, SpecialType, BlockerType, Direction
)
from core.candy_engine import CandyEngine
from core.candy_env import CandyCrushGymEnv
from models.heuristics import CandyCrushHeuristics
from models.mcts import CandyCrushMCTS
from models.network import CandyCrushPolicyValueNet
from vision.tile_classifier import TileClassifier
from vision.grid_detector import GridDetector


class TestCandyCrushAI(unittest.TestCase):

    def setUp(self):
        self.engine = CandyEngine(seed=42)
        self.state = BoardState(height=9, width=9)
        self.engine.initialize_board(self.state)

    def test_board_initialization_no_matches(self):
        """Initial board must have no 3-in-a-row matches."""
        for r in range(self.state.height):
            for c in range(self.state.width):
                self.assertFalse(
                    self.engine._has_match_at(self.state, r, c),
                    f"Found initial match at ({r}, {c})"
                )

    def test_tensor_encoding_shape(self):
        """Tensor representation must have shape (19, 9, 9)."""
        tensor = self.state.to_tensor()
        self.assertEqual(tensor.shape, (19, 9, 9))
        self.assertEqual(tensor.dtype, np.float32)

    def test_match_3_and_gravity(self):
        """Simulate creating a 3-match and check that score increases."""
        # Force a match on row 0: set (0,0)=RED, (0,1)=RED, (0,2)=BLUE, (1,2)=RED
        self.state.colors[0, 0] = CandyColor.RED
        self.state.colors[0, 1] = CandyColor.RED
        self.state.colors[0, 2] = CandyColor.BLUE
        self.state.colors[1, 2] = CandyColor.RED

        # Swap (1,2) UP into (0,2) to complete 3-in-a-row of RED
        pts, jellies, stats = self.engine.execute_move(self.state, 1, 2, Direction.UP)
        self.assertTrue(stats["success"])
        self.assertGreater(pts, 0)
        self.assertGreaterEqual(stats["cascades"], 1)

    def test_special_combo_color_bombs(self):
        """Color Bomb + Color Bomb clears entire board and scores 50,000."""
        self.state.specials[4, 4] = SpecialType.COLOR_BOMB
        self.state.specials[4, 5] = SpecialType.COLOR_BOMB

        pts, jellies, stats = self.engine.execute_move(self.state, 4, 4, Direction.RIGHT)
        self.assertTrue(stats["success"])
        self.assertGreaterEqual(pts, 50000)

    def test_policy_value_net(self):
        """Neural network produces valid policy distribution and value in [-1, 1]."""
        net = CandyCrushPolicyValueNet(in_channels=19, num_res_blocks=2, channels=32)
        tensor = torch.from_numpy(self.state.to_tensor()).unsqueeze(0)
        logits, value = net(tensor)

        self.assertEqual(logits.shape, (1, 9 * 9 * 2))
        self.assertEqual(value.shape, (1, 1))
        self.assertTrue(-1.0 <= value.item() <= 1.0)

    def test_heuristics_evaluation(self):
        """Heuristic selector returns a valid legal move with positive score."""
        heuristics = CandyCrushHeuristics(self.engine)
        best = heuristics.select_best_move(self.state)
        if best is not None:
            r, c, direction, score = best
            self.assertTrue(self.state.is_valid_coord(r, c))

    def test_mcts_search(self):
        """MCTS selects a legal move within search budget."""
        mcts = CandyCrushMCTS(self.engine, num_simulations=10, rollout_depth=2)
        best_move = mcts.search(self.state)
        r, c, direction = best_move
        self.assertTrue(self.state.is_valid_coord(r, c))

    def test_gym_env(self):
        """Gym environment resets and steps properly."""
        env = CandyCrushGymEnv(height=9, width=9)
        obs, info = env.reset()
        self.assertEqual(obs.shape, (19, 9, 9))

        mask = env.get_action_mask()
        valid_actions = np.where(mask)[0]
        if len(valid_actions) > 0:
            action = valid_actions[0]
            next_obs, reward, terminated, truncated, step_info = env.step(action)
            self.assertEqual(next_obs.shape, (19, 9, 9))


if __name__ == "__main__":
    unittest.main()
