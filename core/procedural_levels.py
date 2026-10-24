"""
Procedural Level Generator for Infinite Candy Crush Saga Levels (up to 25,000+).
Emulates King's weekly automated level generation pipeline and difficulty scaling curves.
"""

from typing import Tuple, Optional
import numpy as np
import random
from .reverse_engine import (
    LevelConfig, LevelType, Blocker, Ingredient
)


class ProceduralKingGenerator:
    """
    Generates realistic King-style Candy Crush levels from Level 10,296 to 25,000+.
    """

    def __init__(self, seed: Optional[int] = None):
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

    def generate_level(self, level_number: int) -> LevelConfig:
        """
        Synthesizes a realistic Candy Crush Saga level configuration for level_number.
        """
        rng = random.Random(level_number * 31337)
        np_rng = np.random.RandomState(level_number * 31337)

        # 1. Determine Level Type by King's cycle
        cycle = level_number % 4
        if cycle == 0:
            level_type = LevelType.JELLY
        elif cycle == 1:
            level_type = LevelType.INGREDIENTS
        elif cycle == 2:
            level_type = LevelType.BLOCKERS
        else:
            level_type = LevelType.SCORE

        # 2. Number of Colors (scales 4 to 6)
        num_colors = 4 if level_number % 5 == 0 else (5 if level_number % 2 == 0 else 6)

        # 3. Move Limit (typically 20 to 35 in modern high-number levels)
        moves_limit = rng.randint(22, 32)

        # 4. Target Score (logarithmic scaling with level)
        base_score = int(15000 * (1.0 + np.log10(max(1, level_number)) ** 1.5))
        target_score = (base_score // 5000) * 5000

        # 5. Grid Active Mask (9x9 with corner cuts / realistic shape)
        h, w = 9, 9
        active_grid = np.ones((h, w), dtype=bool)

        # High levels have irregular shapes (cut corners or central void)
        if level_number > 500:
            # Cut extreme corners
            for r in [0, h - 1]:
                for c in [0, w - 1]:
                    if rng.random() < 0.6:
                        active_grid[r, c] = False

        # 6. Obstacles & Jellies
        jellies = np.zeros((h, w), dtype=np.int32)
        blockers = np.zeros((h, w), dtype=np.int32)
        ingredients = np.zeros((h, w), dtype=np.int32)
        target_ingredients = 0

        if level_type == LevelType.JELLY:
            # High-level jelly distribution
            jelly_density = min(0.75, 0.35 + (level_number / 40000.0))
            for r in range(h):
                for c in range(w):
                    if active_grid[r, c] and np_rng.rand() < jelly_density:
                        # 2 layers for higher difficulty
                        jellies[r, c] = 2 if np_rng.rand() < 0.6 else 1

            # Scatter some frostings or licorice
            for r in range(2, 7):
                for c in range(2, 7):
                    if active_grid[r, c] and np_rng.rand() < 0.2:
                        blockers[r, c] = Blocker.FROSTING

        elif level_type == LevelType.INGREDIENTS:
            target_ingredients = rng.randint(1, 3)
            # Place initial ingredient at row 0 of a valid column
            valid_cols = [c for c in range(w) if active_grid[0, c]]
            if valid_cols:
                ing_col = rng.choice(valid_cols)
                ingredients[0, ing_col] = Ingredient.CHERRY

            # Add bottom blockers (chocolate or licorice barriers)
            for c in range(w):
                if active_grid[h - 1, c] and np_rng.rand() < 0.5:
                    blockers[h - 1, c] = Blocker.CHOCOLATE

        elif level_type == LevelType.BLOCKERS:
            # Dense chocolate / frosting outbreak
            for r in range(3, 8):
                for c in range(1, 8):
                    if active_grid[r, c] and np_rng.rand() < 0.35:
                        blockers[r, c] = Blocker.CHOCOLATE

        return LevelConfig(
            level_id=level_number,
            level_type=level_type,
            moves_limit=moves_limit,
            target_score=target_score,
            height=h,
            width=w,
            num_colors=num_colors,
            active_grid=active_grid,
            initial_jellies=jellies,
            initial_blockers=blockers,
            initial_ingredients=ingredients,
            target_ingredients=target_ingredients
        )
