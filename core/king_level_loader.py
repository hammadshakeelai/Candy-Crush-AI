"""
King Level Database Loader & Parser.
Loads official Candy Crush Saga levels (1 to 10,299) directly from King's JSON configurations.
"""

import json
import os
from typing import Dict, Any, Optional, List
import numpy as np

from .reverse_engine import (
    LevelConfig, LevelType, Blocker, Ingredient, SpecialCandy
)


class KingLevelLoader:
    """
    Parses King's official level blueprints into LevelConfig objects.
    """

    CONFIG_FILE = "data/candycrush_repo/CCS_Scripts/json_extract/sample/configs_v1.208.json"

    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path or self.CONFIG_FILE
        self._levels_cache: Dict[int, Dict[str, Any]] = {}
        self._loaded = False

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"King configs file not found at: {self.config_path}")
        with open(self.config_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        for item in data:
            lvl_num = item.get("totalLevel", item.get("level"))
            if lvl_num is not None:
                try:
                    self._levels_cache[int(lvl_num)] = item
                except (ValueError, TypeError):
                    continue
        self._loaded = True
        print(f"[KingLevelLoader] Successfully indexed {len(self._levels_cache)} official King levels.")

    def get_total_levels(self) -> int:
        self._ensure_loaded()
        return len(self._levels_cache)

    def load_level(self, level_number: int) -> LevelConfig:
        """
        Extracts LevelConfig for a specific King level number (e.g. 1 to 10299).
        """
        self._ensure_loaded()
        if level_number not in self._levels_cache:
            raise KeyError(f"Level {level_number} not found in database.")

        entry = self._levels_cache[level_number]
        gd = entry.get("gameData", {})

        move_limit = gd.get("moveLimit", 30)
        num_colors = gd.get("numberOfColours", 5)
        score_targets = gd.get("scoreTargets", [10000, 20000, 30000])
        target_score = score_targets[0] if score_targets else 10000
        game_mode = gd.get("gameModeName", "Classic moves")

        # Map King game modes to LevelType
        if "Light up" in game_mode or "Jelly" in game_mode:
            level_type = LevelType.JELLY
        elif "Drop" in game_mode:
            level_type = LevelType.INGREDIENTS
        elif "Order" in game_mode:
            level_type = LevelType.JELLY  # Order levels frequently revolve around clearing jellies/orders
        else:
            level_type = LevelType.SCORE

        tile_map = gd.get("tileMap", [])
        raw_h = len(tile_map)
        raw_w = len(tile_map[0]) if raw_h > 0 else 0

        # Enforce standard 9x9 bounding box
        h = max(9, raw_h)
        w = max(9, raw_w)

        active_grid = np.zeros((h, w), dtype=bool)
        jellies = np.zeros((h, w), dtype=np.int32)
        blockers = np.zeros((h, w), dtype=np.int32)
        ingredients = np.zeros((h, w), dtype=np.int32)
        colors = np.zeros((h, w), dtype=np.int32)
        specials = np.zeros((h, w), dtype=np.int32)

        color_token_map = {"055": 1, "056": 2, "057": 3, "058": 4, "059": 5, "060": 6}

        ingredient_count = 0

        for r in range(raw_h):
            for c in range(raw_w):
                cell_str = tile_map[r][c]
                # Split cell_str into 3-digit tokens
                tokens = [cell_str[i:i+3] for i in range(0, len(cell_str), 3)]

                if "000" in tokens or len(tokens) == 0:
                    active_grid[r, c] = False
                    continue

                active_grid[r, c] = True

                # Jelly Layer 1 ('003') and Layer 2 ('004')
                if "003" in tokens:
                    jellies[r, c] = 1
                elif "004" in tokens:
                    jellies[r, c] = 2

                # Blockers: Chocolate ('009', '010'), Licorice ('008', '017'), Frosting ('007', '021', '022')
                if "009" in tokens or "010" in tokens:
                    blockers[r, c] = Blocker.CHOCOLATE
                elif "008" in tokens or "017" in tokens:
                    blockers[r, c] = Blocker.LICORICE
                elif any(tok in tokens for tok in ["007", "021", "022"]):
                    blockers[r, c] = Blocker.FROSTING

                # Fixed candy color
                for tok, col in color_token_map.items():
                    if tok in tokens:
                        colors[r, c] = col
                        break

                # Pre-placed Special Candies
                if "045" in tokens:
                    specials[r, c] = SpecialCandy.STRIPED_HORIZONTAL
                elif "046" in tokens:
                    specials[r, c] = SpecialCandy.STRIPED_VERTICAL
                elif "050" in tokens:
                    specials[r, c] = SpecialCandy.WRAPPED
                elif "054" in tokens:
                    specials[r, c] = SpecialCandy.COLOR_BOMB

                # Ingredients: Cherry ('018'), Hazelnut ('020' when drop mode)
                if "018" in tokens:
                    ingredients[r, c] = Ingredient.CHERRY
                    ingredient_count += 1
                elif "020" in tokens and level_type == LevelType.INGREDIENTS:
                    ingredients[r, c] = Ingredient.HAZELNUT
                    ingredient_count += 1

        target_ing = max(ingredient_count, 1) if level_type == LevelType.INGREDIENTS else 0

        return LevelConfig(
            level_id=level_number,
            level_type=level_type,
            moves_limit=move_limit,
            target_score=target_score,
            height=h,
            width=w,
            num_colors=num_colors,
            active_grid=active_grid,
            initial_jellies=jellies,
            initial_blockers=blockers,
            initial_ingredients=ingredients,
            initial_colors=colors,
            initial_specials=specials,
            target_ingredients=target_ing
        )
