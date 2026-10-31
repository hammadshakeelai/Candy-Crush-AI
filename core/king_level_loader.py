"""
King Level Database Loader & Parser.
Loads official Candy Crush Saga levels (1 to 23,355) directly from King's JSON configurations
and the Master SQLite level database.
"""

import json
import os
import sqlite3
from typing import Dict, Any, Optional, List
import numpy as np

from .reverse_engine import (
    LevelConfig, LevelType, Blocker, Ingredient, SpecialCandy
)


class KingLevelLoader:
    """
    Parses King's official level blueprints into LevelConfig objects.
    Seamlessly supports all 23,355 levels via master SQLite database or raw JSON archives.
    """

    CONFIG_FILE = os.environ.get(
        "KING_CONFIG_PATH",
        "data/candycrush_repo/CCS_Scripts/json_extract/sample/configs_v1.208.json"
    )
    SQLITE_DB = os.environ.get(
        "KING_SQLITE_PATH",
        "data/candycrush_master_levels.sqlite"
    )

    def __init__(self, config_path: Optional[str] = None, sqlite_path: Optional[str] = None):
        # Resolve paths at construction time so env vars set after import (e.g. on Kaggle) are honoured
        self.config_path = config_path or os.environ.get("KING_CONFIG_PATH") or self.CONFIG_FILE
        self.sqlite_path = sqlite_path or os.environ.get("KING_SQLITE_PATH") or self.SQLITE_DB
        if not os.path.exists(self.sqlite_path):
            found = self._discover_file("candycrush_master_levels.sqlite")
            if found:
                self.sqlite_path = found
        if not os.path.exists(self.config_path):
            found = self._discover_file("configs_v1.208.json")
            if found:
                self.config_path = found
            else:
                found_bundle = self._discover_file("king_benchmark_500.json") or self._discover_file("sample_levels.json")
                if found_bundle:
                    self.config_path = found_bundle
        self._levels_cache: Dict[int, Dict[str, Any]] = {}
        self._loaded = False

    @staticmethod
    def _discover_file(filename: str) -> Optional[str]:
        """Searches common data roots (local data/ dir and Kaggle /kaggle/input) for a file."""
        for root_dir in ("data", "/kaggle/input", "/kaggle/working"):
            if not os.path.isdir(root_dir):
                continue
            for root, _dirs, files in os.walk(root_dir):
                if filename in files:
                    return os.path.join(root, filename)
        return None

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except Exception as e:
                print(f"[KingLevelLoader] WARNING: could not read {self.config_path}: {e}")
                data = []
            # Only raw King config lists (entries containing gameData/tileMap) are usable here.
            # Summary-only files (e.g. candycrush_master_levels.json, a dict) are skipped.
            if isinstance(data, list):
                for item in data:
                    if not isinstance(item, dict) or "gameData" not in item:
                        continue
                    lvl_num = item.get("totalLevel", item.get("level"))
                    if lvl_num is not None:
                        try:
                            self._levels_cache[int(lvl_num)] = item
                        except (ValueError, TypeError):
                            continue
                print(f"[KingLevelLoader] Successfully indexed {len(self._levels_cache)} official King levels from JSON.")
            else:
                print(f"[KingLevelLoader] {self.config_path} is not a raw King config list; using SQLite only.")
        self._loaded = True

    def get_total_levels(self) -> int:
        if os.path.exists(self.sqlite_path):
            try:
                conn = sqlite3.connect(self.sqlite_path)
                c = conn.cursor()
                c.execute("SELECT count(*) FROM levels")
                count = c.fetchone()[0]
                conn.close()
                return count
            except Exception:
                pass
        self._ensure_loaded()
        return len(self._levels_cache)

    def _parse_game_data(self, level_number: int, gd: Dict[str, Any]) -> LevelConfig:
        """Parses a King gameData dictionary into a LevelConfig."""
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
            level_type = LevelType.JELLY
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
        cannon_specials = np.zeros((h, w), dtype=np.int32)

        color_token_map = {"055": 1, "056": 2, "057": 3, "058": 4, "059": 5, "060": 6}
        ingredient_count = 0

        for r in range(raw_h):
            for c in range(raw_w):
                cell_str = tile_map[r][c]
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

                # Blockers: Chocolate ('009'), Licorice ('008', '017'), Frosting ('007', '021', '022')
                # Note: '010' is the Drop-Down Ingredient Exit arrow, NOT Chocolate!
                if "009" in tokens:
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
                elif "047" in tokens or "050" in tokens:
                    specials[r, c] = SpecialCandy.WRAPPED
                elif "048" in tokens or "049" in tokens or "054" in tokens:
                    specials[r, c] = SpecialCandy.COLOR_BOMB
                elif "036" in tokens:
                    # King Candy Frog / Jelly Fish token: homing clearance for isolated islands
                    specials[r, c] = SpecialCandy.JELLY_FISH

                # Special Candy Cannons ('071' Striped Cannon, '072' Wrapped Cannon, or preset special dispenser)
                if "071" in tokens or "072" in tokens:
                    if specials[r, c] != SpecialCandy.NONE:
                        cannon_specials[r, c] = specials[r, c]
                    elif "072" in tokens:
                        cannon_specials[r, c] = SpecialCandy.WRAPPED
                    else:
                        cannon_specials[r, c] = SpecialCandy.STRIPED_HORIZONTAL
                elif "026" in tokens and specials[r, c] != SpecialCandy.NONE:
                    cannon_specials[r, c] = specials[r, c]

                # Ingredients: Cherry ('018'), Hazelnut ('020' when drop mode)
                if "018" in tokens:
                    ingredients[r, c] = Ingredient.CHERRY
                    ingredient_count += 1
                elif "020" in tokens and level_type == LevelType.INGREDIENTS:
                    ingredients[r, c] = Ingredient.HAZELNUT
                    ingredient_count += 1

        target_ing = max(ingredient_count, 1) if level_type == LevelType.INGREDIENTS else 0

        # Calibrate King "Order" mode levels that contain 0 jellies on the board:
        # In ReverseCandyEngine, regular candy clears award 60 pts (without +1,000 jelly bonus),
        # so Order levels with 0 jellies map to LevelType.SCORE with target_score calibrated
        # to the order item quantity (`_itemsToOrder`) and move budget.
        if "Order" in game_mode and int(np.sum(jellies)) == 0:
            level_type = LevelType.SCORE
            order_items = gd.get("_itemsToOrder", [])
            order_qty = sum(int(item.get("quantity", 0)) for item in order_items if isinstance(item, dict))
            calibrated = max(6000, min(max(order_qty, 15) * 120, move_limit * 850))
            target_score = min(target_score, calibrated)

        # Advanced King mechanics extraction
        portals = gd.get("portals", [])
        cannons = gd.get("cannons", [])
        ammo_cannons = gd.get("ammoCannons", [])
        frog_stomach = gd.get("frogStomachSize", 0) or 0
        evil_spawner = None
        if gd.get("evilSpawnerElements"):
            evil_spawner = {
                "elements": gd.get("evilSpawnerElements", [2]),
                "interval": gd.get("evilSpawnerInterval", 3),
                "amount": gd.get("evilSpawnerAmount", 1),
            }

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
            cannon_specials=cannon_specials,
            target_ingredients=target_ing,
            portals=portals,
            cannons=cannons,
            ammo_cannons=ammo_cannons,
            evil_spawner_config=evil_spawner,
            frog_stomach_size=frog_stomach
        )

    def load_level(self, level_number: int) -> LevelConfig:
        """
        Extracts LevelConfig for any King level number from 1 to 23,355.
        """
        self._ensure_loaded()

        # 1. Check in-memory JSON cache
        if level_number in self._levels_cache:
            entry = self._levels_cache[level_number]
            gd = entry.get("gameData", {})
            cfg = self._parse_game_data(level_number, gd)
            # Reconcile with Master SQLite Database if available (for patched King move limits)
            if os.path.exists(self.sqlite_path):
                try:
                    conn = sqlite3.connect(self.sqlite_path)
                    c = conn.cursor()
                    c.execute("SELECT move_limit FROM levels WHERE level_id = ?", (level_number,))
                    r = c.fetchone()
                    conn.close()
                    if r and r[0] and r[0] > cfg.moves_limit:
                        cfg.moves_limit = r[0]
                except Exception:
                    pass
            return cfg

        # 2. Check Master SQLite Database (spans all 23,355 levels)
        if os.path.exists(self.sqlite_path):
            conn = sqlite3.connect(self.sqlite_path)
            c = conn.cursor()
            c.execute(
                "SELECT game_mode, move_limit, num_colours, score_target_1, has_full_raw_config, game_data_json FROM levels WHERE level_id = ?",
                (level_number,)
            )
            row = c.fetchone()
            conn.close()

            if row:
                raw_mode, move_limit, num_colors, score_tgt, has_raw, gd_json = row
                if has_raw and gd_json:
                    try:
                        gd = json.loads(gd_json)
                        return self._parse_game_data(level_number, gd)
                    except Exception:
                        pass

                # Procedural generation using official King parameters
                from .procedural_levels import ProceduralKingGenerator
                gen = ProceduralKingGenerator()
                cfg = gen.generate_level(level_number)
                cfg.moves_limit = move_limit or cfg.moves_limit
                cfg.num_colors = num_colors or cfg.num_colors
                if score_tgt:
                    cfg.target_score = score_tgt

                if raw_mode:
                    if "Light" in str(raw_mode) or "Jelly" in str(raw_mode):
                        cfg.level_type = LevelType.JELLY
                    elif "Drop" in str(raw_mode):
                        cfg.level_type = LevelType.INGREDIENTS

                return cfg

        raise KeyError(f"Level {level_number} not found in database or configurations.")
