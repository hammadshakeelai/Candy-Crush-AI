"""
Robust Visual Candy, Special Item, and Blocker Classifier for Live Candy Crush Saga.
Combines HSV color space histograms, Sobel frequency analysis, and morphological moments
to accurately perceive the commercial game under varying lighting and candy animations.
"""

from typing import Tuple, List, Optional
import numpy as np
import cv2
from core.candy_board import CandyColor, SpecialType, BlockerType, BoardState
from core.reverse_engine import ReverseEngineState, SpecialCandy, Blocker, LevelType


class RobustTileClassifier:
    """
    Classifies 81 tile image crops into full symbolic board representations:
    (CandyColor, SpecialType, BlockerType).
    """

    def __init__(self):
        # Calibrated HSV ranges (H: 0-180, S: 0-255, V: 0-255) for King Candy Crush Saga
        self.color_ranges = [
            (CandyColor.RED, [(0, 100, 100), (9, 255, 255)], [(171, 100, 100), (180, 255, 255)]),
            (CandyColor.ORANGE, [(10, 120, 120), (24, 255, 255)], None),
            (CandyColor.YELLOW, [(25, 120, 120), (38, 255, 255)], None),
            (CandyColor.GREEN, [(39, 90, 80), (85, 255, 255)], None),
            (CandyColor.BLUE, [(86, 100, 90), (130, 255, 255)], None),
            (CandyColor.PURPLE, [(131, 80, 70), (170, 255, 255)], None),
        ]

    def classify_tile(self, tile_img: np.ndarray) -> Tuple[CandyColor, SpecialType, BlockerType]:
        """
        Classifies a single cropped tile image.
        Returns: (CandyColor, SpecialType, BlockerType)
        """
        if tile_img is None or tile_img.size == 0 or tile_img.shape[0] < 5 or tile_img.shape[1] < 5:
            return CandyColor.EMPTY, SpecialType.NORMAL, BlockerType.NONE

        h, w = tile_img.shape[:2]
        total_pixels = h * w
        hsv = cv2.cvtColor(tile_img, cv2.COLOR_BGR2HSV)

        # 1. Detect Chocolate Blocker
        # Chocolate has low-to-medium value, warm brown hue
        choc_mask = cv2.inRange(hsv, np.array([8, 60, 30]), np.array([22, 210, 110]))
        if cv2.countNonZero(choc_mask) > total_pixels * 0.45:
            return CandyColor.EMPTY, SpecialType.NORMAL, BlockerType.CHOCOLATE

        # 2. Detect Frosting / Ice Blocker
        # Frosting is white/pale light blue, low saturation, high value
        frost_mask = cv2.inRange(hsv, np.array([0, 0, 175]), np.array([180, 55, 255]))
        if cv2.countNonZero(frost_mask) > total_pixels * 0.50:
            return CandyColor.EMPTY, SpecialType.NORMAL, BlockerType.FROSTING

        # 3. Detect Color Bomb (Chocolate sphere with rainbow sprinkles)
        # Check center core brightness: dark chocolate core (< 65 V) with colorful speckles
        center_crop = hsv[int(h * 0.25):int(h * 0.75), int(w * 0.25):int(w * 0.75)]
        mean_v = np.mean(center_crop[:, :, 2])
        if mean_v < 65:
            # Check for colorful sprinkles (saturation > 100 in the dark core)
            high_sat = cv2.inRange(center_crop, np.array([0, 100, 60]), np.array([180, 255, 255]))
            if cv2.countNonZero(high_sat) > (center_crop.shape[0] * center_crop.shape[1] * 0.08):
                return CandyColor.EMPTY, SpecialType.COLOR_BOMB, BlockerType.NONE

        # 4. Candy Color Identification
        best_color = CandyColor.EMPTY
        max_pixels = 0

        for color_enum, r1, r2 in self.color_ranges:
            mask = cv2.inRange(hsv, np.array(r1[0]), np.array(r1[1]))
            if r2 is not None:
                mask2 = cv2.inRange(hsv, np.array(r2[0]), np.array(r2[1]))
                mask = cv2.bitwise_or(mask, mask2)
            count = cv2.countNonZero(mask)
            if count > max_pixels:
                max_pixels = count
                best_color = color_enum

        # If too few colored pixels match, it's either an empty cell or void board space
        if max_pixels < (total_pixels * 0.12):
            return CandyColor.EMPTY, SpecialType.NORMAL, BlockerType.NONE

        # 5. Detect Special Candy Types (Striped H, Striped V, Wrapped)
        special = self._detect_special(tile_img)

        return best_color, special, BlockerType.NONE

    def _detect_special(self, tile_img: np.ndarray) -> SpecialType:
        """
        Analyzes directional Sobel gradients to differentiate:
        - Normal Candies
        - Striped Vertical Candies
        - Striped Horizontal Candies
        - Wrapped Candies (high gradient energy in both directions)
        """
        gray = cv2.cvtColor(tile_img, cv2.COLOR_BGR2GRAY)

        sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)

        abs_x = np.abs(sobel_x)
        abs_y = np.abs(sobel_y)

        energy_x = float(np.mean(abs_x))
        energy_y = float(np.mean(abs_y))

        # Vertical stripes produce intense horizontal gradients
        if energy_x > energy_y * 1.75 and energy_x > 22.0:
            return SpecialType.STRIPED_VERTICAL

        # Horizontal stripes produce intense vertical gradients
        if energy_y > energy_x * 1.75 and energy_y > 22.0:
            return SpecialType.STRIPED_HORIZONTAL

        # Wrapped candy: intense high-frequency texture in both directions (shiny wrapper pinch)
        if energy_x > 28.0 and energy_y > 28.0 and abs(energy_x - energy_y) < 10.0:
            return SpecialType.WRAPPED

        return SpecialType.NORMAL

    def classify_board(self, tiles_matrix: List[List[np.ndarray]]) -> BoardState:
        """Converts 2D matrix of tile crops into standard BoardState."""
        rows = len(tiles_matrix)
        cols = len(tiles_matrix[0]) if rows > 0 else 0
        state = BoardState(rows, cols)

        for r in range(rows):
            for c in range(cols):
                color, special, blocker = self.classify_tile(tiles_matrix[r][c])
                state.colors[r, c] = color
                state.specials[r, c] = special
                state.blockers[r, c] = blocker

        return state

    def to_reverse_engine_state(self, board_state: BoardState, moves_left: int = 30) -> ReverseEngineState:
        """
        Bridges visual BoardState into ReverseEngineState required by AlphaCandyTitanAgent (79.3M).
        """
        from core.reverse_engine import LevelConfig, LevelType
        rows, cols = board_state.colors.shape
        cfg = LevelConfig(
            level_id=999,
            level_type=LevelType.JELLY,
            moves_limit=moves_left,
            height=rows,
            width=cols,
            num_colors=6,
        )
        engine_state = ReverseEngineState(cfg)
        engine_state.moves_left = moves_left

        # Mapping enums
        color_map = {
            CandyColor.RED: 1,
            CandyColor.BLUE: 2,
            CandyColor.GREEN: 3,
            CandyColor.YELLOW: 4,
            CandyColor.ORANGE: 5,
            CandyColor.PURPLE: 6,
            CandyColor.EMPTY: 0,
        }

        special_map = {
            SpecialType.NORMAL: SpecialCandy.NONE,
            SpecialType.STRIPED_HORIZONTAL: SpecialCandy.STRIPED_HORIZONTAL,
            SpecialType.STRIPED_VERTICAL: SpecialCandy.STRIPED_VERTICAL,
            SpecialType.WRAPPED: SpecialCandy.WRAPPED,
            SpecialType.COLOR_BOMB: SpecialCandy.COLOR_BOMB,
        }

        blocker_map = {
            BlockerType.NONE: Blocker.NONE,
            BlockerType.CHOCOLATE: Blocker.CHOCOLATE,
            BlockerType.FROSTING: Blocker.FROSTING,
            BlockerType.LICORICE: Blocker.LICORICE,
        }

        for r in range(rows):
            for c in range(cols):
                color_val = board_state.colors[r, c]
                special_val = board_state.specials[r, c]
                blocker_val = board_state.blockers[r, c]

                engine_state.colors[r, c] = color_map.get(color_val, 0)
                engine_state.specials[r, c] = special_map.get(special_val, SpecialCandy.NONE)
                engine_state.blockers[r, c] = blocker_map.get(blocker_val, Blocker.NONE)

                # Set active grid (active if not completely empty/void)
                if color_val != CandyColor.EMPTY or blocker_val != BlockerType.NONE or special_val != SpecialType.NORMAL:
                    engine_state.active_grid[r, c] = True
                else:
                    engine_state.active_grid[r, c] = False

        return engine_state
