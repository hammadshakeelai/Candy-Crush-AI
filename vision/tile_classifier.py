"""
Visual Candy & Blocker Classifier.
Extracts CandyColor and SpecialType from image patches using HSV analysis and texture filters.
"""

from typing import Tuple, List
import numpy as np
import cv2
from core.candy_board import CandyColor, SpecialType, BlockerType, BoardState


class TileClassifier:
    """
    High-speed CV classifier mapping image patches to Candy Crush symbolic state.
    """

    def __init__(self):
        # HSV Color Range Definitions (H in [0, 180], S in [0, 255], V in [0, 255])
        self.color_ranges = [
            (CandyColor.RED, [(0, 120, 120), (10, 255, 255)], [(170, 120, 120), (180, 255, 255)]),
            (CandyColor.ORANGE, [(11, 140, 140), (24, 255, 255)], None),
            (CandyColor.YELLOW, [(25, 140, 140), (38, 255, 255)], None),
            (CandyColor.GREEN, [(39, 100, 100), (85, 255, 255)], None),
            (CandyColor.BLUE, [(86, 120, 120), (130, 255, 255)], None),
            (CandyColor.PURPLE, [(131, 90, 90), (169, 255, 255)], None),
        ]

    def classify_tile(self, tile_img: np.ndarray) -> Tuple[CandyColor, SpecialType, BlockerType]:
        """
        Classifies single tile image patch into (color, special_type, blocker_type).
        """
        if tile_img.size == 0:
            return CandyColor.EMPTY, SpecialType.NORMAL, BlockerType.NONE

        hsv = cv2.cvtColor(tile_img, cv2.COLOR_BGR2HSV)
        h, w = hsv.shape[:2]

        # 1. Check for Color Bomb (Dark circular core with multi-color dots)
        center_crop = hsv[int(h * 0.25):int(h * 0.75), int(w * 0.25):int(w * 0.75)]
        mean_val = np.mean(center_crop[:, :, 2])
        if mean_val < 60:
            return CandyColor.EMPTY, SpecialType.COLOR_BOMB, BlockerType.NONE

        # 2. Match Candy Color by pixel count
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

        # If too few colored pixels matched, tile might be empty or a blocker
        if max_pixels < (h * w * 0.15):
            return CandyColor.EMPTY, SpecialType.NORMAL, BlockerType.NONE

        # 3. Detect Special Candy Type (Striped H, Striped V, Wrapped)
        special = self._detect_special(tile_img)

        return best_color, special, BlockerType.NONE

    def _detect_special(self, tile_img: np.ndarray) -> SpecialType:
        """
        Analyzes directional gradients to detect horizontal/vertical stripes or wrapped wrapping.
        """
        gray = cv2.cvtColor(tile_img, cv2.COLOR_BGR2GRAY)
        
        # Sobel gradients
        sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)

        energy_x = np.mean(np.abs(sobel_x))
        energy_y = np.mean(np.abs(sobel_y))

        # Vertical stripes produce strong horizontal gradients (sobel_x >> sobel_y)
        if energy_x > energy_y * 1.8 and energy_x > 25.0:
            return SpecialType.STRIPED_VERTICAL
        # Horizontal stripes produce strong vertical gradients (sobel_y >> sobel_x)
        if energy_y > energy_x * 1.8 and energy_y > 25.0:
            return SpecialType.STRIPED_HORIZONTAL

        return SpecialType.NORMAL

    def classify_board(self, tiles_matrix: List[List[np.ndarray]]) -> BoardState:
        """
        Converts the 9x9 cropped image patches into a clean BoardState object.
        """
        height = len(tiles_matrix)
        width = len(tiles_matrix[0])
        state = BoardState(height, width)

        for r in range(height):
            for c in range(width):
                color, special, blocker = self.classify_tile(tiles_matrix[r][c])
                state.colors[r, c] = color
                state.specials[r, c] = special
                state.blockers[r, c] = blocker

        return state
