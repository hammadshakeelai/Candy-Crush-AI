"""
Candy Crush Grid Localization, Patch Slicing, and Animation Settling Detector.
"""

from typing import Tuple, List, Optional
import numpy as np
import cv2
import time


class GridDetector:
    """
    Locates the 9x9 Candy Crush board from a full-resolution screen image,
    crops individual tiles, and checks when animations have settled.
    """

    def __init__(self, height: int = 9, width: int = 9):
        self.height = height
        self.width = width
        self.last_frame: Optional[np.ndarray] = None

    def detect_board_region(self, frame: np.ndarray) -> Tuple[int, int, int, int]:
        """
        Calculates board bounding box (x, y, w, h).
        In mobile Candy Crush (portrait 1080x1920 or 1080x2400):
        The board is centered horizontally with typical 1:1 aspect ratio.
        """
        img_h, img_w = frame.shape[:2]
        
        # Standard mobile portrait Candy Crush layout calibration:
        # Board starts roughly at 30% from the top and spans ~90% of screen width.
        board_w = int(img_w * 0.92)
        board_h = board_w  # 9x9 square grid
        board_x = int((img_w - board_w) / 2)
        board_y = int(img_h * 0.28)

        return board_x, board_y, board_w, board_h

    def extract_tiles(self, frame: np.ndarray) -> List[List[np.ndarray]]:
        """
        Crops the 9x9 board into an array of 81 tile images.
        """
        bx, by, bw, bh = self.detect_board_region(frame)
        board_crop = frame[by:by + bh, bx:bx + bw]

        cell_h = bh // self.height
        cell_w = bw // self.width

        tiles = []
        for r in range(self.height):
            row_tiles = []
            for c in range(self.width):
                y1 = r * cell_h
                y2 = (r + 1) * cell_h
                x1 = c * cell_w
                x2 = (c + 1) * cell_w
                
                # Inset padding by 15% to avoid borders/grid lines
                pad_y = int(cell_h * 0.15)
                pad_x = int(cell_w * 0.15)
                tile_crop = board_crop[y1 + pad_y:y2 - pad_y, x1 + pad_x:x2 - pad_x]
                row_tiles.append(tile_crop)
            tiles.append(row_tiles)

        return tiles

    def is_board_settled(self, current_frame: np.ndarray, threshold: float = 0.98) -> bool:
        """
        Determines if candy cascades and animations have finished settling.
        Compares structural similarity/pixel correlation with previous frame.
        """
        if self.last_frame is None:
            self.last_frame = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
            return False

        gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
        bx, by, bw, bh = self.detect_board_region(gray)
        crop1 = self.last_frame[by:by + bh, bx:bx + bw]
        crop2 = gray[by:by + bh, bx:bx + bw]

        # Calculate absolute difference
        diff = cv2.absdiff(crop1, crop2)
        mean_diff = np.mean(diff)

        # Update last frame
        self.last_frame = gray

        # If mean pixel shift is below threshold (< 2.5 intensity change), board is settled
        return mean_diff < 2.5
