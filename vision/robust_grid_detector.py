"""
Robust Candy Crush Grid Localization, Tile Slicing, and Temporal Settling Engine.
Supports persistent JSON calibration, automatic contour-based localization,
and multi-frame temporal settling verification.
"""

import os
import json
import time
from pathlib import Path
from typing import Tuple, List, Optional, Dict, Any
import numpy as np
import cv2

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "board_roi.json"


class RobustGridDetector:
    """
    Locates the Candy Crush playing board from screen captures,
    slices tiles, computes screen pixel coordinates for moves,
    and detects cascade settling.
    """

    def __init__(self, config_path: Optional[Path] = None, default_grid_size: Tuple[int, int] = (9, 9)):
        self.config_path = config_path or DEFAULT_CONFIG_PATH
        self.rows, self.cols = default_grid_size
        self.cached_roi: Optional[Tuple[int, int, int, int]] = None
        self.last_frames: List[np.ndarray] = []
        self.load_calibration()

    def load_calibration(self) -> bool:
        """Loads saved board ROI from JSON config if available."""
        if self.config_path.exists():
            try:
                with open(self.config_path, "r") as f:
                    data = json.load(f)
                self.cached_roi = (data["x"], data["y"], data["w"], data["h"])
                self.rows = data.get("rows", self.rows)
                self.cols = data.get("cols", self.cols)
                print(f"[GridDetector] Loaded board ROI from {self.config_path}: {self.cached_roi} ({self.rows}x{self.cols})")
                return True
            except Exception as e:
                print(f"[GridDetector] Failed to load calibration: {e}")
        return False

    def save_calibration(self, x: int, y: int, w: int, h: int, rows: int = 9, cols: int = 9):
        """Saves verified board ROI to JSON config."""
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "x": int(x),
            "y": int(y),
            "w": int(w),
            "h": int(h),
            "rows": int(rows),
            "cols": int(cols),
            "updated_at": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        with open(self.config_path, "w") as f:
            json.dump(data, f, indent=2)
        self.cached_roi = (int(x), int(y), int(w), int(h))
        self.rows = rows
        self.cols = cols
        print(f"[GridDetector] Saved calibration to {self.config_path}")

    def detect_board_region(self, frame: np.ndarray) -> Tuple[int, int, int, int]:
        """
        Locates the board bounding box (x, y, w, h).
        Uses cached ROI if available; otherwise applies automatic edge & contour search,
        with portrait/landscape heuristics as fallback.
        """
        if self.cached_roi is not None:
            return self.cached_roi

        img_h, img_w = frame.shape[:2]

        # 1. Automatic Contour & Contrast Search
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 40, 120)

        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        best_box = None
        best_area = 0

        # Looking for a large nearly-square region (aspect ratio 0.8 to 1.25)
        # occupying 15% to 80% of screen area
        min_area = (img_w * img_h) * 0.12
        max_area = (img_w * img_h) * 0.85

        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            area = w * h
            if min_area <= area <= max_area:
                aspect = float(w) / float(h)
                if 0.75 <= aspect <= 1.30 and area > best_area:
                    best_area = area
                    best_box = (x, y, w, h)

        if best_box is not None:
            return best_box

        # 2. Geometric Layout Fallbacks
        if img_h > img_w:
            # Portrait (typical phone / mobile aspect ratio)
            board_w = int(img_w * 0.92)
            board_h = board_w
            board_x = int((img_w - board_w) / 2)
            board_y = int(img_h * 0.28)
        else:
            # Landscape (desktop browser or tablet)
            board_h = int(img_h * 0.75)
            board_w = board_h
            board_x = int((img_w - board_w) / 2)
            board_y = int(img_h * 0.15)

        return (board_x, board_y, board_w, board_h)

    def extract_tiles(self, frame: np.ndarray, roi: Optional[Tuple[int, int, int, int]] = None) -> List[List[np.ndarray]]:
        """
        Extracts cropped image patches for each tile in the grid.
        Returns 2D list of shape [rows][cols].
        """
        bx, by, bw, bh = roi or self.detect_board_region(frame)
        board_crop = frame[by:by + bh, bx:bx + bw]

        cell_h = bh // self.rows
        cell_w = bw // self.cols

        tiles = []
        pad_y = max(2, int(cell_h * 0.14))
        pad_x = max(2, int(cell_w * 0.14))

        for r in range(self.rows):
            row_tiles = []
            for c in range(self.cols):
                y1 = r * cell_h + pad_y
                y2 = (r + 1) * cell_h - pad_y
                x1 = c * cell_w + pad_x
                x2 = (c + 1) * cell_w - pad_x

                tile_crop = board_crop[y1:y2, x1:x2]
                row_tiles.append(tile_crop)
            tiles.append(row_tiles)

        return tiles

    def get_cell_center_pixel(self, r: int, c: int, roi: Optional[Tuple[int, int, int, int]] = None) -> Tuple[int, int]:
        """Calculates global screen (x, y) coordinates for the center of cell (r, c)."""
        bx, by, bw, bh = roi or self.cached_roi or (0, 0, 100, 100)
        cell_w = bw // self.cols
        cell_h = bh // self.rows
        center_x = bx + int((c + 0.5) * cell_w)
        center_y = by + int((r + 0.5) * cell_h)
        return center_x, center_y

    def is_board_settled(self, current_frame: np.ndarray, roi: Optional[Tuple[int, int, int, int]] = None,
                         threshold: float = 2.0, min_stable_frames: int = 2) -> bool:
        """
        Determines if cascades, matches, and refill animations have completed.
        Requires board pixels to remain stationary across multiple frames.
        """
        bx, by, bw, bh = roi or self.detect_board_region(current_frame)
        board_crop = current_frame[by:by + bh, bx:bx + bw]
        gray = cv2.cvtColor(board_crop, cv2.COLOR_BGR2GRAY)

        if len(self.last_frames) == 0:
            self.last_frames.append(gray)
            return False

        # Compare with the most recent frame
        prev = self.last_frames[-1]
        if prev.shape != gray.shape:
            self.last_frames = [gray]
            return False

        diff = cv2.absdiff(prev, gray)
        mean_diff = float(np.mean(diff))

        self.last_frames.append(gray)
        if len(self.last_frames) > 5:
            self.last_frames.pop(0)

        # Check if difference is negligible (< threshold)
        if mean_diff < threshold:
            # We also check previous differences in history if available
            return len(self.last_frames) >= min_stable_frames
        else:
            # Reset stable frame count on large delta
            self.last_frames = [gray]
            return False
