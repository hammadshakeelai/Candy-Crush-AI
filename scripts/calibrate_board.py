#!/usr/bin/env python3
"""
Board ROI Calibration Tool for Real Candy Crush Saga.
Captures screen, auto-detects or accepts manual bounding box coordinates,
draws an alignment verification overlay, and saves to config/board_roi.json.
"""

import os
import sys
import argparse
from pathlib import Path
import cv2
import numpy as np

# Ensure project root in sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from automation.display_utils import DesktopScreenCapture, setup_display_environment
from vision.robust_grid_detector import RobustGridDetector


def calibrate(auto: bool = True, x: int = None, y: int = None, w: int = None, h: int = None, rows: int = 9, cols: int = 9):
    setup_display_environment()
    cap = DesktopScreenCapture()
    frame = cap.grab_screen()
    cap.close()

    detector = RobustGridDetector()

    if x is not None and y is not None and w is not None and h is not None:
        roi = (x, y, w, h)
        print(f"[Calibrate] Using explicit coordinates: {roi}")
    else:
        # Auto-detect board
        detector.cached_roi = None
        roi = detector.detect_board_region(frame)
        print(f"[Calibrate] Auto-detected board region: {roi}")

    bx, by, bw, bh = roi
    detector.save_calibration(bx, by, bw, bh, rows=rows, cols=cols)

    # Generate visual verification preview
    preview = frame.copy()
    # Draw outer bounding box
    cv2.rectangle(preview, (bx, by), (bx + bw, by + bh), (0, 255, 0), 3)

    cell_w = bw // cols
    cell_h = bh // rows

    # Draw grid lines and cell centers
    for r in range(rows):
        for c in range(cols):
            cx = bx + int((c + 0.5) * cell_w)
            cy = by + int((r + 0.5) * cell_h)
            cv2.circle(preview, (cx, cy), 5, (0, 0, 255), -1)
            # Draw cell borders
            cv2.rectangle(
                preview,
                (bx + c * cell_w, by + r * cell_h),
                (bx + (c + 1) * cell_w, by + (r + 1) * cell_h),
                (255, 255, 0),
                1
            )

    out_preview = ROOT_DIR / "config" / "board_preview.png"
    out_preview.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_preview), preview)
    print(f"[Calibrate] Calibration verification overlay saved to: {out_preview}")
    print(f"[Calibrate] Board ROI successfully saved: x={bx}, y={by}, w={bw}, h={bh} ({rows}x{cols})")
    return roi


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calibrate Candy Crush Board ROI")
    parser.add_argument("--auto", action="store_true", default=True, help="Auto-detect board region")
    parser.add_argument("-x", type=int, default=None, help="Left X coordinate")
    parser.add_argument("-y", type=int, default=None, help="Top Y coordinate")
    parser.add_argument("-w", type=int, default=None, help="Width in pixels")
    parser.add_argument("-h_px", type=int, default=None, help="Height in pixels")
    parser.add_argument("--rows", type=int, default=9, help="Grid rows (default: 9)")
    parser.add_argument("--cols", type=int, default=9, help="Grid cols (default: 9)")
    args = parser.parse_args()

    calibrate(auto=args.auto, x=args.x, y=args.y, w=args.w, h=args.h_px, rows=args.rows, cols=args.cols)
