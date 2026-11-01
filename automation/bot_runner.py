#!/usr/bin/env python3
"""
End-to-End Live ADB Bot Runner for Real Candy Crush Saga (Mobile Android / Emulator).
Perceives via screencap -> Classifies via RobustTileClassifier -> Evaluates via AlphaCandy Titan (79.3M) -> Injects Touch Swipes via ADB.
Includes automated life replenishment (5-lives time-leap protocol).
"""

import os
import sys
import time
import argparse
from pathlib import Path
from typing import Optional, Tuple
import cv2
import numpy as np

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from core.candy_board import Direction, DIRECTION_OFFSETS
from core.reverse_engine import Direction as RevDirection, OFFSETS as REV_OFFSETS
from vision.robust_grid_detector import RobustGridDetector
from vision.robust_tile_classifier import RobustTileClassifier
from models.alphacandy_titan_agent import AlphaCandyTitanAgent
from automation.adb_controller import ADBController

CANDY_EMOJIS = {
    1: "🔴", 2: "🔵", 3: "🟢", 4: "🟡", 5: "🟠", 6: "🟣", 0: " ·"
}


class CandyCrushLiveADBBot:
    """Automated agent playing live Candy Crush on real Android devices via ADB."""

    def __init__(self, device_id: Optional[str] = None, checkpoint_path: Optional[str] = None):
        print("=" * 65)
        print("📱 ALPHACANDY TITAN — LIVE REAL GAME ADB MOBILE BOT (79.3M PARAMS)")
        print("=" * 65)
        self.adb = ADBController(device_id=device_id)
        self.detector = RobustGridDetector()
        self.classifier = RobustTileClassifier()
        self.agent = AlphaCandyTitanAgent(checkpoint_path=checkpoint_path)

        self.consecutive_failed_reads = 0
        print("[ADBBot] Initialized and connected to ADB.")

    def play_step(self) -> bool:
        """Executes one full decision cycle: screencap -> classify -> Titan evaluate -> touch swipe."""
        frame = self.adb.capture_screen()
        h, w = frame.shape[:2]
        roi = self.detector.detect_board_region(frame)

        # 1. Wait for animations / cascades to settle
        if not self.detector.is_board_settled(frame, roi=roi, threshold=2.2, min_stable_frames=2):
            time.sleep(0.2)
            return True

        # 2. Extract tiles and classify board
        tiles = self.detector.extract_tiles(frame, roi=roi)
        board_state = self.classifier.classify_board(tiles)
        engine_state = self.classifier.to_reverse_engine_state(board_state, moves_left=30)

        # 3. Check if active board is visible
        non_empty = (engine_state.colors > 0).sum()
        if non_empty < 15:
            self.consecutive_failed_reads += 1
            if self.consecutive_failed_reads % 4 == 0:
                print(f"[ADBBot] Board not visible (active candies={non_empty}). Checking for dialog...")
                if self.consecutive_failed_reads >= 8:
                    print("[ADBBot] Attempting dialog dismissal or retry tap...")
                    self.adb.restart_level_or_dismiss_dialog(w, h)
                    time.sleep(1.5)
                    self.consecutive_failed_reads = 0
            time.sleep(0.5)
            return True

        self.consecutive_failed_reads = 0

        # 4. Neural inference with AlphaCandy Titan
        best_move_tuple = self.agent.select_best_move(engine_state)
        r, c, direction, score, diagnostics = best_move_tuple

        regime = diagnostics.get("regime", "TITAN_INFERENCE")
        win_prob = diagnostics.get("win_prob", 0.0)
        print(f"\n🎯 [Titan 79.3M Move] ({r}, {c}) ➔ {direction.name} | Score: {score:.1f} | Win Prob: {win_prob*100:.1f}% | Regime: {regime}")

        # 5. Convert RevDirection to candy_board Direction
        dir_mapping = {
            RevDirection.UP: Direction.UP,
            RevDirection.DOWN: Direction.DOWN,
            RevDirection.LEFT: Direction.LEFT,
            RevDirection.RIGHT: Direction.RIGHT,
        }
        adb_dir = dir_mapping[direction]

        # 6. Execute touch swipe on Android device
        self.adb.execute_grid_move(roi, r, c, adb_dir)

        # Allow time for swipe registration and cascade trigger
        time.sleep(0.6)
        return True

    def run_continuous(self, max_steps: int = 1000):
        """Runs the live bot continuously."""
        print("\n🚀 ADB BOT RUNNING! Press Ctrl+C to stop.")
        step = 0
        try:
            while step < max_steps:
                self.play_step()
                step += 1
        except KeyboardInterrupt:
            print("\n[ADBBot] Stopped by user.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Candy Crush AI Titan ADB Bot")
    parser.add_argument("--device", type=str, default=None, help="ADB Device ID")
    parser.add_argument("--checkpoint", type=str, default=None, help="Custom checkpoint path")
    parser.add_argument("--steps", type=int, default=500, help="Max steps to play")
    args = parser.parse_args()

    bot = CandyCrushLiveADBBot(device_id=args.device, checkpoint_path=args.checkpoint)
    bot.run_continuous(max_steps=args.steps)
