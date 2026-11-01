#!/usr/bin/env python3
"""
Desktop Screen Vision Bot for Live King Candy Crush Saga.
Runs directly against any desktop window (Browser, Facebook, scrcpy, emulator):
Perceives screen via MSS -> Classifies via CV -> Evaluates via AlphaCandy Titan (79.3M) -> Executes Drag-and-Drop.
"""

import sys
import os
import time
import argparse
from pathlib import Path
from typing import Optional, Tuple

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from automation.display_utils import DesktopScreenCapture, DesktopInputController, setup_display_environment
from vision.robust_grid_detector import RobustGridDetector
from vision.robust_tile_classifier import RobustTileClassifier
from models.alphacandy_titan_agent import AlphaCandyTitanAgent
from core.reverse_engine import Direction, OFFSETS

CANDY_EMOJIS = {
    1: "🔴",  # Red
    2: "🔵",  # Blue
    3: "🟢",  # Green
    4: "🟡",  # Yellow
    5: "🟠",  # Orange
    6: "🟣",  # Purple
    0: " ·",  # Empty
}


class DesktopCandyBot:
    """Live desktop automation bot powered by AlphaCandy Titan 79.3M."""

    def __init__(self, checkpoint_path: Optional[str] = None):
        print("=" * 65)
        print("🍬 ALPHACANDY TITAN — LIVE REAL GAME DESKTOP BOT (79.3M PARAMS)")
        print("=" * 65)
        setup_display_environment()
        self.capture = DesktopScreenCapture()
        self.input_ctl = DesktopInputController()
        self.detector = RobustGridDetector()
        self.classifier = RobustTileClassifier()
        self.agent = AlphaCandyTitanAgent(checkpoint_path=checkpoint_path)

        self.consecutive_stalls = 0
        print("[DesktopBot] Initialized and connected to desktop display server.")

    def render_terminal_hud(self, state, best_move, diagnostics):
        """Prints live visual ASCII board and Titan diagnostics."""
        r_src, c_src, direction, score = best_move[:4]
        dr, dc = OFFSETS[direction]
        r_tgt, c_tgt = r_src + dr, c_src + dc

        print("\n" + "─" * 45)
        print("     0   1   2   3   4   5   6   7   8")
        print("   ┌───┬───┬───┬───┬───┬───┬───┬───┬───┐")
        for r in range(9):
            row_str = f" {r} │"
            for c in range(9):
                val = state.colors[r, c]
                glyph = CANDY_EMOJIS.get(val, "  ")
                if (r == r_src and c == c_src) or (r == r_tgt and c == c_tgt):
                    row_str += f"[{glyph}]"
                else:
                    row_str += f" {glyph} "
            row_str += "│"
            print(row_str)
        print("   └───┴───┴───┴───┴───┴───┴───┴───┴───┘")

        regime = diagnostics.get("regime", "TITAN_INFERENCE")
        win_prob = diagnostics.get("win_prob", 0.0)
        conf = diagnostics.get("epistemic_confidence", 1.0)
        rho = diagnostics.get("pace_rho", 0.0)

        print(f"🎯 Titan Move: ({r_src}, {c_src}) ➔ {direction.name} | Target: ({r_tgt}, {c_tgt})")
        print(f"🧠 Score: {score:.1f} | Win Prob: {win_prob*100:.1f}% | Confidence: {conf:.2f} | Pace ρ: {rho:.2f} | Regime: {regime}")
        print("─" * 45)

    def play_step(self) -> bool:
        """Executes a single perception -> decision -> drag gesture cycle."""
        frame = self.capture.grab_screen()
        roi = self.detector.detect_board_region(frame)

        # 1. Wait for cascades and animations to settle
        if not self.detector.is_board_settled(frame, roi=roi, threshold=2.2, min_stable_frames=2):
            time.sleep(0.2)
            return True

        # 2. Extract tiles and classify board
        tiles = self.detector.extract_tiles(frame, roi=roi)
        board_state = self.classifier.classify_board(tiles)
        engine_state = self.classifier.to_reverse_engine_state(board_state, moves_left=30)

        # Verify whether there are candies on the board
        non_empty = (engine_state.colors > 0).sum()
        if non_empty < 15:
            self.consecutive_stalls += 1
            if self.consecutive_stalls % 5 == 0:
                print(f"[DesktopBot] Waiting for active game board (non-empty tiles={non_empty})...")
            time.sleep(0.5)
            return True

        self.consecutive_stalls = 0

        # 3. Neural inference via AlphaCandy Titan Agent
        best_move_tuple = self.agent.select_best_move(engine_state)
        r, c, direction, score, diagnostics = best_move_tuple

        # Render HUD
        self.render_terminal_hud(engine_state, (r, c, direction, score), diagnostics)

        # 4. Compute screen pixel coordinates for drag gesture
        dr, dc = OFFSETS[direction]
        target_r, target_c = r + dr, c + dc

        x1, y1 = self.detector.get_cell_center_pixel(r, c, roi=roi)
        x2, y2 = self.detector.get_cell_center_pixel(target_r, target_c, roi=roi)

        # 5. Execute mouse drag
        print(f"[DesktopBot] Executing drag: ({x1}, {y1}) ➔ ({x2}, {y2})...")
        self.input_ctl.drag(x1, y1, x2, y2, duration=0.22)

        # Pause to let swipe register and trigger cascades
        time.sleep(0.6)
        return True

    def run(self, max_steps: int = 500):
        """Continuously plays until max_steps or stopped."""
        print("\n🚀 BOT RUNNING! [Move mouse to any screen corner or press Ctrl+C to stop]")
        step = 0
        try:
            while step < max_steps:
                self.play_step()
                step += 1
        except KeyboardInterrupt:
            print("\n[DesktopBot] Stopped by user.")
        finally:
            self.capture.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AlphaCandy Titan Real Game Desktop Bot")
    parser.add_argument("--steps", type=int, default=300, help="Max steps to play")
    parser.add_argument("--checkpoint", type=str, default=None, help="Custom model checkpoint path")
    args = parser.parse_args()

    bot = DesktopCandyBot(checkpoint_path=args.checkpoint)
    bot.run(max_steps=args.steps)
