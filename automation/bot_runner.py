"""
End-to-End Live Bot Runner for Candy Crush.
Connects CV Screen Perception -> AI Decision Engine (MCTS/Heuristic) -> ADB Execution -> Life Management.
"""

import time
import argparse
from typing import Optional
import cv2
import numpy as np

from core.candy_board import BoardState
from core.candy_engine import CandyEngine
from models.heuristics import CandyCrushHeuristics
from models.mcts import CandyCrushMCTS
from vision.grid_detector import GridDetector
from vision.tile_classifier import TileClassifier
from automation.adb_controller import ADBController


class CandyCrushLiveBot:
    """
    Automated agent playing the live Candy Crush mobile game.
    """

    def __init__(self, device_id: Optional[str] = None, use_mcts: bool = True, mcts_sims: int = 40):
        self.adb = ADBController(device_id=device_id)
        self.detector = GridDetector()
        self.classifier = TileClassifier()
        self.engine = CandyEngine()
        self.heuristics = CandyCrushHeuristics(self.engine)
        self.use_mcts = use_mcts
        self.mcts = CandyCrushMCTS(self.engine, self.heuristics, num_simulations=mcts_sims)

        self.consecutive_failed_reads = 0

    def play_step(self) -> bool:
        """
        Executes one full decision cycle:
        1. Capture frame
        2. Verify board settling
        3. Parse board state
        4. Select optimal move
        5. Execute swipe
        """
        frame = self.adb.capture_screen()
        h, w = frame.shape[:2]

        # 1. Wait for animations / cascades to settle
        if not self.detector.is_board_settled(frame):
            time.sleep(0.3)
            return True

        # 2. Extract tiles and classify board
        tiles = self.detector.extract_tiles(frame)
        board_state = self.classifier.classify_board(tiles)

        # 3. Find legal moves
        legal_moves = self.engine.get_legal_moves(board_state)
        if not legal_moves:
            self.consecutive_failed_reads += 1
            print(f"[Bot] No legal moves detected on current frame (count={self.consecutive_failed_reads}).")
            
            # If board hasn't shown legal moves for several frames, we might be on level complete / fail screen
            if self.consecutive_failed_reads >= 4:
                print("[Bot] Screen might be on menu or life exhaustion dialog. Attempting action...")
                self.adb.restart_level_or_dismiss_dialog(w, h)
                time.sleep(2.0)
                self.consecutive_failed_reads = 0
            return True

        self.consecutive_failed_reads = 0

        # 4. AI Decision: Select Best Move
        if self.use_mcts:
            r, c, direction = self.mcts.search(board_state)
            print(f"[AI Solver - MCTS] Selected Move: ({r}, {c}) -> {direction.name}")
        else:
            best = self.heuristics.select_best_move(board_state)
            r, c, direction, score = best
            print(f"[AI Solver - Heuristic] Selected Move: ({r}, {c}) -> {direction.name} (Score: {score:.1f})")

        # 5. Execute Swipe on Device
        bbox = self.detector.detect_board_region(frame)
        self.adb.execute_grid_move(bbox, r, c, direction)

        # Allow time for swipe registration and cascade trigger
        time.sleep(0.5)
        return True

    def run_continuous(self, max_steps: int = 1000):
        """Runs the live bot continuously."""
        print("=== Starting Candy Crush AI Superhuman Bot ===")
        print("Press Ctrl+C to stop.")
        step = 0
        try:
            while step < max_steps:
                self.play_step()
                step += 1
        except KeyboardInterrupt:
            print("\n[Bot] Stopped by user.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Candy Crush AI Bot")
    parser.add_argument("--device", type=str, default=None, help="ADB Device ID")
    parser.add_argument("--mcts", action="store_true", default=True, help="Use MCTS planning")
    parser.add_argument("--sims", type=int, default=40, help="MCTS simulation budget")
    args = parser.parse_args()

    bot = CandyCrushLiveBot(device_id=args.device, use_mcts=args.mcts, mcts_sims=args.sims)
    bot.run_continuous()
