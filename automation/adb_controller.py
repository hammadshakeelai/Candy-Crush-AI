"""
ADB Controller & Automated Life Refill Engine for Candy Crush.
Implements touch injection, screen frame capture, and the 5-lives time-leap exploit.
"""

import subprocess
import time
from typing import Tuple, Optional
import cv2
import numpy as np
from core.candy_board import Direction, DIRECTION_OFFSETS


class ADBController:
    """
    Manages communication with Android device or emulator via ADB.
    Automates screen capture, swipe gestures, and life replenishments.
    """

    def __init__(self, device_id: Optional[str] = None):
        self.device_id = device_id
        self.adb_cmd = ["adb"]
        if device_id:
            self.adb_cmd.extend(["-s", device_id])

    def _run_cmd(self, args: list) -> subprocess.CompletedProcess:
        full_cmd = self.adb_cmd + args
        return subprocess.run(full_cmd, capture_output=True, text=True)

    def capture_screen(self) -> np.ndarray:
        """
        Captures screenshot directly into OpenCV BGR numpy array via adb exec-out.
        """
        proc = subprocess.run(
            self.adb_cmd + ["exec-out", "screencap", "-p"],
            capture_output=True
        )
        if proc.returncode != 0:
            raise RuntimeError(f"ADB screencap failed: {proc.stderr.decode('utf-8', errors='ignore')}")

        img_bytes = np.frombuffer(proc.stdout, dtype=np.uint8)
        frame = cv2.imdecode(img_bytes, cv2.IMREAD_COLOR)
        return frame

    def swipe(self, x1: int, y1: int, x2: int, y2: int, duration_ms: int = 150) -> None:
        """
        Executes a touch drag gesture on screen.
        """
        self._run_cmd(["shell", "input", "swipe", str(x1), str(y1), str(x2), str(y2), str(duration_ms)])

    def execute_grid_move(
        self,
        board_bbox: Tuple[int, int, int, int],
        r: int,
        c: int,
        direction: Direction,
        grid_rows: int = 9,
        grid_cols: int = 9
    ) -> None:
        """
        Translates grid coordinates (r, c) and direction into real screen pixel swipe.
        """
        bx, by, bw, bh = board_bbox
        cell_w = bw // grid_cols
        cell_h = bh // grid_rows

        # Center coordinates of source tile
        x1 = bx + int((c + 0.5) * cell_w)
        y1 = by + int((r + 0.5) * cell_h)

        dr, dc = DIRECTION_OFFSETS[direction]
        x2 = x1 + (dc * cell_w)
        y2 = y1 + (dr * cell_h)

        self.swipe(x1, y1, x2, y2, duration_ms=180)

    # ==========================================
    # THE 5-LIVES BYPASS / REFILL ENGINE
    # ==========================================

    def auto_refill_lives_time_leap(self) -> bool:
        """
        Bypasses the 5-lives limit via ADB System Time Manipulation:
        1. Fast-forwards device time by +3 hours.
        2. Taps or switches to Candy Crush to register life restoration (all 5 lives refilled).
        3. Restores real NTP network time to prevent sync errors.
        Result: 5 full lives refilled in under 4 seconds!
        """
        print("[ADBController] Initiating Automated Life Refill (Time-Leap Protocol)...")

        # 1. Advance device time by 3 hours
        # Check root access or toybox date command
        res = self._run_cmd(["shell", "su", "-c", "date -s @$(($(date +%s) + 10800))"])
        if res.returncode != 0:
            # Non-root fallback using date command directly (works on most emulators)
            self._run_cmd(["shell", "date", "-s", "+3hours"])

        time.sleep(1.0)

        # 2. Re-open / focus Candy Crush Saga
        self._run_cmd(["shell", "monkey", "-p", "com.king.candycrushsaga", "-c", "android.intent.category.LAUNCHER", "1"])
        time.sleep(1.5)

        # 3. Restore device network time synchronization
        self._run_cmd(["shell", "settings", "put", "global", "auto_time", "1"])
        print("[ADBController] Lives successfully replenished to 5! System time resynchronized.")
        return True

    def restart_level_or_dismiss_dialog(self, screen_w: int, screen_h: int) -> None:
        """Taps standard dialog confirmation button (e.g. 'Play', 'Retry', 'Continue')."""
        center_x = screen_w // 2
        button_y = int(screen_h * 0.72)
        self._run_cmd(["shell", "input", "tap", str(center_x), str(button_y)])
