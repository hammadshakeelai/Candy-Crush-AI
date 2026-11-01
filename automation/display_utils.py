"""
Display & Input Automation Utilities for Linux (GNOME Wayland / Xwayland).
Handles Xauthority cookie extraction, high-speed MSS screen capture,
and PyAutoGUI drag-and-drop input injection.
"""

import os
import glob
import time
import struct
from typing import Tuple, Optional
import numpy as np

def setup_display_environment() -> bool:
    """
    Ensures XAUTHORITY and DISPLAY are configured to interact with Xwayland on GNOME/Wayland.
    Reads mutter Xwayland cookie, adds display number '0', and saves to /tmp/.Xauthority-candy.
    """
    os.environ["DISPLAY"] = ":0"
    
    # Check if existing XAUTHORITY works
    if "XAUTHORITY" in os.environ and os.path.exists(os.environ["XAUTHORITY"]):
        try:
            from Xlib import display
            d = display.Display(":0")
            d.close()
            return True
        except Exception:
            pass

    # Look for Mutter Xwayland auth files
    mutter_auths = glob.glob("/run/user/1000/.mutter-Xwaylandauth.*")
    if not mutter_auths:
        # Fallback to standard locations
        fallback_auth = os.path.expanduser("~/.Xauthority")
        if os.path.exists(fallback_auth):
            os.environ["XAUTHORITY"] = fallback_auth
            return True
        return False

    try:
        from Xlib import xauth
        mutter_path = mutter_auths[0]
        m_auth = xauth.Xauthority(mutter_path)
        if not m_auth.entries:
            return False

        cookie = m_auth.entries[0][4]
        new_entries = list(m_auth.entries)
        # Add entries matching dispno '0'
        new_entries.append((256, b"archlinux", b"0", b"MIT-MAGIC-COOKIE-1", cookie))
        new_entries.append((65535, b"archlinux", b"0", b"MIT-MAGIC-COOKIE-1", cookie))
        new_entries.append((256, b"", b"0", b"MIT-MAGIC-COOKIE-1", cookie))
        new_entries.append((0, b"127.0.0.1", b"0", b"MIT-MAGIC-COOKIE-1", cookie))

        out_path = "/tmp/.Xauthority-candy"
        with open(out_path, "wb") as f:
            for fam, addr, disp, name, data in new_entries:
                f.write(struct.pack(">H", fam))
                f.write(struct.pack(">H", len(addr)) + addr)
                f.write(struct.pack(">H", len(disp)) + disp)
                f.write(struct.pack(">H", len(name)) + name)
                f.write(struct.pack(">H", len(data)) + data)

        os.environ["XAUTHORITY"] = out_path
        return True
    except Exception as e:
        print(f"[DisplayUtils] Warning during display auth setup: {e}")
        return False


# Initialize display environment on module import
setup_display_environment()

import mss
import pyautogui
import cv2

# Set PyAutoGUI fail-safe (moving mouse to any corner raises exception)
pyautogui.FAILSAFE = True
pyautogui.PAUSE = 0.05


class DesktopScreenCapture:
    """High-performance desktop screen capture using MSS."""

    def __init__(self, monitor_index: int = 1):
        self.sct = mss.mss()
        # monitor 0 is all monitors combined, monitor 1 is primary monitor
        if len(self.sct.monitors) > monitor_index:
            self.monitor = self.sct.monitors[monitor_index]
        else:
            self.monitor = self.sct.monitors[0]

    def grab_screen(self) -> np.ndarray:
        """Captures primary monitor and returns standard BGR numpy array."""
        shot = self.sct.grab(self.monitor)
        # Convert BGRA to BGR
        frame = np.array(shot)[:, :, :3]
        return np.ascontiguousarray(frame)

    def grab_roi(self, x: int, y: int, w: int, h: int) -> np.ndarray:
        """Captures specific bounding box on screen."""
        bbox = {"top": int(y), "left": int(x), "width": int(w), "height": int(h)}
        shot = self.sct.grab(bbox)
        frame = np.array(shot)[:, :, :3]
        return np.ascontiguousarray(frame)

    def close(self):
        self.sct.close()


class DesktopInputController:
    """Controls mouse swipes / drag-and-drop on desktop windows."""

    def __init__(self):
        setup_display_environment()

    def drag(self, x1: int, y1: int, x2: int, y2: int, duration: float = 0.22):
        """Simulates smooth drag-and-drop swipe gesture."""
        try:
            pyautogui.moveTo(x1, y1)
            time.sleep(0.04)
            pyautogui.dragTo(x2, y2, duration=duration, button='left')
        except pyautogui.FailSafeException:
            print("\n[InputController] Fail-safe triggered by user (mouse in screen corner)!")
            raise

    def tap(self, x: int, y: int):
        """Clicks at target coordinate."""
        pyautogui.click(x, y)
