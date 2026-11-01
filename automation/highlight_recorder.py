#!/usr/bin/env python3
"""
AlphaCandy Titan — Highlight & Big Moments Milestone Recorder
============================================================
Captures and compiles key grandmaster accomplishments during live gameplay:
  - Hard / Super-Hard level victories
  - Miracle cascade combos (Color Bomb + Wrapped, Double Color Bomb)
  - 2-Ply Endgame Checkmates & Isolated Jelly Snipes
  - Dramatic low-move turnarounds

Saves animated GIFs, high-res frame snapshots, and metadata JSON in `recordings/highlights/`.
"""

import os
import time
import json
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from collections import deque
import cv2
import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent.parent
RECORDINGS_DIR = REPO_ROOT / "recordings" / "highlights"


class HighlightRecorder:
    """Ring-buffered recorder that saves milestone clips when big moments happen."""

    def __init__(self, buffer_size: int = 12):
        RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)
        self.buffer = deque(maxlen=buffer_size)
        self.recorded_events = []
        print(f"[HighlightRecorder] Initialized. Milestones directory: {RECORDINGS_DIR}")

    def add_frame(self, frame_bgr: np.ndarray, metadata: Optional[Dict[str, Any]] = None):
        """Adds a frame and its current tactical metadata to the ring buffer."""
        # Annotate HUD banner on top of frame for video playback
        annotated = frame_bgr.copy()
        if metadata:
            move_str = metadata.get("move_str", "")
            regime = metadata.get("regime", "")
            score = metadata.get("score", 0.0)
            win_prob = metadata.get("win_prob", 0.0)

            # Top overlay banner
            h, w = annotated.shape[:2]
            cv2.rectangle(annotated, (0, 0), (w, 50), (20, 20, 20), -1)
            cv2.putText(
                annotated,
                f"AlphaCandy Titan 79.3M | {move_str} | Q:{score:.1f} | P(Win):{win_prob*100:.0f}% | {regime}",
                (15, 32),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )

        self.buffer.append({
            "timestamp": time.time(),
            "frame": annotated,
            "metadata": metadata or {}
        })

    def should_trigger_highlight(self, metadata: Dict[str, Any]) -> Tuple[bool, str]:
        """Evaluates whether the current state/move qualifies as a big moment."""
        regime = metadata.get("regime", "")
        score = metadata.get("score", 0.0)
        is_victory = metadata.get("is_victory", False)
        is_endgame_snipe = metadata.get("is_endgame_snipe", False)
        moves_left = metadata.get("moves_left", 30)

        if is_victory:
            return True, "LEVEL_VICTORY_ACHIEVED"
        if is_endgame_snipe or (moves_left <= 3 and score >= 50.0):
            return True, "CLUTCH_ENDGAME_SNIPE"
        if regime == "MIRACLE_UPPER_CVAR85" and score >= 40.0:
            return True, "MIRACLE_CASCADE_COMBO"
        if score >= 90.0:
            return True, "TITAN_TACTICAL_BREAKTHROUGH"

        return False, ""

    def save_highlight(self, event_name: str, extra_meta: Optional[Dict[str, Any]] = None) -> str:
        """Flushes the buffered frames into an animated GIF clip and JSON metadata."""
        if len(self.buffer) < 2:
            return ""

        timestamp_str = time.strftime("%Y%m%d_%H%M%S")
        slug = f"{timestamp_str}_{event_name.lower()}"
        gif_path = RECORDINGS_DIR / f"{slug}.gif"
        json_path = RECORDINGS_DIR / f"{slug}.json"
        thumb_path = RECORDINGS_DIR / f"{slug}_thumb.png"

        # Convert BGR frames to RGB PIL Images for smooth GIF export
        pil_images = []
        for item in self.buffer:
            rgb = cv2.cvtColor(item["frame"], cv2.COLOR_BGR2RGB)
            pil_images.append(Image.fromarray(rgb))

        # Save animated GIF (10 FPS / 250ms per frame)
        pil_images[0].save(
            str(gif_path),
            save_all=True,
            append_images=pil_images[1:],
            duration=350,
            loop=0,
            optimize=True,
        )

        # Save thumbnail
        cv2.imwrite(str(thumb_path), self.buffer[-1]["frame"])

        # Save metadata
        meta_payload = {
            "timestamp": timestamp_str,
            "event": event_name,
            "frame_count": len(self.buffer),
            "gif_file": gif_path.name,
            "thumb_file": thumb_path.name,
            "extra": extra_meta or {},
            "recent_moves": [b.get("metadata", {}) for b in self.buffer]
        }
        with open(json_path, "w") as f:
            json.dump(meta_payload, f, indent=2)

        print(f"\n🎬 [HIGHLIGHT RECORDED] {event_name} saved to: {gif_path}")
        self.recorded_events.append(str(gif_path))
        return str(gif_path)
