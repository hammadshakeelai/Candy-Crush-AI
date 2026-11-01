#!/usr/bin/env python3
"""
AlphaCandy Titan — Live Web Browser Bot for Official King Candy Crush Saga
========================================================================
Controls a live, headful Google Chrome browser session via Playwright:
  1. Opens persistent Chrome window on the user's desktop display (:0).
  2. Keeps the window open for the user to log into Facebook / King.
  3. Detects the live game canvas/iframe.
  4. Perceives board state -> classifies tiles -> evaluates via AlphaCandy Titan 79.3M.
  5. Injects drag gestures directly through Playwright's mouse API.
"""

import os
import sys
import time
import asyncio
from pathlib import Path
from typing import Optional, Tuple
import cv2
import numpy as np

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from playwright.async_api import async_playwright, Page, BrowserContext
from core.reverse_engine import Direction, OFFSETS
from vision.robust_grid_detector import RobustGridDetector
from vision.robust_tile_classifier import RobustTileClassifier
from models.alphacandy_titan_agent import AlphaCandyTitanAgent

CANDY_EMOJIS = {
    1: "🔴",  # Red
    2: "🔵",  # Blue
    3: "🟢",  # Green
    4: "🟡",  # Yellow
    5: "🟠",  # Orange
    6: "🟣",  # Purple
    0: " ·",  # Empty
}


class AlphaCandyWebBot:
    """Direct web player for official King Candy Crush Saga on Facebook Gaming."""

    def __init__(self, checkpoint_path: Optional[str] = None, cookies_path: Optional[str] = None):
        print("=" * 65)
        print("🍬 ALPHACANDY TITAN — LIVE WEB BROWSER BOT (79.3M PARAMS)")
        print("=" * 65)
        self.detector = RobustGridDetector()
        self.classifier = RobustTileClassifier()
        self.agent = AlphaCandyTitanAgent(checkpoint_path=checkpoint_path)
        self.user_data_dir = os.path.expanduser("~/.config/alphacandy-browser")
        os.makedirs(self.user_data_dir, exist_ok=True)
        self.cookies_path = cookies_path or str(ROOT_DIR / "config" / "facebook_cookies.json")

    async def inject_cookies(self, ctx: BrowserContext):
        """Loads and injects cookies from JSON or raw cookie header string."""
        raw_cookie_txt = ROOT_DIR / "config" / "facebook_cookies.txt"
        cookie_json = Path(self.cookies_path)

        cookie_list = []
        if cookie_json.exists():
            try:
                import json
                with open(cookie_json) as f:
                    data = json.load(f)
                if isinstance(data, list):
                    cookie_list = data
                elif isinstance(data, dict):
                    for k, v in data.items():
                        cookie_list.append({
                            "name": k,
                            "value": str(v),
                            "domain": ".facebook.com",
                            "path": "/",
                            "secure": True,
                        })
                print(f"[WebBot] Loaded {len(cookie_list)} cookies from {cookie_json.name}")
            except Exception as e:
                print(f"[WebBot] Warning reading {cookie_json}: {e}")

        elif raw_cookie_txt.exists():
            try:
                content = raw_cookie_txt.read_text().strip()
                for item in content.split(";"):
                    if "=" in item:
                        name, val = item.strip().split("=", 1)
                        cookie_list.append({
                            "name": name.strip(),
                            "value": val.strip(),
                            "domain": ".facebook.com",
                            "path": "/",
                            "secure": True,
                        })
                print(f"[WebBot] Loaded {len(cookie_list)} cookies from {raw_cookie_txt.name}")
            except Exception as e:
                print(f"[WebBot] Warning reading {raw_cookie_txt}: {e}")

        if cookie_list:
            # Ensure proper domain formatting for Playwright
            for c in cookie_list:
                if "domain" not in c:
                    c["domain"] = ".facebook.com"
                if "path" not in c:
                    c["path"] = "/"
            await ctx.add_cookies(cookie_list)
            print("[WebBot] ✓ Injected cookies into browser context successfully!")

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
        print("─" * 45, flush=True)

    async def find_game_frame(self, page: Page):
        """Locates the game iframe or canvas holding the King WebGL player."""
        # Check all child frames
        for frame in page.frames:
            url = frame.url.lower()
            if "king.com" in url or "candycrush" in url or "game" in url:
                canvas = await frame.query_selector("canvas")
                if canvas:
                    return frame, canvas
        
        # Check main page canvas
        canvases = await page.query_selector_all("canvas")
        for c in canvases:
            box = await c.bounding_box()
            if box and box["width"] >= 400 and box["height"] >= 400:
                return page, c
        return None, None

    async def execute_swipe(self, target, x1: float, y1: float, x2: float, y2: float):
        """Simulates an authentic human mouse drag gesture across the canvas."""
        page = target if isinstance(target, Page) else target.page
        await page.mouse.move(x1, y1)
        await asyncio.sleep(0.08)
        await page.mouse.down()
        await asyncio.sleep(0.05)
        # Interpolate drag motion
        steps = 6
        for s in range(1, steps + 1):
            cur_x = x1 + (x2 - x1) * (s / steps)
            cur_y = y1 + (y2 - y1) * (s / steps)
            await page.mouse.move(cur_x, cur_y)
            await asyncio.sleep(0.02)
        await asyncio.sleep(0.06)
        await page.mouse.up()

    async def run(self, max_steps: int = 500, dry_run: bool = False):
        async with async_playwright() as p:
            print("[WebBot] Launching persistent Google Chrome on desktop display...")
            ctx: BrowserContext = await p.chromium.launch_persistent_context(
                self.user_data_dir,
                headless=False,
                executable_path="/opt/google/chrome/chrome",
                args=[
                    "--no-sandbox",
                    "--window-size=1380,980",
                    "--disable-blink-features=AutomationControlled",
                ],
                viewport={"width": 1380, "height": 980},
            )

            page = ctx.pages[0] if ctx.pages else await ctx.new_page()
            await self.inject_cookies(ctx)
            target_url = "https://www.facebook.com/gaming/play/210831918949520/?source=canvas_to_ig_app_migration"
            
            print(f"[WebBot] Navigating to {target_url}...")
            await page.goto(target_url)

            print("\n" + "=" * 65)
            print("🔑 BROWSER OPEN ON YOUR DESKTOP SCREEN!")
            print("Please log into Facebook if prompted and let the game load.")
            print("The bot will automatically detect the candy board as soon as it appears.")
            print("=" * 65 + "\n", flush=True)

            step = 0
            consecutive_stalls = 0

            while step < max_steps:
                try:
                    # 1. Check for game canvas or iframe
                    frame_target, canvas_el = await self.find_game_frame(page)
                    active_page = frame_target if isinstance(frame_target, Page) else (frame_target.page if frame_target else page)
                    
                    # Grab frame from canvas if available, else viewport
                    if canvas_el:
                        screenshot_bytes = await canvas_el.screenshot()
                        c_box = await canvas_el.bounding_box()
                        canvas_offset_x = c_box["x"] if c_box else 0
                        canvas_offset_y = c_box["y"] if c_box else 0
                    else:
                        screenshot_bytes = await active_page.screenshot()
                        canvas_offset_x, canvas_offset_y = 0, 0

                    img_arr = np.frombuffer(screenshot_bytes, dtype=np.uint8)
                    frame_bgr = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)

                    if frame_bgr is None:
                        await asyncio.sleep(1.0)
                        continue

                    # 2. Detect board ROI
                    roi = self.detector.detect_board_region(frame_bgr)
                    tiles = self.detector.extract_tiles(frame_bgr, roi=roi)
                    board_state = self.classifier.classify_board(tiles)
                    engine_state = self.classifier.to_reverse_engine_state(board_state, moves_left=30)

                    non_empty = (engine_state.colors > 0).sum()
                    if non_empty < 15:
                        consecutive_stalls += 1
                        if consecutive_stalls % 6 == 0:
                            print(f"[WebBot] Waiting for active board (candies visible: {non_empty})... Log in or select level in browser.")
                        await asyncio.sleep(0.8)
                        continue

                    consecutive_stalls = 0

                    # 3. Neural inference via AlphaCandy Titan (79.3M)
                    best_move = self.agent.select_best_move(engine_state)
                    if not best_move:
                        await asyncio.sleep(0.5)
                        continue

                    r, c, direction, score, diag = best_move
                    self.render_terminal_hud(engine_state, (r, c, direction, score), diag)

                    # 4. Calculate cell center pixel coordinates
                    dr, dc = OFFSETS[direction]
                    target_r, target_c = r + dr, c + dc

                    x1, y1 = self.detector.get_cell_center_pixel(r, c, roi=roi)
                    x2, y2 = self.detector.get_cell_center_pixel(target_r, target_c, roi=roi)

                    # Adjust by canvas offset if inside page
                    screen_x1 = canvas_offset_x + x1
                    screen_y1 = canvas_offset_y + y1
                    screen_x2 = canvas_offset_x + x2
                    screen_y2 = canvas_offset_y + y2

                    if dry_run:
                        print(f"[WebBot - DRY RUN] Move predicted: ({screen_x1:.0f}, {screen_y1:.0f}) ➔ ({screen_x2:.0f}, {screen_y2:.0f})")
                    else:
                        print(f"[WebBot] Executing swipe: ({screen_x1:.0f}, {screen_y1:.0f}) ➔ ({screen_x2:.0f}, {screen_y2:.0f})...")
                        await self.execute_swipe(active_page, screen_x1, screen_y1, screen_x2, screen_y2)

                    step += 1
                    # Give time for cascades to complete
                    await asyncio.sleep(2.5)

                except Exception as e:
                    await asyncio.sleep(1.0)

            print("[WebBot] Session complete.")
            await ctx.close()


def main():
    import argparse
    parser = argparse.ArgumentParser(description="AlphaCandy Titan Live Web Bot")
    parser.add_argument("--steps", type=int, default=500, help="Max steps to play")
    parser.add_argument("--dry-run", action="store_true", help="Predict and show moves without swiping")
    parser.add_argument("--cookies", type=str, default=None, help="Path to cookies file (JSON or raw header string)")
    args = parser.parse_args()

    bot = AlphaCandyWebBot(cookies_path=args.cookies)
    asyncio.run(bot.run(max_steps=args.steps, dry_run=args.dry_run))


if __name__ == "__main__":
    main()
