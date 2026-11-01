#!/usr/bin/env python3
"""
AlphaCandy-Titan: Real Candy Crush Saga Live Player & Interactive CLI.
Supports both autonomous AI live playthroughs and human interactive play
with real-time AlphaCandy Titan (79.3M) move analysis and recommendations.
"""

import sys
import os
import time
import argparse
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any

# Ensure project root is in python path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from core.king_level_loader import KingLevelLoader
from core.reverse_engine import (
    ReverseCandyEngine,
    ReverseEngineState,
    Direction,
    LevelType,
    SpecialCandy,
    Blocker,
    OFFSETS,
)
from models.alphacandy_titan_agent import AlphaCandyTitanAgent

# Visual Glyphs for Terminal Rendering
CANDY_EMOJIS = {
    1: "🔴",  # Red
    2: "🔵",  # Blue
    3: "🟢",  # Green
    4: "🟡",  # Yellow
    5: "🟠",  # Orange
    6: "🟣",  # Purple
}

SPECIAL_BADGES = {
    SpecialCandy.STRIPED_HORIZONTAL: "↔",
    SpecialCandy.STRIPED_VERTICAL: "↕",
    SpecialCandy.WRAPPED: "📦",
    SpecialCandy.COLOR_BOMB: "💣",
}

BLOCKER_EMOJIS = {
    Blocker.CHOCOLATE: "🍫",
    Blocker.FROSTING: "❄️ ",
    Blocker.LICORICE: "🔒",
}


def render_board(state: ReverseEngineState, highlight_swap: Optional[Tuple[int, int, Direction]] = None) -> str:
    """Renders the 9x9 board into a high-visibility terminal ASCII/Unicode grid."""
    lines = []
    lines.append("     0   1   2   3   4   5   6   7   8")
    lines.append("   ┌───┬───┬───┬───┬───┬───┬───┬───┬───┐")

    h_r, h_c, h_dir = highlight_swap if highlight_swap else (-1, -1, None)
    dr, dc = OFFSETS.get(h_dir, (0, 0)) if h_dir else (0, 0)
    tgt_r, tgt_c = h_r + dr, h_c + dc

    for r in range(9):
        row_str = f" {r} │"
        for c in range(9):
            if not state.active_grid[r, c]:
                row_str += " · │"
                continue

            # Check if this cell is highlighted as part of a swap
            is_swap_source = (r == h_r and c == h_c)
            is_swap_target = (r == tgt_r and c == tgt_c)

            # Blockers take visual precedence
            blocker = state.blockers[r, c]
            candy_val = state.colors[r, c]
            special = state.specials[r, c]
            jelly = state.jellies[r, c]

            glyph = "  "
            if blocker != Blocker.NONE:
                glyph = BLOCKER_EMOJIS.get(blocker, "⬛")
            elif special == SpecialCandy.COLOR_BOMB or candy_val == 6:
                glyph = "💣"
            elif candy_val in CANDY_EMOJIS:
                base = CANDY_EMOJIS[candy_val]
                badge = SPECIAL_BADGES.get(special, "")
                glyph = f"{badge}{base}" if badge else f" {base}"
            else:
                glyph = "  "

            # Overlay Jelly indicator if present
            if jelly == 2:
                # Double jelly: magenta border/tint
                cell_repr = f"*{glyph[0:2]}" if len(glyph) >= 2 else f"*{glyph} "
            elif jelly == 1:
                # Single jelly
                cell_repr = glyph[0:2] if len(glyph) >= 2 else f"{glyph} "
            else:
                cell_repr = glyph[0:2] if len(glyph) >= 2 else f"{glyph} "

            if is_swap_source:
                row_str += f"[{cell_repr[0:1]}]│"
            elif is_swap_target:
                row_str += f"<{cell_repr[0:1]}>│"
            else:
                row_str += f"{cell_repr:2s}│"

        lines.append(row_str)
        if r < 8:
            lines.append("   ├───┼───┼───┼───┼───┼───┼───┼───┼───┤")
        else:
            lines.append("   └───┴───┴───┴───┴───┴───┴───┴───┴───┘")

    return "\n".join(lines)


def play_level_live(level_id: int, auto: bool = True, max_moves: Optional[int] = None, seed: int = 42):
    """Executes a real game playthrough with live board rendering and AlphaCandy Titan analysis."""
    print("=" * 78)
    print(f"🎮 REAL CANDY CRUSH SAGA LIVE PLAYER — LEVEL {level_id:04d}")
    print("   AI Flagship: AlphaCandy-Titan (79.30M Parameters)")
    print("=" * 78)

    loader = KingLevelLoader()
    cfg = loader.load_level(level_id)
    if max_moves:
        cfg.moves_limit = max_moves

    engine = ReverseCandyEngine()
    state = ReverseEngineState(cfg)
    engine.initialize_board(state, seed=seed)

    print(f"\n📋 Level Configuration:")
    print(f"   • Level Type:        {state.config.level_type.name}")
    print(f"   • Move Limit:        {state.config.moves_limit} turns")
    print(f"   • Target Score:      {state.config.target_score:,} pts")
    print(f"   • Initial Jellies:   {int(state.jellies.sum())}")
    print(f"   • Available Colors:  {state.config.num_colors}")
    print(f"   • Seed:              {seed}")

    agent = AlphaCandyTitanAgent(checkpoint_path="checkpoints/alphacandy_titan_master.pt")

    turn = 1
    total_score = 0

    while state.moves_left > 0:
        print("\n" + "—" * 65)
        print(f"⏳ TURN {turn} | Moves Remaining: {state.moves_left} | Score: {state.score:,} / {state.config.target_score:,}")
        if state.config.level_type == LevelType.JELLY:
            print(f"   🎯 Objectives Left: {int(state.jellies.sum())} jellies")
        print("—" * 65)

        # Print current board
        print(render_board(state))

        # Query AlphaCandy Titan for Top Move & Decision Analysis
        move_result = agent.select_best_move(state)
        if move_result is None:
            print("⚠️ No legal moves remaining on board (reshuffle required)!")
            break

        best_r, best_c, best_dir, pred_score, info = move_result
        regime = info.get("regime", "NORMAL_EV")
        conf = info.get("epistemic_confidence", 0.95)
        win_prob = info.get("win_prob", 0.90)

        dir_arrow = {"UP": "↑", "DOWN": "↓", "LEFT": "←", "RIGHT": "→"}.get(best_dir.name, best_dir.name)

        print(f"\n🧠 AlphaCandy Titan Neural Policy:")
        print(f"   • Recommended Move:   Swap ({best_r}, {best_c}) {best_dir.name} {dir_arrow}")
        print(f"   • Risk Regime:        {regime} (Monotonic 32-Quantile Tilting)")
        print(f"   • Predicted P(Win):    {win_prob * 100:.1f}%")
        print(f"   • Epistemic Conf:     {conf * 100:.1f}%")

        chosen_r, chosen_c, chosen_dir = best_r, best_c, best_dir

        if not auto:
            # Interactive Human Play
            user_input = input("\n👉 Enter move (e.g. '5 6 r', 'hint', or press Enter to accept Titan move): ").strip()
            if user_input.lower() in ("q", "quit", "exit"):
                print("Game exited by user.")
                return
            elif user_input.lower() == "hint":
                print(f"💡 Titan Hint: Swap ({best_r}, {best_c}) {best_dir.name} gives maximum cascade potential!")
                continue
            elif user_input:
                parts = user_input.split()
                if len(parts) == 3:
                    try:
                        pr, pc, pdir = int(parts[0]), int(parts[1]), parts[2].lower()
                        d_map = {"u": Direction.UP, "d": Direction.DOWN, "l": Direction.LEFT, "r": Direction.RIGHT}
                        chosen_r, chosen_c, chosen_dir = pr, pc, d_map.get(pdir, Direction.RIGHT)
                    except Exception as e:
                        print(f"Invalid format ({e}), proceeding with Titan recommendation.")

        # Show the chosen swap on the board
        print(f"\n⚡ Executing Swap: ({chosen_r}, {chosen_c}) <-> ({chosen_r + OFFSETS[chosen_dir][0]}, {chosen_c + OFFSETS[chosen_dir][1]})")
        print(render_board(state, highlight_swap=(chosen_r, chosen_c, chosen_dir)))

        # Execute on Reverse Candy Engine
        pts, step_info = engine.execute_move(state, chosen_r, chosen_c, chosen_dir)
        total_score += pts

        # Report Step Micro-Physics
        print(f"💥 Cascade Triggered:")
        print(f"   • Points Gained:      +{pts:,}")
        print(f"   • Specials Cleared:   {step_info.get('specials_cleared', 0)}")
        print(f"   • Jellies Popped:     {step_info.get('jellies_cleared', 0)}")
        print(f"   • Total Score:        {state.score:,}")

        # Check Win Conditions
        won = state.has_won()

        if won:
            print("\n" + "=" * 78)
            print(f"🎉🎉🎉 VICTORY! LEVEL {level_id:04d} CLEARED ON TURN {turn}! 🎉🎉🎉")
            print(f"   • Final Score:        {state.score:,} (Target: {state.config.target_score:,})")
            print(f"   • Moves Remaining:    {state.moves_left}")
            print(f"   • Turns Used:         {turn} of {state.config.moves_limit}")
            print("=" * 78)
            return True

        turn += 1
        if auto:
            time.sleep(0.4)

    # Check loss
    print("\n" + "=" * 78)
    print(f"💔 OUT OF MOVES! Final Score: {state.score:,} / {state.config.target_score:,}")
    print("=" * 78)
    return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AlphaCandy Live Real Game Player")
    parser.add_argument("--level", type=int, default=65, help="King level ID to play (default: 65)")
    parser.add_argument("--interactive", action="store_true", help="Enable human turn-by-turn interactive play")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed for candy refill generation")
    args = parser.parse_args()

    play_level_live(level_id=args.level, auto=not args.interactive, seed=args.seed)
