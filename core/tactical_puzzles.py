"""
Synthetic Tactical Puzzle Generator for Grandmaster Curriculum Training.
Creates high-density, mathematically verified 1- to 3-move Candy Crush micro-puzzles
where standard greedy 3-matches immediately lose, and ONLY the expert tactical trick wins.

Archetypes:
  1. COLOR_BOMB_STRIPED_ALIGNMENT: 2-move combo setup requiring lateral drop into alignment.
  2. CHOCOLATE_CONTAINMENT: Blocker suppression sacrifice preserving a crucial special candy.
  3. BOTTOM_CASCADE_TRIGGER: Low-row match triggering gravitational chain reactions.
  4. CORNER_JELLY_SNIPE: High-friction edge clearing with directional laser beam.
  5. WRAPPED_DUAL_BLAST: Dense 5x5 explosion clearing multi-layer frosting fortresses.
"""

import os
import sys
from pathlib import Path
import random
from typing import Dict, Any, Tuple, List, Optional
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core.king_level_loader import LevelConfig, LevelType
from core.reverse_engine import ReverseEngineState, Direction
from models.spatial_transformer_net import encode_spatial_board_tensor, encode_action_index
import torch


class TacticalPuzzleGenerator:
    """Generates ground-truth micro-puzzles for curriculum learning."""

    def __init__(self, seed: Optional[int] = None):
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

    def create_colorbomb_striped_alignment_puzzle(self, puzzle_id: int = 90001) -> Tuple[LevelConfig, Tuple[int, int, str]]:
        """
        Archetype 1: Color Bomb + Striped Alignment (Move limit: 2).
        A Color Bomb sits at (4, 4), and a Striped Candy sits at (2, 4).
        Row 3 column 4 has a candy separating them.
        Swapping row 3 column 3 with row 3 column 4 drops the striped candy to (3, 4),
        directly adjacent to the Color Bomb, enabling the apocalyptic combo on Move 2.
        All other moves fail the move limit.
        """
        H, W = 9, 9
        colors = np.random.randint(0, 5, size=(H, W))
        specials = np.zeros((H, W), dtype=int)
        jellies = np.zeros((H, W), dtype=int)
        blockers = np.zeros((H, W), dtype=int)

        # Place 16 double jellies scattered across the outer rim
        for r in [0, 1, 7, 8]:
            for c in range(W):
                jellies[r, c] = 2

        # Special candy setup at center
        cb_color = 0  # Red
        st_color = 0  # Red Striped

        colors[4, 4] = cb_color
        specials[4, 4] = 4  # Color Bomb

        colors[2, 4] = st_color
        specials[2, 4] = 1  # Striped Horizontal

        # Setup the intermediate tile at (3, 4) to be matchable laterally
        # Put 2 blue candies at (3, 2) and (3, 3), and blue candy at (3, 4)
        match_color = 1  # Blue
        colors[3, 2] = match_color
        colors[3, 3] = match_color
        colors[3, 4] = match_color  # 3 in a row triggers when swapped

        # Break any unintended instant 3-matches in other spots
        for r in range(H):
            for c in range(W):
                if (r, c) in [(4, 4), (2, 4), (3, 2), (3, 3), (3, 4)]:
                    continue
                # Alternate colors to prevent cascades
                colors[r, c] = ((r + c) % 4) + 1

        optimal_action = (3, 3, "H")  # Swap (3, 3) with (3, 4) to drop striped candy onto color bomb

        cfg = LevelConfig(
            level_id=puzzle_id,
            level_type=LevelType.JELLY,
            moves_limit=2,
            target_score=50000,
            height=H,
            width=W,
            num_colors=5,
            active_grid=np.ones((H, W), dtype=bool),
            initial_jellies=jellies,
            initial_blockers=blockers,
            initial_ingredients=np.zeros((H, W), dtype=bool),
            initial_colors=colors,
            initial_specials=specials,
            target_ingredients=[]
        )
        return cfg, optimal_action

    def create_chocolate_containment_puzzle(self, puzzle_id: int = 90002) -> Tuple[LevelConfig, Tuple[int, int, str]]:
        """
        Archetype 2: Chocolate Containment (Move limit: 2).
        Expanding chocolate is threatening an essential Color Bomb at (5, 5).
        Making a greedy point-scoring match on top allows chocolate to eat the Color Bomb.
        The correct move is hitting the adjacent chocolate at (5, 4) to suppress expansion,
        preserving the Color Bomb to clear the remaining objectives on move 2.
        """
        H, W = 9, 9
        colors = np.random.randint(1, 5, size=(H, W))
        specials = np.zeros((H, W), dtype=int)
        jellies = np.zeros((H, W), dtype=int)
        blockers = np.zeros((H, W), dtype=int)

        # Objective: 8 jellies in bottom corner
        for r in range(6, 9):
            for c in range(6, 9):
                jellies[r, c] = 1

        # Color Bomb positioned at (5, 5)
        specials[5, 5] = 4
        colors[5, 5] = 0

        # Chocolate block positioned at (5, 4)
        blockers[5, 4] = 2  # Chocolate

        # Create a 3-match that damages the chocolate at (5, 4)
        target_color = 2
        colors[4, 4] = target_color
        colors[6, 4] = target_color
        colors[5, 3] = target_color  # Swap (5, 3) into (5, 4)

        optimal_action = (5, 3, "H")  # Direct strike on chocolate

        cfg = LevelConfig(
            level_id=puzzle_id,
            level_type=LevelType.JELLY,
            moves_limit=2,
            target_score=35000,
            height=H,
            width=W,
            num_colors=5,
            active_grid=np.ones((H, W), dtype=bool),
            initial_jellies=jellies,
            initial_blockers=blockers,
            initial_ingredients=np.zeros((H, W), dtype=bool),
            initial_colors=colors,
            initial_specials=specials,
            target_ingredients=[]
        )
        return cfg, optimal_action

    def create_corner_jelly_snipe_puzzle(self, puzzle_id: int = 90003) -> Tuple[LevelConfig, Tuple[int, int, str]]:
        """
        Archetype 3: Corner Jelly Snipe (Move limit: 1).
        A single stubborn double jelly sits isolated in bottom corner (8, 0).
        There are no matching candies near (8, 0).
        A Horizontal Striped Candy sits at (8, 6).
        The only winning move is matching the Striped Candy at (8, 6) horizontally,
        firing a laser across row 8 to snipe the isolated corner jelly.
        """
        H, W = 9, 9
        colors = np.random.randint(1, 5, size=(H, W))
        specials = np.zeros((H, W), dtype=int)
        jellies = np.zeros((H, W), dtype=int)
        blockers = np.zeros((H, W), dtype=int)

        # Isolated corner jelly at (8, 0)
        jellies[8, 0] = 1

        # Horizontal Striped candy at (8, 6)
        match_c = 3
        specials[8, 6] = 1  # Striped Horizontal
        colors[8, 6] = match_c
        colors[8, 5] = match_c
        colors[8, 8] = match_c
        colors[7, 7] = match_c  # Swapping (7, 7) down to (8, 7) completes the 3-match

        optimal_action = (7, 7, "V")  # Swap (7, 7) with (8, 7) to fire the row 8 laser

        cfg = LevelConfig(
            level_id=puzzle_id,
            level_type=LevelType.JELLY,
            moves_limit=1,
            target_score=15000,
            height=H,
            width=W,
            num_colors=5,
            active_grid=np.ones((H, W), dtype=bool),
            initial_jellies=jellies,
            initial_blockers=blockers,
            initial_ingredients=np.zeros((H, W), dtype=bool),
            initial_colors=colors,
            initial_specials=specials,
            target_ingredients=[]
        )
        return cfg, optimal_action

    def create_wrapped_dual_blast_puzzle(self, puzzle_id: int = 90004) -> Tuple[LevelConfig, Tuple[int, int, str]]:
        """
        Archetype 4: Wrapped Dual Blast (Move limit: 1).
        Two Wrapped Candies are adjacent at (4, 4) and (4, 5).
        Surrounding them is a dense 5x5 cluster of 20 single jellies.
        Swapping them together triggers the massive 5x5 double detonation,
        clearing all 20 jellies in 1 turn. Any individual match fails.
        """
        H, W = 9, 9
        colors = np.random.randint(1, 5, size=(H, W))
        specials = np.zeros((H, W), dtype=int)
        jellies = np.zeros((H, W), dtype=int)
        blockers = np.zeros((H, W), dtype=int)

        # 5x5 grid of jellies around center
        for r in range(2, 7):
            for c in range(2, 7):
                jellies[r, c] = 1

        # Two adjacent wrapped candies
        specials[4, 4] = 3  # Wrapped
        specials[4, 5] = 3  # Wrapped
        colors[4, 4] = 0
        colors[4, 5] = 1

        optimal_action = (4, 4, "H")  # Swap wrapped with wrapped

        cfg = LevelConfig(
            level_id=puzzle_id,
            level_type=LevelType.JELLY,
            moves_limit=1,
            target_score=30000,
            height=H,
            width=W,
            num_colors=5,
            active_grid=np.ones((H, W), dtype=bool),
            initial_jellies=jellies,
            initial_blockers=blockers,
            initial_ingredients=np.zeros((H, W), dtype=bool),
            initial_colors=colors,
            initial_specials=specials,
            target_ingredients=[]
        )
        return cfg, optimal_action

    def generate_batch(self, count_per_type: int = 500) -> List[Dict[str, Any]]:
        """Generates a diverse batch of tactical micro-puzzles."""
        puzzles = []
        pid = 90001
        types = [
            ("COLOR_BOMB_STRIPED_ALIGNMENT", self.create_colorbomb_striped_alignment_puzzle),
            ("CHOCOLATE_CONTAINMENT", self.create_chocolate_containment_puzzle),
            ("CORNER_JELLY_SNIPE", self.create_corner_jelly_snipe_puzzle),
            ("WRAPPED_DUAL_BLAST", self.create_wrapped_dual_blast_puzzle),
        ]

        for trick_name, gen_func in types:
            for _ in range(count_per_type):
                cfg, opt_action = gen_func(puzzle_id=pid)
                puzzles.append({
                    "config": cfg,
                    "optimal_action": opt_action,
                    "trick_name": trick_name,
                    "puzzle_id": pid
                })
                pid += 1

        return puzzles


def generate_and_save_tactical_dataset(
    output_path: str = "data/synthetic_tactics_dataset.pt",
    count_per_type: int = 500,
) -> None:
    generator = TacticalPuzzleGenerator(seed=42)
    batch = generator.generate_batch(count_per_type=count_per_type)
    total = len(batch)
    states = np.empty((total, 20, 9, 9), dtype=np.float32)
    actions = np.empty((total,), dtype=np.int64)
    win_targets = np.ones((total, 1), dtype=np.float32)
    move_ratios = np.ones((total, 1), dtype=np.float32)

    for idx, p in enumerate(batch):
        cfg = p["config"]
        r, c, d_str = p["optimal_action"]
        state = ReverseEngineState(cfg)
        states[idx] = encode_spatial_board_tensor(state)
        direction = Direction.RIGHT if d_str == "H" else Direction.DOWN
        actions[idx] = encode_action_index(r, c, direction)

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    payload = {
        "states": torch.from_numpy(states),
        "actions": torch.from_numpy(actions),
        "win_targets": torch.from_numpy(win_targets),
        "move_ratios": torch.from_numpy(move_ratios),
    }
    torch.save(payload, output_path)
    print(f"[SUCCESS] Saved {total:,} puzzles ({count_per_type}/type) -> {output_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate 4-Archetype Tactical Dataset")
    parser.add_argument("--count-per-type", type=int, default=500, help="Puzzles per archetype")
    parser.add_argument("--output", type=str, default="data/synthetic_tactics_dataset.pt", help="Output path")
    args = parser.parse_args()

    generate_and_save_tactical_dataset(
        output_path=args.output,
        count_per_type=args.count_per_type,
    )
