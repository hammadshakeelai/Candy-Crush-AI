"""
Advanced Synthetic Tactical Puzzle Generator (Part 2) for Grandmaster Curriculum Training.
Adds 8 brand-new Grandmaster Tactical Puzzle Archetypes (bringing the total to 12 distinct
tactical puzzle generators across `core/tactical_puzzles.py` and `core/advanced_tactical_puzzles.py`).

New Archetypes (5..12):
  5.  DOUBLE_COLOR_BOMB_BOARD_WIPE: Two adjacent Color Bombs surrounded by multi-layer jellies/blockers.
  6.  COLOR_BOMB_WRAPPED_DOUBLE_CLEAR: Color Bomb adjacent to Wrapped Candy clearing two full colors.
  7.  WRAPPED_STRIPED_3X3_CROSS_CANNON: Wrapped + Striped combo aligned with a 3x3 cross of jellies.
  8.  INGREDIENT_COLUMN_PARITY_SHIFT: Ingredient trapped above a licorice-locked column shifted to open lane.
  9.  DISPENSER_CORK_UNCAPPING: Frosting blocker at (1, 4) choking a top dispenser column.
  10. PORTAL_WORMHOLE_CHAIN_SNIPE: Disjoint chambers linked via portals from bottom-left to top-right.
  11. ENDGAME_BOTTOM_AVALANCHE_TRIGGER: Bottom-row match at r=8 dropping stacked pairs into a 3-step cascade.
  12. COLOR_BOMB_PRESERVATION_DISCIPLINE: Avoiding premature scarce-color bomb detonation by clearing blockers first.
"""

import os
import sys
import time
import random
from typing import Dict, Any, Tuple, List, Optional
import numpy as np
import torch

sys.path.insert(0, os.getcwd())
from core.king_level_loader import LevelConfig, LevelType
from core.reverse_engine import (
    ReverseEngineState, Direction, CandyColor, SpecialCandy, Blocker, Ingredient
)
from core.tactical_puzzles import TacticalPuzzleGenerator
from models.spatial_transformer_net import encode_spatial_board_tensor, encode_action_index


class AdvancedTacticalPuzzleGenerator:
    """Generates 8 advanced Grandmaster tactical puzzle archetypes (and unifies all 12 archetypes)."""

    def __init__(self, seed: Optional[int] = None):
        self.seed = seed
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        self.base_generator = TacticalPuzzleGenerator(seed=seed)

    def _safe_alternating_board(self, H: int = 9, W: int = 9) -> np.ndarray:
        """Creates a base board with zero unintended initial 3-matches."""
        colors = np.zeros((H, W), dtype=np.int32)
        offset = np.random.randint(0, 5)
        for r in range(H):
            for c in range(W):
                colors[r, c] = ((r * 2 + c + offset) % 5) + 1
        return colors

    def create_double_color_bomb_board_wipe_puzzle(
        self, puzzle_id: int = 91001
    ) -> Tuple[LevelConfig, Tuple[int, int, str]]:
        """
        Archetype 5: DOUBLE_COLOR_BOMB_BOARD_WIPE (Move limit: 1).
        Two adjacent Color Bombs surrounded by multi-layer jellies and blockers;
        swapping them together wipes the entire board in 1 move.
        """
        H, W = 9, 9
        colors = self._safe_alternating_board(H, W)
        specials = np.zeros((H, W), dtype=np.int32)
        jellies = np.full((H, W), 2, dtype=np.int32)
        blockers = np.zeros((H, W), dtype=np.int32)

        # Surround the perimeter and mid-ring with multi-layer blockers
        for r in range(H):
            for c in range(W):
                if r in (0, 8) or c in (0, 8):
                    blockers[r, c] = Blocker.FROSTING
                    colors[r, c] = CandyColor.EMPTY
                elif (r + c) % 3 == 0 and (r, c) not in [(4, 3), (4, 4)]:
                    blockers[r, c] = Blocker.LICORICE

        # Two adjacent Color Bombs at (4, 3) and (4, 4)
        blockers[4, 3] = Blocker.NONE
        blockers[4, 4] = Blocker.NONE
        colors[4, 3] = CandyColor.RED
        colors[4, 4] = CandyColor.BLUE
        specials[4, 3] = SpecialCandy.COLOR_BOMB
        specials[4, 4] = SpecialCandy.COLOR_BOMB

        optimal_action = (4, 3, "H")  # Swap (4, 3) with (4, 4) for complete board wipe

        cfg = LevelConfig(
            level_id=puzzle_id,
            level_type=LevelType.JELLY,
            moves_limit=1,
            target_score=60000,
            height=H,
            width=W,
            num_colors=5,
            active_grid=np.ones((H, W), dtype=bool),
            initial_jellies=jellies,
            initial_blockers=blockers,
            initial_ingredients=np.zeros((H, W), dtype=np.int32),
            initial_colors=colors,
            initial_specials=specials,
            target_ingredients=0
        )
        return cfg, optimal_action

    def create_color_bomb_wrapped_double_clear_puzzle(
        self, puzzle_id: int = 91002
    ) -> Tuple[LevelConfig, Tuple[int, int, str]]:
        """
        Archetype 6: COLOR_BOMB_WRAPPED_DOUBLE_CLEAR (Move limit: 1).
        Color Bomb adjacent to Wrapped Candy; swapping them clears two entire colors
        across the board and eliminates all scattered jellies on those colors.
        """
        H, W = 9, 9
        colors = self._safe_alternating_board(H, W)
        specials = np.zeros((H, W), dtype=np.int32)
        jellies = np.zeros((H, W), dtype=np.int32)
        blockers = np.zeros((H, W), dtype=np.int32)

        primary_color = CandyColor.RED
        secondary_color = CandyColor.BLUE

        # Place jellies on all cells holding primary or secondary color
        for r in range(H):
            for c in range(W):
                if (r + c) % 2 == 0:
                    colors[r, c] = primary_color if (r % 2 == 0) else secondary_color
                    jellies[r, c] = 1

        # Color Bomb at (4, 4) adjacent to Wrapped Candy at (4, 5)
        colors[4, 4] = primary_color
        specials[4, 4] = SpecialCandy.COLOR_BOMB
        jellies[4, 4] = 1

        colors[4, 5] = primary_color
        specials[4, 5] = SpecialCandy.WRAPPED
        jellies[4, 5] = 1

        optimal_action = (4, 4, "H")  # Swap Color Bomb with Wrapped Candy

        cfg = LevelConfig(
            level_id=puzzle_id,
            level_type=LevelType.JELLY,
            moves_limit=1,
            target_score=35000,
            height=H,
            width=W,
            num_colors=5,
            active_grid=np.ones((H, W), dtype=bool),
            initial_jellies=jellies,
            initial_blockers=blockers,
            initial_ingredients=np.zeros((H, W), dtype=np.int32),
            initial_colors=colors,
            initial_specials=specials,
            target_ingredients=0
        )
        return cfg, optimal_action

    def create_wrapped_striped_3x3_cross_cannon_puzzle(
        self, puzzle_id: int = 91003
    ) -> Tuple[LevelConfig, Tuple[int, int, str]]:
        """
        Archetype 7: WRAPPED_STRIPED_3X3_CROSS_CANNON (Move limit: 1).
        Wrapped Candy adjacent to Striped Candy aligned with a 3x3 cross of jellies
        along rows r-1..r+1 and cols c-1..c+1.
        """
        H, W = 9, 9
        colors = self._safe_alternating_board(H, W)
        specials = np.zeros((H, W), dtype=np.int32)
        jellies = np.zeros((H, W), dtype=np.int32)
        blockers = np.zeros((H, W), dtype=np.int32)

        center_r, center_c = 4, 4

        # 3-row by 3-column cross of jellies aligned with (center_r, center_c)
        for rr in [center_r - 1, center_r, center_r + 1]:
            for cc in range(W):
                jellies[rr, cc] = 1
        for cc in [center_c - 1, center_c, center_c + 1]:
            for rr in range(H):
                jellies[rr, cc] = 1

        # Wrapped Candy at (4, 3) and Striped Candy at (4, 4)
        colors[center_r, center_c - 1] = CandyColor.GREEN
        specials[center_r, center_c - 1] = SpecialCandy.WRAPPED

        colors[center_r, center_c] = CandyColor.YELLOW
        specials[center_r, center_c] = SpecialCandy.STRIPED_HORIZONTAL

        optimal_action = (center_r, center_c - 1, "H")  # Fire giant 3x3 cross cannon centered at (4, 4)

        cfg = LevelConfig(
            level_id=puzzle_id,
            level_type=LevelType.JELLY,
            moves_limit=1,
            target_score=20000,
            height=H,
            width=W,
            num_colors=5,
            active_grid=np.ones((H, W), dtype=bool),
            initial_jellies=jellies,
            initial_blockers=blockers,
            initial_ingredients=np.zeros((H, W), dtype=np.int32),
            initial_colors=colors,
            initial_specials=specials,
            target_ingredients=0
        )
        return cfg, optimal_action

    def create_ingredient_column_parity_shift_puzzle(
        self, puzzle_id: int = 91004
    ) -> Tuple[LevelConfig, Tuple[int, int, str]]:
        """
        Archetype 8: INGREDIENT_COLUMN_PARITY_SHIFT (Move limit: 2).
        An Ingredient is trapped at (3, 4) above a licorice/frosting-locked column 4;
        only a horizontal swap at (3, 4) shifts it into open drop lane column 5 to exit.
        """
        H, W = 9, 9
        colors = self._safe_alternating_board(H, W)
        specials = np.zeros((H, W), dtype=np.int32)
        jellies = np.zeros((H, W), dtype=np.int32)
        blockers = np.zeros((H, W), dtype=np.int32)
        ingredients = np.zeros((H, W), dtype=np.int32)

        # Lock column 4 below row 3 with licorice/frosting so the ingredient cannot drop down col 4
        for r in range(4, H):
            blockers[r, 4] = Blocker.LICORICE

        # Place trapped Cherry Ingredient at (3, 4)
        ingredients[3, 4] = Ingredient.CHERRY
        colors[3, 4] = CandyColor.RED

        # Setup horizontal swap (3, 4) <-> (3, 5) so (3, 5) completes a 3-match in col 4 at (1,4),(2,4),(3,4)
        match_color = CandyColor.BLUE
        colors[1, 4] = match_color
        colors[2, 4] = match_color
        colors[3, 5] = match_color

        # Ensure column 5 below (3, 5) is completely open and has a vertical striped candy ready for Move 2 drop
        for r in range(4, H):
            blockers[r, 5] = Blocker.NONE
        specials[7, 5] = SpecialCandy.STRIPED_VERTICAL
        colors[7, 5] = CandyColor.GREEN
        colors[7, 6] = CandyColor.GREEN
        colors[7, 7] = CandyColor.GREEN

        optimal_action = (3, 4, "H")  # Shift Ingredient from locked col 4 into open drop lane col 5

        cfg = LevelConfig(
            level_id=puzzle_id,
            level_type=LevelType.INGREDIENTS,
            moves_limit=2,
            target_score=10000,
            height=H,
            width=W,
            num_colors=5,
            active_grid=np.ones((H, W), dtype=bool),
            initial_jellies=jellies,
            initial_blockers=blockers,
            initial_ingredients=ingredients,
            initial_colors=colors,
            initial_specials=specials,
            target_ingredients=1
        )
        return cfg, optimal_action

    def create_dispenser_cork_uncapping_puzzle(
        self, puzzle_id: int = 91005
    ) -> Tuple[LevelConfig, Tuple[int, int, str]]:
        """
        Archetype 9: DISPENSER_CORK_UNCAPPING (Move limit: 1).
        A frosting blocker sits at (1, 4) directly choking a top dispenser column;
        breaking it via an adjacent match at row 2 uncaps the column and refills the lane.
        """
        H, W = 9, 9
        colors = self._safe_alternating_board(H, W)
        specials = np.zeros((H, W), dtype=np.int32)
        jellies = np.zeros((H, W), dtype=np.int32)
        blockers = np.zeros((H, W), dtype=np.int32)

        # Frosting cork at (1, 4) choking column 4
        blockers[1, 4] = Blocker.FROSTING
        colors[1, 4] = CandyColor.EMPTY

        # Starved cells in column 4 beneath the cork
        for r in range(3, H):
            colors[r, 4] = CandyColor.EMPTY

        # Create horizontal 3-match at (2, 2), (2, 3), (2, 4) by swapping (2, 4) with (2, 5)
        cork_breaker_color = CandyColor.YELLOW
        colors[2, 2] = cork_breaker_color
        colors[2, 3] = cork_breaker_color
        colors[2, 4] = CandyColor.RED
        colors[2, 5] = cork_breaker_color
        jellies[2, 2] = 1
        jellies[2, 3] = 1
        jellies[2, 4] = 1

        optimal_action = (2, 4, "H")  # Swap (2, 4) with (2, 5) to match at (2, 4) and shatter frosting at (1, 4)

        cfg = LevelConfig(
            level_id=puzzle_id,
            level_type=LevelType.BLOCKERS,
            moves_limit=1,
            target_score=3000,
            height=H,
            width=W,
            num_colors=5,
            active_grid=np.ones((H, W), dtype=bool),
            initial_jellies=jellies,
            initial_blockers=blockers,
            initial_ingredients=np.zeros((H, W), dtype=np.int32),
            initial_colors=colors,
            initial_specials=specials,
            target_ingredients=0,
            cannons=[[], [], [], [], [71], [], [], [], []],
            ammo_cannons=[0, 0, 0, 0, 5, 0, 0, 0, 0]
        )
        return cfg, optimal_action

    def create_portal_wormhole_chain_snipe_puzzle(
        self, puzzle_id: int = 91006
    ) -> Tuple[LevelConfig, Tuple[int, int, str]]:
        """
        Archetype 10: PORTAL_WORMHOLE_CHAIN_SNIPE (Move limit: 1).
        Level with `portals` connecting bottom of Left Chamber (cols 0..3, row 8)
        to top of Right Chamber (cols 5..8, row 0) across an impassable void column 4.
        """
        H, W = 9, 9
        colors = self._safe_alternating_board(H, W)
        specials = np.zeros((H, W), dtype=np.int32)
        jellies = np.zeros((H, W), dtype=np.int32)
        blockers = np.zeros((H, W), dtype=np.int32)
        active_grid = np.ones((H, W), dtype=bool)

        # Divide Left Chamber (cols 0..3) and Right Chamber (cols 5..8) with void wall at col 4
        active_grid[:, 4] = False
        colors[:, 4] = CandyColor.EMPTY

        # Portals linking bottom of Left Chamber (r=8, c=0..3) -> top of Right Chamber (r=0, c=5..8)
        portals = [
            [[c, 8, 14], [c + 5, 0, 14]]
            for c in range(4)
        ]

        # Target jelly in Right Chamber at (7, 7) aligned with Horizontal Striped laser in Left Chamber at (7, 2)
        jellies[7, 7] = 1
        jellies[7, 1] = 1
        jellies[7, 2] = 1

        snipe_color = CandyColor.ORANGE
        colors[7, 0] = snipe_color
        colors[7, 1] = snipe_color
        colors[7, 2] = CandyColor.RED
        colors[7, 3] = snipe_color
        specials[7, 1] = SpecialCandy.STRIPED_HORIZONTAL

        optimal_action = (7, 2, "H")  # Trigger portal-linked chamber snipe from Left Chamber

        cfg = LevelConfig(
            level_id=puzzle_id,
            level_type=LevelType.JELLY,
            moves_limit=1,
            target_score=10000,
            height=H,
            width=W,
            num_colors=5,
            active_grid=active_grid,
            initial_jellies=jellies,
            initial_blockers=blockers,
            initial_ingredients=np.zeros((H, W), dtype=np.int32),
            initial_colors=colors,
            initial_specials=specials,
            target_ingredients=0,
            portals=portals
        )
        return cfg, optimal_action

    def create_endgame_bottom_avalanche_trigger_puzzle(
        self, puzzle_id: int = 91007
    ) -> Tuple[LevelConfig, Tuple[int, int, str]]:
        """
        Archetype 11: ENDGAME_BOTTOM_AVALANCHE_TRIGGER (Move limit: 1).
        A bottom-row match at r=8 drops 3 stacked pairs in rows 4..7 into a
        deterministic 3-step gravitational cascade.
        """
        H, W = 9, 9
        colors = self._safe_alternating_board(H, W)
        specials = np.zeros((H, W), dtype=np.int32)
        jellies = np.zeros((H, W), dtype=np.int32)
        blockers = np.zeros((H, W), dtype=np.int32)

        # Step 1: Bottom-row match at r=8 across (8, 3), (8, 4), (8, 5) triggered by swapping (8, 2) <-> (8, 3)
        c_step1 = CandyColor.RED
        colors[8, 2] = c_step1
        colors[8, 3] = CandyColor.BLUE
        colors[8, 4] = c_step1
        colors[8, 5] = c_step1
        jellies[8, 3] = 1
        jellies[8, 4] = 1
        jellies[8, 5] = 1

        # Step 2 (Cascade 1): When (8,3..5) clears, (6, 3) drops to (7, 3) to match stationary (7, 1) and (7, 2)
        c_step2 = CandyColor.GREEN
        colors[7, 1] = c_step2
        colors[7, 2] = c_step2
        colors[6, 3] = c_step2
        jellies[7, 1] = 1
        jellies[7, 2] = 1
        jellies[7, 3] = 1

        # Step 3 (Cascade 2): When (7,1..3) clears, (4, 3) (which dropped to 5,3 then 6,3) matches (5, 4) & (5, 5) (dropped to 6,4 & 6,5)
        c_step3 = CandyColor.YELLOW
        colors[4, 3] = c_step3
        colors[5, 4] = c_step3
        colors[5, 5] = c_step3
        jellies[6, 3] = 1
        jellies[6, 4] = 1
        jellies[6, 5] = 1

        optimal_action = (8, 2, "H")  # Bottom-row match at r=8 igniting the 3-step avalanche

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
            initial_ingredients=np.zeros((H, W), dtype=np.int32),
            initial_colors=colors,
            initial_specials=specials,
            target_ingredients=0
        )
        return cfg, optimal_action

    def create_color_bomb_preservation_discipline_puzzle(
        self, puzzle_id: int = 91008
    ) -> Tuple[LevelConfig, Tuple[int, int, str]]:
        """
        Archetype 12: COLOR_BOMB_PRESERVATION_DISCIPLINE (Move limit: 2).
        A naive match on the left would accidentally detonate a Color Bomb at (4, 1) on a
        scarce non-objective color; the winning move clears frosting blockers on the right
        side (4, 6) first so the Color Bomb is preserved to sweep all remaining jellies on Move 2.
        """
        H, W = 9, 9
        colors = self._safe_alternating_board(H, W)
        specials = np.zeros((H, W), dtype=np.int32)
        jellies = np.zeros((H, W), dtype=np.int32)
        blockers = np.zeros((H, W), dtype=np.int32)

        # Color Bomb at (4, 1) next to a scarce decoy color (ORANGE) at (4, 0)
        specials[4, 1] = SpecialCandy.COLOR_BOMB
        colors[4, 1] = CandyColor.RED
        colors[4, 0] = CandyColor.ORANGE

        # Abundant objective jellies on BLUE candies across top and bottom rows
        for c in range(W):
            colors[0, c] = CandyColor.BLUE
            jellies[0, c] = 1
            colors[8, c] = CandyColor.BLUE
            jellies[8, c] = 1
        # Keep (4, 2) BLUE so Color Bomb can swipe it on Move 2
        colors[4, 2] = CandyColor.BLUE

        # Frosting blocker on the right side at (4, 8) shielding a jelly, breakable via match at (4, 5..7)
        blockers[4, 8] = Blocker.FROSTING
        colors[4, 8] = CandyColor.EMPTY
        jellies[4, 7] = 1

        # Right-side blocker-clearing match: swap (4, 5) with (4, 6) to match vertically at (3,6), (4,6), (5,6) or horizontally at (4,6), (4,7)
        clear_color = CandyColor.GREEN
        colors[3, 7] = clear_color
        colors[5, 7] = clear_color
        colors[4, 6] = clear_color
        colors[4, 7] = CandyColor.YELLOW

        optimal_action = (4, 6, "H")  # Swap (4, 6) with (4, 7) to clear right-side blocker first while preserving Color Bomb

        cfg = LevelConfig(
            level_id=puzzle_id,
            level_type=LevelType.JELLY,
            moves_limit=2,
            target_score=25000,
            height=H,
            width=W,
            num_colors=5,
            active_grid=np.ones((H, W), dtype=bool),
            initial_jellies=jellies,
            initial_blockers=blockers,
            initial_ingredients=np.zeros((H, W), dtype=np.int32),
            initial_colors=colors,
            initial_specials=specials,
            target_ingredients=0
        )
        return cfg, optimal_action

    def get_advanced_8_archetypes(self) -> List[Tuple[str, Any]]:
        """Returns the 8 new Grandmaster tactical puzzle generators."""
        return [
            ("DOUBLE_COLOR_BOMB_BOARD_WIPE", self.create_double_color_bomb_board_wipe_puzzle),
            ("COLOR_BOMB_WRAPPED_DOUBLE_CLEAR", self.create_color_bomb_wrapped_double_clear_puzzle),
            ("WRAPPED_STRIPED_3X3_CROSS_CANNON", self.create_wrapped_striped_3x3_cross_cannon_puzzle),
            ("INGREDIENT_COLUMN_PARITY_SHIFT", self.create_ingredient_column_parity_shift_puzzle),
            ("DISPENSER_CORK_UNCAPPING", self.create_dispenser_cork_uncapping_puzzle),
            ("PORTAL_WORMHOLE_CHAIN_SNIPE", self.create_portal_wormhole_chain_snipe_puzzle),
            ("ENDGAME_BOTTOM_AVALANCHE_TRIGGER", self.create_endgame_bottom_avalanche_trigger_puzzle),
            ("COLOR_BOMB_PRESERVATION_DISCIPLINE", self.create_color_bomb_preservation_discipline_puzzle),
        ]

    def get_all_12_archetypes(self) -> List[Tuple[str, Any]]:
        """Returns all 12 Grandmaster tactical puzzle generators (4 base + 8 advanced)."""
        base_4 = [
            ("COLOR_BOMB_STRIPED_ALIGNMENT", self.base_generator.create_colorbomb_striped_alignment_puzzle),
            ("CHOCOLATE_CONTAINMENT", self.base_generator.create_chocolate_containment_puzzle),
            ("CORNER_JELLY_SNIPE", self.base_generator.create_corner_jelly_snipe_puzzle),
            ("WRAPPED_DUAL_BLAST", self.base_generator.create_wrapped_dual_blast_puzzle),
        ]
        return base_4 + self.get_advanced_8_archetypes()

    def generate_batch(self, count_per_type: int = 250) -> List[Dict[str, Any]]:
        """Generates a batch across the 8 new advanced tactical archetypes."""
        puzzles = []
        pid = 91001
        for trick_name, gen_func in self.get_advanced_8_archetypes():
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

    def generate_all_12_batch(self, count_per_type: int = 250) -> List[Dict[str, Any]]:
        """Generates a complete batch across all 12 Grandmaster tactical archetypes."""
        puzzles = []
        pid = 90001
        for trick_name, gen_func in self.get_all_12_archetypes():
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


def generate_all_12_archetypes_dataset(
    output_path: str = "data/synthetic_tactics_12archetypes.pt",
    count_per_type: int = 250
) -> Dict[str, Any]:
    """
    Generates 3,000 total tactical demonstration tensors (250 across all 12 archetypes)
    and saves them to `data/synthetic_tactics_12archetypes.pt`.
    """
    t0 = time.time()
    print("=" * 72)
    print("   12-ARCHETYPE GRANDMASTER TACTICAL DATASET FACTORY")
    print("=" * 72)

    if os.path.dirname(output_path):
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

    generator = AdvancedTacticalPuzzleGenerator(seed=42)
    puzzles = generator.generate_all_12_batch(count_per_type=count_per_type)
    total_puzzles = len(puzzles)

    states = np.empty((total_puzzles, 20, 9, 9), dtype=np.float32)
    actions = np.empty((total_puzzles,), dtype=np.int64)
    win_targets = np.ones((total_puzzles, 1), dtype=np.float32)
    move_ratios = np.ones((total_puzzles, 1), dtype=np.float32)
    archetypes: List[str] = []

    for idx, p in enumerate(puzzles):
        cfg = p["config"]
        r, c, d_str = p["optimal_action"]
        trick = p["trick_name"]

        state = ReverseEngineState(cfg)
        states[idx] = encode_spatial_board_tensor(state)
        direction = Direction.RIGHT if d_str == "H" else Direction.DOWN
        actions[idx] = encode_action_index(r, c, direction)
        archetypes.append(trick)

    dataset = {
        "states": torch.from_numpy(states),
        "actions": torch.from_numpy(actions),
        "win_targets": torch.from_numpy(win_targets),
        "move_ratios": torch.from_numpy(move_ratios),
        "archetypes": archetypes,
        "sample_count": total_puzzles
    }
    torch.save(dataset, output_path)
    elapsed = time.time() - t0
    file_size_mb = os.path.getsize(output_path) / (1024 * 1024)

    archetype_counts: Dict[str, int] = {}
    for a in archetypes:
        archetype_counts[a] = archetype_counts.get(a, 0) + 1

    print(f"[INFO] Generated {total_puzzles:,} puzzles across {len(archetype_counts)} archetypes in {elapsed:.3f}s:")
    for i, (arch_name, cnt) in enumerate(archetype_counts.items(), 1):
        print(f"  {i:2d}. {arch_name:<38s} : {cnt:4d} tensors")
    print(f"[SUCCESS] Saved to {output_path} ({file_size_mb:.2f} MB, elapsed={elapsed:.3f}s)")
    print("=" * 72)
    return dataset


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate 12-Archetype Tactical Dataset")
    parser.add_argument("--count-per-type", type=int, default=250, help="Puzzles per archetype")
    parser.add_argument("--output", type=str, default="data/synthetic_tactics_12archetypes.pt", help="Output path")
    args = parser.parse_args()

    generate_all_12_archetypes_dataset(
        output_path=args.output,
        count_per_type=args.count_per_type,
    )
