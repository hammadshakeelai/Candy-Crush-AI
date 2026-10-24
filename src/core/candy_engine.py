"""
High-Performance Match-3 Simulation Engine for Candy Crush.
Simulates cascades, gravity, special candies creation and combos.
"""

from typing import List, Tuple, Set, Optional, Dict
import numpy as np
import random
from .candy_board import (
    BoardState, CandyColor, SpecialType, BlockerType,
    Direction, DIRECTION_OFFSETS
)


class CandyEngine:
    """
    Simulates Candy Crush mechanics with high throughput (>20,000 steps/sec).
    """

    def __init__(self, seed: Optional[int] = None):
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

    def initialize_board(self, state: BoardState) -> None:
        """
        Fills the board with random candies ensuring NO initial matches of 3.
        """
        for r in range(state.height):
            for c in range(state.width):
                state.specials[r, c] = SpecialType.NORMAL
                state.blockers[r, c] = BlockerType.NONE
                
                # Pick color that doesn't create match with left or top neighbors
                forbidden = set()
                if c >= 2 and state.colors[r, c - 1] == state.colors[r, c - 2]:
                    forbidden.add(state.colors[r, c - 1])
                if r >= 2 and state.colors[r - 1, c] == state.colors[r - 2, c]:
                    forbidden.add(state.colors[r - 1, c])
                    
                valid_colors = [col for col in range(1, state.num_colors + 1) if col not in forbidden]
                state.colors[r, c] = random.choice(valid_colors)

    def get_legal_moves(self, state: BoardState) -> List[Tuple[int, int, Direction]]:
        """
        Finds all legal moves (swaps that result in at least one match or valid combo).
        Returns list of (row, col, direction).
        """
        legal = []
        for r in range(state.height):
            for c in range(state.width):
                # We only need to check RIGHT and DOWN to avoid redundant duplicate checks
                for direction in [Direction.RIGHT, Direction.DOWN]:
                    dr, dc = DIRECTION_OFFSETS[direction]
                    nr, nc = r + dr, c + dc
                    if state.is_valid_coord(nr, nc):
                        if self.is_legal_swap(state, r, c, nr, nc):
                            legal.append((r, c, direction))
        return legal

    def is_legal_swap(self, state: BoardState, r1: int, c1: int, r2: int, c2: int) -> bool:
        """
        Tests if swapping (r1, c1) and (r2, c2) creates a match or triggers special candy combo.
        """
        s1 = state.specials[r1, c1]
        s2 = state.specials[r2, c2]

        # Any combination of two special candies is a legal move
        if (s1 != SpecialType.NORMAL and s2 != SpecialType.NORMAL):
            return True

        # Color bomb with any candy is legal
        if s1 == SpecialType.COLOR_BOMB or s2 == SpecialType.COLOR_BOMB:
            return True

        # Perform temporary swap
        state.colors[r1, c1], state.colors[r2, c2] = state.colors[r2, c2], state.colors[r1, c1]
        has_match = self._has_match_at(state, r1, c1) or self._has_match_at(state, r2, c2)
        # Revert
        state.colors[r1, c1], state.colors[r2, c2] = state.colors[r2, c2], state.colors[r1, c1]

        return has_match

    def _has_match_at(self, state: BoardState, r: int, c: int) -> bool:
        color = state.colors[r, c]
        if color == CandyColor.EMPTY:
            return False

        # Horizontal check
        h_count = 1
        # Left
        sc = c - 1
        while sc >= 0 and state.colors[r, sc] == color:
            h_count += 1
            sc -= 1
        # Right
        sc = c + 1
        while sc < state.width and state.colors[r, sc] == color:
            h_count += 1
            sc += 1
        if h_count >= 3:
            return True

        # Vertical check
        v_count = 1
        # Up
        sr = r - 1
        while sr >= 0 and state.colors[sr, c] == color:
            v_count += 1
            sr -= 1
        # Down
        sr = r + 1
        while sr < state.height and state.colors[sr, c] == color:
            v_count += 1
            sr += 1
        return v_count >= 3

    def execute_move(
        self, state: BoardState, r1: int, c1: int, direction: Direction
    ) -> Tuple[int, int, Dict[str, Any]]:
        """
        Executes a swap, cascades the board until quiescence, and spawns new candies.
        Returns:
          (points_scored, jellies_cleared, stats_dict)
        """
        dr, dc = DIRECTION_OFFSETS[direction]
        r2, c2 = r1 + dr, c1 + dc

        if not state.is_valid_coord(r2, c2):
            return 0, 0, {"success": False}

        # Decrement moves left
        state.moves_left -= 1

        # Check special candy combos first
        combo_cleared, combo_score = self._check_special_combo(state, r1, c1, r2, c2)
        total_score = combo_score
        total_jellies = 0

        # Swap candies
        state.colors[r1, c1], state.colors[r2, c2] = state.colors[r2, c2], state.colors[r1, c1]
        state.specials[r1, c1], state.specials[r2, c2] = state.specials[r2, c2], state.specials[r1, c1]

        cascade_count = 0
        cleared_any = combo_cleared

        # Loop until no more matches (Cascades)
        while True:
            matches, specials_to_spawn = self._find_all_matches(state)
            if not matches:
                break

            cleared_any = True
            cascade_count += 1
            cascade_multiplier = 1.0 + (cascade_count * 0.5)

            # Clear candies and damage jellies
            round_score = 0
            for (mr, mc) in matches:
                round_score += 60
                state.colors[mr, mc] = CandyColor.EMPTY
                state.specials[mr, mc] = SpecialType.NORMAL
                
                # Jelly damage
                if state.jellies[mr, mc] > 0:
                    state.jellies[mr, mc] -= 1
                    total_jellies += 1
                    round_score += 1000

            total_score += int(round_score * cascade_multiplier)

            # Spawn special candies formed by 4-matches, 5-matches, L/T matches
            for (sr, sc, spec_type, col) in specials_to_spawn:
                state.specials[sr, sc] = spec_type
                state.colors[sr, sc] = col

            # Apply gravity and refill
            self._apply_gravity_and_refill(state)

        if not cleared_any:
            # Revert if illegal/no match occurred
            state.colors[r1, c1], state.colors[r2, c2] = state.colors[r2, c2], state.colors[r1, c1]
            state.specials[r1, c1], state.specials[r2, c2] = state.specials[r2, c2], state.specials[r1, c1]
            return 0, 0, {"success": False, "cascades": 0}

        state.score += total_score
        return total_score, total_jellies, {
            "success": True,
            "cascades": cascade_count,
            "score": total_score,
            "jellies": total_jellies,
        }

    def _check_special_combo(
        self, state: BoardState, r1: int, c1: int, r2: int, c2: int
    ) -> Tuple[bool, int]:
        """Handles special-to-special candy interactions."""
        s1, s2 = state.specials[r1, c1], state.specials[r2, c2]
        
        # Color bomb + Color bomb: Clears the entire board!
        if s1 == SpecialType.COLOR_BOMB and s2 == SpecialType.COLOR_BOMB:
            state.colors.fill(CandyColor.EMPTY)
            state.specials.fill(SpecialType.NORMAL)
            self._apply_gravity_and_refill(state)
            return True, 50000

        # Color bomb + Striped: All candies of striped color become striped & detonate!
        if (s1 == SpecialType.COLOR_BOMB and s2 in (SpecialType.STRIPED_HORIZONTAL, SpecialType.STRIPED_VERTICAL)) or \
           (s2 == SpecialType.COLOR_BOMB and s1 in (SpecialType.STRIPED_HORIZONTAL, SpecialType.STRIPED_VERTICAL)):
            target_color = state.colors[r2, c2] if s1 == SpecialType.COLOR_BOMB else state.colors[r1, c1]
            for r in range(state.height):
                for c in range(state.width):
                    if state.colors[r, c] == target_color:
                        state.specials[r, c] = random.choice([
                            SpecialType.STRIPED_HORIZONTAL, SpecialType.STRIPED_VERTICAL
                        ])
            state.colors[r1, c1] = CandyColor.EMPTY
            state.colors[r2, c2] = CandyColor.EMPTY
            return True, 15000

        # Color bomb + Normal: Clears all of that color
        if s1 == SpecialType.COLOR_BOMB or s2 == SpecialType.COLOR_BOMB:
            target_color = state.colors[r2, c2] if s1 == SpecialType.COLOR_BOMB else state.colors[r1, c1]
            for r in range(state.height):
                for c in range(state.width):
                    if state.colors[r, c] == target_color:
                        state.colors[r, c] = CandyColor.EMPTY
                        state.specials[r, c] = SpecialType.NORMAL
            state.colors[r1, c1] = CandyColor.EMPTY
            state.colors[r2, c2] = CandyColor.EMPTY
            return True, 5000

        return False, 0

    def _find_all_matches(
        self, state: BoardState
    ) -> Tuple[Set[Tuple[int, int]], List[Tuple[int, int, SpecialType, int]]]:
        """
        Identifies all current matching groups and detects special candy creations.
        """
        matched_cells: Set[Tuple[int, int]] = set()
        specials_to_spawn: List[Tuple[int, int, SpecialType, int]] = []

        # Find horizontal lines
        for r in range(state.height):
            c = 0
            while c < state.width:
                color = state.colors[r, c]
                if color == CandyColor.EMPTY:
                    c += 1
                    continue
                start_c = c
                while c < state.width and state.colors[r, c] == color:
                    c += 1
                length = c - start_c
                if length >= 3:
                    for col_idx in range(start_c, c):
                        matched_cells.add((r, col_idx))
                    # Special creation
                    mid_c = start_c + length // 2
                    if length == 4:
                        specials_to_spawn.append((r, mid_c, SpecialType.STRIPED_VERTICAL, color))
                    elif length >= 5:
                        specials_to_spawn.append((r, mid_c, SpecialType.COLOR_BOMB, CandyColor.EMPTY))

        # Find vertical lines
        for c in range(state.width):
            r = 0
            while r < state.height:
                color = state.colors[r, c]
                if color == CandyColor.EMPTY:
                    r += 1
                    continue
                start_r = r
                while r < state.height and state.colors[r, c] == color:
                    r += 1
                length = r - start_r
                if length >= 3:
                    for row_idx in range(start_r, r):
                        matched_cells.add((row_idx, c))
                    mid_r = start_r + length // 2
                    if length == 4:
                        specials_to_spawn.append((mid_r, c, SpecialType.STRIPED_HORIZONTAL, color))
                    elif length >= 5:
                        specials_to_spawn.append((mid_r, c, SpecialType.COLOR_BOMB, CandyColor.EMPTY))

        # Expand detonations for special candies inside matched_cells
        expanded = set(matched_cells)
        for (r, c) in matched_cells:
            spec = state.specials[r, c]
            if spec == SpecialType.STRIPED_HORIZONTAL:
                # Clears full row
                for cc in range(state.width):
                    expanded.add((r, cc))
            elif spec == SpecialType.STRIPED_VERTICAL:
                # Clears full col
                for rr in range(state.height):
                    expanded.add((rr, c))
            elif spec == SpecialType.WRAPPED:
                # 3x3 blast
                for dr in [-1, 0, 1]:
                    for dc in [-1, 0, 1]:
                        if state.is_valid_coord(r + dr, c + dc):
                            expanded.add((r + dr, c + dc))

        return expanded, specials_to_spawn

    def _apply_gravity_and_refill(self, state: BoardState) -> None:
        """
        Shifts candies downward to fill empty slots and spawns new candies at the top.
        """
        for c in range(state.width):
            write_idx = state.height - 1
            for r in range(state.height - 1, -1, -1):
                if state.colors[r, c] != CandyColor.EMPTY:
                    if r != write_idx:
                        state.colors[write_idx, c] = state.colors[r, c]
                        state.specials[write_idx, c] = state.specials[r, c]
                        state.colors[r, c] = CandyColor.EMPTY
                        state.specials[r, c] = SpecialType.NORMAL
                    write_idx -= 1

            # Refill empty slots from write_idx up to 0 with random candies
            for r in range(write_idx, -1, -1):
                state.colors[r, c] = random.randint(1, state.num_colors)
                state.specials[r, c] = SpecialType.NORMAL
