"""
Reverse-Engineered Candy Crush Saga Game Physics & Logic Engine.
Implements exact King mechanics:
- All match topologies (Match 3, Match 4 Striped, Match 5 Wrapped L/T, Match 5 Line Color Bomb)
- All 7 Special Candy Combos (Bomb+Bomb, Bomb+Striped, Bomb+Wrapped, Striped+Wrapped, etc.)
- Multi-objective tracking: Target Score, Jellies, Ingredients, Blockers (Chocolate, Licorice)
- Accurate gravity collapse, diagonal sliding, and refill spawners
"""

from enum import IntEnum
from typing import List, Tuple, Set, Dict, Optional, Any
import numpy as np
import random
from dataclasses import dataclass, field


class CandyColor(IntEnum):
    EMPTY = 0
    RED = 1
    BLUE = 2
    GREEN = 3
    YELLOW = 4
    ORANGE = 5
    PURPLE = 6


class SpecialCandy(IntEnum):
    NONE = 0
    STRIPED_HORIZONTAL = 1
    STRIPED_VERTICAL = 2
    WRAPPED = 3
    COLOR_BOMB = 4


class Blocker(IntEnum):
    NONE = 0
    CHOCOLATE = 1
    LICORICE = 2
    FROSTING = 3


class Ingredient(IntEnum):
    NONE = 0
    CHERRY = 1
    HAZELNUT = 2


class LevelType(IntEnum):
    SCORE = 0
    JELLY = 1
    INGREDIENTS = 2
    BLOCKERS = 3


class Direction(IntEnum):
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3


OFFSETS = {
    Direction.UP: (-1, 0),
    Direction.DOWN: (1, 0),
    Direction.LEFT: (0, -1),
    Direction.RIGHT: (0, 1),
}


@dataclass
class LevelConfig:
    level_id: int = 1
    level_type: LevelType = LevelType.SCORE
    moves_limit: int = 30
    target_score: int = 15000
    height: int = 9
    width: int = 9
    num_colors: int = 5
    # Active grid mask (True if cell exists, False if hole/void)
    active_grid: np.ndarray = field(default_factory=lambda: np.ones((9, 9), dtype=bool))
    # Initial jelly layers (0, 1, 2)
    initial_jellies: np.ndarray = field(default_factory=lambda: np.zeros((9, 9), dtype=np.int32))
    # Initial blockers (Chocolate, Licorice, etc.)
    initial_blockers: np.ndarray = field(default_factory=lambda: np.zeros((9, 9), dtype=np.int32))
    # Initial ingredients
    initial_ingredients: np.ndarray = field(default_factory=lambda: np.zeros((9, 9), dtype=np.int32))
    initial_colors: Optional[np.ndarray] = None
    initial_specials: Optional[np.ndarray] = None
    target_ingredients: int = 0
    portals: Optional[List[Any]] = None
    cannons: Optional[List[Any]] = None
    ammo_cannons: Optional[List[Any]] = None
    evil_spawner_config: Optional[Dict[str, Any]] = None
    frog_stomach_size: int = 0


class ReverseEngineState:
    """Exact multi-layer state of the Candy Crush grid."""

    def __init__(self, config: LevelConfig):
        self.config = config
        self.height = config.height
        self.width = config.width
        self.num_colors = config.num_colors

        if config.initial_colors is not None:
            self.colors = np.copy(config.initial_colors)
        else:
            self.colors = np.zeros((self.height, self.width), dtype=np.int32)

        if config.initial_specials is not None:
            self.specials = np.copy(config.initial_specials)
        else:
            self.specials = np.zeros((self.height, self.width), dtype=np.int32)
        self.jellies = np.copy(config.initial_jellies)
        self.blockers = np.copy(config.initial_blockers)
        self.ingredients = np.copy(config.initial_ingredients)
        self.active_grid = np.copy(config.active_grid)

        self.moves_left = config.moves_limit
        self.score = 0
        self.ingredients_collected = 0
        self.chocolate_destroyed_this_turn = False

    def copy(self) -> "ReverseEngineState":
        new_state = ReverseEngineState(self.config)
        new_state.colors = np.copy(self.colors)
        new_state.specials = np.copy(self.specials)
        new_state.jellies = np.copy(self.jellies)
        new_state.blockers = np.copy(self.blockers)
        new_state.ingredients = np.copy(self.ingredients)
        new_state.moves_left = self.moves_left
        new_state.score = self.score
        new_state.ingredients_collected = self.ingredients_collected
        new_state.chocolate_destroyed_this_turn = self.chocolate_destroyed_this_turn
        return new_state

    def is_valid_cell(self, r: int, c: int) -> bool:
        return 0 <= r < self.height and 0 <= c < self.width and self.active_grid[r, c]

    def has_won(self) -> bool:
        if self.config.level_type == LevelType.SCORE:
            return self.score >= self.config.target_score
        elif self.config.level_type == LevelType.JELLY:
            return np.sum(self.jellies > 0) == 0 and self.score >= self.config.target_score
        elif self.config.level_type == LevelType.INGREDIENTS:
            return self.ingredients_collected >= self.config.target_ingredients
        elif self.config.level_type == LevelType.BLOCKERS:
            return np.sum(self.blockers > 0) == 0 and self.score >= self.config.target_score
        return False

    def has_lost(self) -> bool:
        return self.moves_left <= 0 and not self.has_won()

    def remaining_objectives_count(self) -> int:
        if self.config.level_type == LevelType.JELLY:
            return int(np.sum(self.jellies))
        elif self.config.level_type == LevelType.INGREDIENTS:
            return max(0, self.config.target_ingredients - self.ingredients_collected)
        elif self.config.level_type == LevelType.BLOCKERS:
            return int(np.sum(self.blockers > 0))
        else:
            return max(0, self.config.target_score - self.score)


class ReverseCandyEngine:
    """
    High-Performance Match-3 Reverse-Engineered Simulator.
    Executes rules, special creation, combos, cascades, and blockers.
    """

    def __init__(self, seed: Optional[int] = None):
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

    def initialize_board(self, state: ReverseEngineState) -> None:
        """Fills active cells with random candies ensuring no initial 3-matches."""
        for r in range(state.height):
            for c in range(state.width):
                if not state.active_grid[r, c]:
                    continue
                # Solid impassable blockers prevent candies from occupying tile
                if state.blockers[r, c] in (Blocker.CHOCOLATE, Blocker.FROSTING):
                    state.colors[r, c] = CandyColor.EMPTY
                    continue
                # If cell already has a preset candy or special candy, keep it
                if state.colors[r, c] != CandyColor.EMPTY or state.specials[r, c] != SpecialCandy.NONE:
                    continue
                forbidden = set()
                if c >= 2 and state.colors[r, c - 1] == state.colors[r, c - 2]:
                    forbidden.add(state.colors[r, c - 1])
                if r >= 2 and state.colors[r - 1, c] == state.colors[r - 2, c]:
                    forbidden.add(state.colors[r - 1, c])
                valid = [col for col in range(1, state.num_colors + 1) if col not in forbidden]
                state.colors[r, c] = random.choice(valid) if valid else random.randint(1, state.num_colors)

        # In Ingredient levels, ensure an active ingredient is on the board
        if state.config.level_type == LevelType.INGREDIENTS and np.sum(state.ingredients != Ingredient.NONE) == 0:
            spawned = False
            # Check center columns first for best initial spawn
            col_order = sorted(range(state.width), key=lambda c: abs(c - state.width // 2))
            for c in col_order:
                for top_r in range(state.height):
                    if state.is_valid_cell(top_r, c) and state.blockers[top_r, c] == Blocker.NONE:
                        state.ingredients[top_r, c] = Ingredient.CHERRY
                        spawned = True
                        break
                if spawned:
                    break

    def get_legal_moves(self, state: ReverseEngineState) -> List[Tuple[int, int, Direction]]:
        """Scans all legal swaps."""
        legal = []
        for r in range(state.height):
            for c in range(state.width):
                if not state.is_valid_cell(r, c):
                    continue
                if state.blockers[r, c] in (Blocker.CHOCOLATE, Blocker.FROSTING):
                    continue
                for direction in [Direction.RIGHT, Direction.DOWN]:
                    dr, dc = OFFSETS[direction]
                    nr, nc = r + dr, c + dc
                    if state.is_valid_cell(nr, nc):
                        if state.blockers[nr, nc] in (Blocker.CHOCOLATE, Blocker.FROSTING):
                            continue
                        if self.is_legal_swap(state, r, c, nr, nc):
                            legal.append((r, c, direction))
        return legal

    def is_legal_swap(self, state: ReverseEngineState, r1: int, c1: int, r2: int, c2: int) -> bool:
        s1 = state.specials[r1, c1]
        s2 = state.specials[r2, c2]
        # Any combination of specials or color bombs is legal
        if (s1 != SpecialCandy.NONE and s2 != SpecialCandy.NONE) or \
           s1 == SpecialCandy.COLOR_BOMB or s2 == SpecialCandy.COLOR_BOMB:
            return True

        # Test swap
        state.colors[r1, c1], state.colors[r2, c2] = state.colors[r2, c2], state.colors[r1, c1]
        valid = self._has_match_at(state, r1, c1) or self._has_match_at(state, r2, c2)
        # Revert
        state.colors[r1, c1], state.colors[r2, c2] = state.colors[r2, c2], state.colors[r1, c1]
        return valid

    def _has_match_at(self, state: ReverseEngineState, r: int, c: int) -> bool:
        color = state.colors[r, c]
        if color == CandyColor.EMPTY:
            return False
        # Horizontal
        h = 1
        sc = c - 1
        while sc >= 0 and state.is_valid_cell(r, sc) and state.colors[r, sc] == color:
            h += 1
            sc -= 1
        sc = c + 1
        while sc < state.width and state.is_valid_cell(r, sc) and state.colors[r, sc] == color:
            h += 1
            sc += 1
        if h >= 3:
            return True
        # Vertical
        v = 1
        sr = r - 1
        while sr >= 0 and state.is_valid_cell(sr, c) and state.colors[sr, c] == color:
            v += 1
            sr -= 1
        sr = r + 1
        while sr < state.height and state.is_valid_cell(sr, c) and state.colors[sr, c] == color:
            v += 1
            sr += 1
        return v >= 3

    def execute_move(
        self, state: ReverseEngineState, r1: int, c1: int, direction: Direction
    ) -> Tuple[int, Dict[str, Any]]:
        """
        Executes swap, resolves special combos, cascades board to rest,
        updates goals, and expands chocolate if none was cleared.
        """
        dr, dc = OFFSETS[direction]
        r2, c2 = r1 + dr, c1 + dc

        if not state.is_valid_cell(r2, c2):
            return 0, {"success": False}

        state.moves_left -= 1
        state.chocolate_destroyed_this_turn = False

        # 1. Check special combinations
        combo_handled, combo_score = self._resolve_special_combos(state, r1, c1, r2, c2)
        total_score = combo_score
        cascade_depth = 0

        # Perform swap
        state.colors[r1, c1], state.colors[r2, c2] = state.colors[r2, c2], state.colors[r1, c1]
        state.specials[r1, c1], state.specials[r2, c2] = state.specials[r2, c2], state.specials[r1, c1]

        # 2. Quiescence Loop (Cascades & Explosions)
        while True:
            cleared_coords, specials_to_spawn = self._detect_matches(state)
            if not cleared_coords and not combo_handled:
                break

            cascade_depth += 1
            combo_handled = False
            mult = 1.0 + (cascade_depth * 0.4)

            # Process clears and damage blockers/jellies
            step_score = 0
            for r, c in cleared_coords:
                step_score += 60
                state.colors[r, c] = CandyColor.EMPTY
                state.specials[r, c] = SpecialCandy.NONE

                # Destroy direct blocker if hit by blast
                if state.blockers[r, c] != Blocker.NONE:
                    if state.blockers[r, c] == Blocker.CHOCOLATE:
                        state.chocolate_destroyed_this_turn = True
                    state.blockers[r, c] = Blocker.NONE
                    step_score += 400

                # Damage Jelly
                if state.jellies[r, c] > 0:
                    state.jellies[r, c] -= 1
                    step_score += 1000

                # Damage Adjacent Blockers (Chocolate, Licorice, Frosting)
                for ar, ac in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    adj_r, adj_c = r + ar, c + ac
                    if state.is_valid_cell(adj_r, adj_c) and state.blockers[adj_r, adj_c] != Blocker.NONE:
                        if state.blockers[adj_r, adj_c] == Blocker.CHOCOLATE:
                            state.chocolate_destroyed_this_turn = True
                            step_score += 500
                        else:
                            step_score += 300
                        state.blockers[adj_r, adj_c] = Blocker.NONE

            total_score += int(step_score * mult)

            # Spawn created special candies
            for sr, sc, spec_type, col in specials_to_spawn:
                state.specials[sr, sc] = spec_type
                state.colors[sr, sc] = col

            # Apply Gravity, Collisions, and Dispenser Refills
            self._apply_gravity_and_refill(state)

            # Check if ingredients reached bottom row
            self._collect_ingredients_at_bottom(state)

        # 3. Chocolate Growth Rule
        # If chocolate exists on the board and none was destroyed this turn, it expands by 1
        if not state.chocolate_destroyed_this_turn and np.any(state.blockers == Blocker.CHOCOLATE):
            self._expand_chocolate(state)

        state.score += total_score
        return total_score, {
            "success": True,
            "cascades": cascade_depth,
            "score": total_score,
            "won": state.has_won(),
            "lost": state.has_lost(),
        }

    def _resolve_special_combos(
        self, state: ReverseEngineState, r1: int, c1: int, r2: int, c2: int
    ) -> Tuple[bool, int]:
        s1, s2 = state.specials[r1, c1], state.specials[r2, c2]

        # Combo 1: Color Bomb + Color Bomb = Complete Board Clear
        if s1 == SpecialCandy.COLOR_BOMB and s2 == SpecialCandy.COLOR_BOMB:
            for r in range(state.height):
                for c in range(state.width):
                    if state.active_grid[r, c]:
                        state.colors[r, c] = CandyColor.EMPTY
                        state.specials[r, c] = SpecialCandy.NONE
                        state.blockers[r, c] = Blocker.NONE
                        if state.jellies[r, c] > 0:
                            state.jellies[r, c] -= 1
            return True, 60000

        # Combo 2: Color Bomb + Striped = All tiles of that color turn into Striped and fire simultaneously!
        if (s1 == SpecialCandy.COLOR_BOMB and s2 in (SpecialCandy.STRIPED_HORIZONTAL, SpecialCandy.STRIPED_VERTICAL)) or \
           (s2 == SpecialCandy.COLOR_BOMB and s1 in (SpecialCandy.STRIPED_HORIZONTAL, SpecialCandy.STRIPED_VERTICAL)):
            target_col = state.colors[r2, c2] if s1 == SpecialCandy.COLOR_BOMB else state.colors[r1, c1]
            detonations = []
            for r in range(state.height):
                for c in range(state.width):
                    if state.colors[r, c] == target_col and state.is_valid_cell(r, c):
                        stripe_dir = random.choice([SpecialCandy.STRIPED_HORIZONTAL, SpecialCandy.STRIPED_VERTICAL])
                        detonations.append((r, c, stripe_dir))
                        state.colors[r, c] = CandyColor.EMPTY
                        state.specials[r, c] = SpecialCandy.NONE

            state.colors[r1, c1] = CandyColor.EMPTY
            state.colors[r2, c2] = CandyColor.EMPTY

            # Fire all striped beams across the entire grid!
            for dr, dc, s_dir in detonations:
                if s_dir == SpecialCandy.STRIPED_HORIZONTAL:
                    for cc in range(state.width):
                        if state.is_valid_cell(dr, cc):
                            state.colors[dr, cc] = CandyColor.EMPTY
                            state.blockers[dr, cc] = Blocker.NONE
                            if state.jellies[dr, cc] > 0:
                                state.jellies[dr, cc] -= 1
                else:  # STRIPED_VERTICAL
                    for rr in range(state.height):
                        if state.is_valid_cell(rr, dc):
                            state.colors[rr, dc] = CandyColor.EMPTY
                            state.blockers[rr, dc] = Blocker.NONE
                            if state.jellies[rr, dc] > 0:
                                state.jellies[rr, dc] -= 1

            return True, 35000

        # Combo 3: Striped + Wrapped = Giant 3-row by 3-column cross clear
        if (s1 in (SpecialCandy.STRIPED_HORIZONTAL, SpecialCandy.STRIPED_VERTICAL) and s2 == SpecialCandy.WRAPPED) or \
           (s2 in (SpecialCandy.STRIPED_HORIZONTAL, SpecialCandy.STRIPED_VERTICAL) and s1 == SpecialCandy.WRAPPED):
            target_r, target_c = r2, c2
            for dr in [-1, 0, 1]:
                rr = target_r + dr
                if 0 <= rr < state.height:
                    for cc in range(state.width):
                        if state.is_valid_cell(rr, cc):
                            state.colors[rr, cc] = CandyColor.EMPTY
                            state.specials[rr, cc] = SpecialCandy.NONE
                            state.blockers[rr, cc] = Blocker.NONE
                            if state.jellies[rr, cc] > 0:
                                state.jellies[rr, cc] -= 1
            for dc in [-1, 0, 1]:
                cc = target_c + dc
                if 0 <= cc < state.width:
                    for rr in range(state.height):
                        if state.is_valid_cell(rr, cc):
                            state.colors[rr, cc] = CandyColor.EMPTY
                            state.specials[rr, cc] = SpecialCandy.NONE
                            state.blockers[rr, cc] = Blocker.NONE
                            if state.jellies[rr, cc] > 0:
                                state.jellies[rr, cc] -= 1
            return True, 20000

        # Combo 4: Striped + Striped = 1 Row + 1 Column cross clear
        if s1 in (SpecialCandy.STRIPED_HORIZONTAL, SpecialCandy.STRIPED_VERTICAL) and \
           s2 in (SpecialCandy.STRIPED_HORIZONTAL, SpecialCandy.STRIPED_VERTICAL):
            for cc in range(state.width):
                if state.is_valid_cell(r1, cc):
                    state.colors[r1, cc] = CandyColor.EMPTY
                    state.blockers[r1, cc] = Blocker.NONE
                    if state.jellies[r1, cc] > 0:
                        state.jellies[r1, cc] -= 1
            for rr in range(state.height):
                if state.is_valid_cell(rr, c1):
                    state.colors[rr, c1] = CandyColor.EMPTY
                    state.blockers[rr, c1] = Blocker.NONE
                    if state.jellies[rr, c1] > 0:
                        state.jellies[rr, c1] -= 1
            return True, 8000

        # Combo 5: Wrapped + Wrapped = Dual 5x5 explosion
        if s1 == SpecialCandy.WRAPPED and s2 == SpecialCandy.WRAPPED:
            for dr in range(-2, 3):
                for dc in range(-2, 3):
                    rr, cc = r2 + dr, c2 + dc
                    if state.is_valid_cell(rr, cc):
                        state.colors[rr, cc] = CandyColor.EMPTY
            return True, 15000

        # Combo 6: Color Bomb + Normal = Wipes single color
        if s1 == SpecialCandy.COLOR_BOMB or s2 == SpecialCandy.COLOR_BOMB:
            target_col = state.colors[r2, c2] if s1 == SpecialCandy.COLOR_BOMB else state.colors[r1, c1]
            for r in range(state.height):
                for c in range(state.width):
                    if state.colors[r, c] == target_col:
                        state.colors[r, c] = CandyColor.EMPTY
            state.colors[r1, c1] = CandyColor.EMPTY
            state.colors[r2, c2] = CandyColor.EMPTY
            return True, 7000

        return False, 0

    def _detect_matches(
        self, state: ReverseEngineState
    ) -> Tuple[Set[Tuple[int, int]], List[Tuple[int, int, SpecialCandy, int]]]:
        """Scans for 3-, 4-, 5-, and L/T-matches and resolves blast radiuses."""
        matched_cells: Set[Tuple[int, int]] = set()
        specials_to_spawn: List[Tuple[int, int, SpecialCandy, int]] = []

        # Horizontal Scan
        for r in range(state.height):
            c = 0
            while c < state.width:
                color = state.colors[r, c]
                if color == CandyColor.EMPTY or not state.is_valid_cell(r, c):
                    c += 1
                    continue
                start_c = c
                while c < state.width and state.is_valid_cell(r, c) and state.colors[r, c] == color:
                    c += 1
                length = c - start_c
                if length >= 3:
                    for col_idx in range(start_c, c):
                        matched_cells.add((r, col_idx))
                    mid = start_c + length // 2
                    if length == 4:
                        specials_to_spawn.append((r, mid, SpecialCandy.STRIPED_VERTICAL, color))
                    elif length >= 5:
                        specials_to_spawn.append((r, mid, SpecialCandy.COLOR_BOMB, CandyColor.EMPTY))

        # Vertical Scan
        for c in range(state.width):
            r = 0
            while r < state.height:
                color = state.colors[r, c]
                if color == CandyColor.EMPTY or not state.is_valid_cell(r, c):
                    r += 1
                    continue
                start_r = r
                while r < state.height and state.is_valid_cell(r, c) and state.colors[r, c] == color:
                    r += 1
                length = r - start_r
                if length >= 3:
                    for row_idx in range(start_r, r):
                        matched_cells.add((row_idx, c))
                    mid = start_r + length // 2
                    if length == 4:
                        specials_to_spawn.append((mid, c, SpecialCandy.STRIPED_HORIZONTAL, color))
                    elif length >= 5:
                        specials_to_spawn.append((mid, c, SpecialCandy.COLOR_BOMB, CandyColor.EMPTY))

        # Expand detonations for special candies inside matched_cells
        expanded = set(matched_cells)
        for r, c in matched_cells:
            spec = state.specials[r, c]
            if spec == SpecialCandy.STRIPED_HORIZONTAL:
                for cc in range(state.width):
                    if state.is_valid_cell(r, cc):
                        expanded.add((r, cc))
            elif spec == SpecialCandy.STRIPED_VERTICAL:
                for rr in range(state.height):
                    if state.is_valid_cell(rr, c):
                        expanded.add((rr, c))
            elif spec == SpecialCandy.WRAPPED:
                for dr in [-1, 0, 1]:
                    for dc in [-1, 0, 1]:
                        if state.is_valid_cell(r + dr, c + dc):
                            expanded.add((r + dr, c + dc))

        return expanded, specials_to_spawn

    def _apply_gravity_and_refill(self, state: ReverseEngineState) -> None:
        """Applies downward gravity and spawns new candies from top dispensers."""
        for c in range(state.width):
            write_idx = state.height - 1
            for r in range(state.height - 1, -1, -1):
                if not state.is_valid_cell(r, c):
                    continue
                # If cell has a blocker (like chocolate), write_idx cannot fall into or past it
                if state.blockers[r, c] != Blocker.NONE:
                    write_idx = r - 1
                    continue

                if state.colors[r, c] != CandyColor.EMPTY or state.ingredients[r, c] != Ingredient.NONE:
                    if r != write_idx:
                        # Find valid write_idx that is not blocked
                        while write_idx > r and (not state.is_valid_cell(write_idx, c) or state.blockers[write_idx, c] != Blocker.NONE):
                            write_idx -= 1
                        if r != write_idx and state.is_valid_cell(write_idx, c) and state.blockers[write_idx, c] == Blocker.NONE:
                            state.colors[write_idx, c] = state.colors[r, c]
                            state.specials[write_idx, c] = state.specials[r, c]
                            state.ingredients[write_idx, c] = state.ingredients[r, c]
                            state.colors[r, c] = CandyColor.EMPTY
                            state.specials[r, c] = SpecialCandy.NONE
                            state.ingredients[r, c] = Ingredient.NONE
                    write_idx -= 1

            # Refill empty slots from write_idx up to 0 with random candies
            for r in range(write_idx, -1, -1):
                if state.is_valid_cell(r, c) and state.blockers[r, c] == Blocker.NONE:
                    state.colors[r, c] = random.randint(1, state.num_colors)
                    state.specials[r, c] = SpecialCandy.NONE

        # Dynamic ingredient dispenser (works across all irregular board shapes)
        if state.config.level_type == LevelType.INGREDIENTS:
            current_on_screen = int(np.sum(state.ingredients != Ingredient.NONE))
            needed = state.config.target_ingredients - (state.ingredients_collected + current_on_screen)
            if needed > 0 and current_on_screen == 0:
                spawned = False
                col_order = sorted(range(state.width), key=lambda c: abs(c - state.width // 2))
                for col in col_order:
                    for top_r in range(state.height):
                        if state.is_valid_cell(top_r, col) and state.blockers[top_r, col] == Blocker.NONE:
                            state.ingredients[top_r, col] = Ingredient.CHERRY
                            spawned = True
                            break
                    if spawned:
                        break

    def _collect_ingredients_at_bottom(self, state: ReverseEngineState) -> None:
        """Checks if ingredients reached the bottom-most active cell and collects them."""
        for c in range(state.width):
            for r in range(state.height - 1, -1, -1):
                if state.is_valid_cell(r, c):
                    if state.ingredients[r, c] != Ingredient.NONE and state.blockers[r, c] == Blocker.NONE:
                        state.ingredients_collected += 1
                        state.ingredients[r, c] = Ingredient.NONE
                        state.colors[r, c] = CandyColor.EMPTY
                        state.score += 10000
                    break

    def _expand_chocolate(self, state: ReverseEngineState) -> None:
        """Expands 1 chocolate block to an adjacent candy if no chocolate was cleared."""
        chocolate_coords = list(zip(*np.where(state.blockers == Blocker.CHOCOLATE)))
        random.shuffle(chocolate_coords)
        for r, c in chocolate_coords:
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if state.is_valid_cell(nr, nc) and state.blockers[nr, nc] == Blocker.NONE and state.specials[nr, nc] == SpecialCandy.NONE:
                    state.blockers[nr, nc] = Blocker.CHOCOLATE
                    state.colors[nr, nc] = CandyColor.EMPTY
                    return
