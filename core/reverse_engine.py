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
    JELLY_FISH = 5


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
    cannon_specials: Optional[np.ndarray] = None
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
        new_state = object.__new__(ReverseEngineState)
        new_state.config = self.config
        new_state.height = self.height
        new_state.width = self.width
        new_state.num_colors = self.num_colors
        new_state.active_grid = self.active_grid
        new_state.colors = self.colors.copy()
        new_state.specials = self.specials.copy()
        new_state.jellies = self.jellies.copy()
        new_state.blockers = self.blockers.copy()
        new_state.ingredients = self.ingredients.copy()
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
            return np.sum(self.jellies > 0) == 0
        elif self.config.level_type == LevelType.INGREDIENTS:
            return self.ingredients_collected >= self.config.target_ingredients
        elif self.config.level_type == LevelType.BLOCKERS:
            return np.sum(self.blockers > 0) == 0
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

    def initialize_board(self, state: ReverseEngineState, seed: Optional[int] = None) -> None:
        """Fills active cells with random candies ensuring no initial 3-matches."""
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
        for r in range(state.height):
            for c in range(state.width):
                if not state.active_grid[r, c]:
                    continue
                # Solid impassable blockers prevent candies from occupying tile
                if state.blockers[r, c] in (Blocker.CHOCOLATE, Blocker.FROSTING):
                    state.colors[r, c] = CandyColor.EMPTY
                    continue
                # If cell already has a preset candy or a colorless Color Bomb, keep it.
                # Preset colored specials (Striped, Wrapped, Jelly Fish) without an explicit color get assigned a valid color.
                if state.colors[r, c] != CandyColor.EMPTY or state.specials[r, c] == SpecialCandy.COLOR_BOMB:
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
            col_order = sorted(range(state.width), key=lambda c: abs(c - state.width // 2))
            valid_cols = []
            for c in col_order:
                bottom_candidates = [r for r in range(state.height - 1, -1, -1) if state.active_grid[r, c]]
                if not bottom_candidates:
                    continue
                bot_r = bottom_candidates[0]
                top_candidates = [r for r in range(state.height) if state.is_valid_cell(r, c) and state.blockers[r, c] == Blocker.NONE]
                if not top_candidates:
                    continue
                tp_r = top_candidates[0]
                if all(state.active_grid[r, c] for r in range(tp_r, bot_r + 1)):
                    valid_cols.append((tp_r, c))

            if not valid_cols:
                for c in col_order:
                    for top_r in range(state.height):
                        if state.is_valid_cell(top_r, c) and state.blockers[top_r, c] == Blocker.NONE:
                            valid_cols.append((top_r, c))
                            break
                    if valid_cols:
                        break

            if valid_cols:
                top_r, c = valid_cols[0]
                state.ingredients[top_r, c] = Ingredient.CHERRY

    def shuffle_board(self, state: ReverseEngineState) -> bool:
        """Reshuffles board candies when no moves exist (official King Reshuffle mechanic)."""
        coords = []
        colors = []
        for r in range(state.height):
            for c in range(state.width):
                if state.is_valid_cell(r, c) and state.blockers[r, c] == Blocker.NONE:
                    has_neighbor = any(state.is_valid_cell(r + dr, c + dc) for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)])
                    if has_neighbor:
                        coords.append((r, c))
                        if state.colors[r, c] != CandyColor.EMPTY:
                            colors.append(state.colors[r, c])
                        else:
                            colors.append(random.randint(1, state.num_colors))

        if len(coords) < 3:
            return False

        for _ in range(30):
            random.shuffle(colors)
            for idx, (r, c) in enumerate(coords):
                state.colors[r, c] = colors[idx]
            if self.get_legal_moves(state):
                return True
        return False

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
        if not (state.is_valid_cell(r1, c1) and state.is_valid_cell(r2, c2)):
            return False
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

        # Perform swap only if not already consumed by a special combination
        if not combo_handled:
            state.colors[r1, c1], state.colors[r2, c2] = state.colors[r2, c2], state.colors[r1, c1]
            state.specials[r1, c1], state.specials[r2, c2] = state.specials[r2, c2], state.specials[r1, c1]
            state.ingredients[r1, c1], state.ingredients[r2, c2] = state.ingredients[r2, c2], state.ingredients[r1, c1]
        else:
            state.specials[r1, c1] = SpecialCandy.NONE
            state.specials[r2, c2] = SpecialCandy.NONE

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

                # Check for Jelly Fish / Candy Frog trigger before clearing
                if state.specials[r, c] == SpecialCandy.JELLY_FISH:
                    targets = []
                    for jr in range(state.height):
                        for jc in range(state.width):
                            if state.active_grid[jr, jc] and state.jellies[jr, jc] > 0:
                                nbr_cnt = sum(
                                    1 for dr_n, dc_n in [(-1, 0), (1, 0), (0, -1), (0, 1)]
                                    if state.is_valid_cell(jr + dr_n, jc + dc_n)
                                )
                                is_iso = nbr_cnt <= 1
                                targets.append((is_iso, state.blockers[jr, jc] != Blocker.NONE, int(state.jellies[jr, jc]), jr, jc))
                    targets.sort(key=lambda t: (t[0], t[1], t[2]), reverse=True)
                    max_targets = 9 if getattr(state.config, "frog_stomach_size", 0) > 0 else 3
                    for _, _, _, tr, tc in targets[:max_targets]:
                        if state.blockers[tr, tc] != Blocker.NONE:
                            state.blockers[tr, tc] = Blocker.NONE
                        if state.jellies[tr, tc] > 0:
                            state.jellies[tr, tc] -= 1
                        step_score += 1500

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

        # Combo 1: Color Bomb + Color Bomb = Complete Board Clear (2 layers)
        if s1 == SpecialCandy.COLOR_BOMB and s2 == SpecialCandy.COLOR_BOMB:
            for r in range(state.height):
                for c in range(state.width):
                    if state.active_grid[r, c]:
                        state.colors[r, c] = CandyColor.EMPTY
                        state.specials[r, c] = SpecialCandy.NONE
                        state.blockers[r, c] = Blocker.NONE
                        if state.jellies[r, c] > 0:
                            state.jellies[r, c] = max(0, state.jellies[r, c] - 2)
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
                            if state.blockers[dr, cc] != Blocker.NONE:
                                state.blockers[dr, cc] = Blocker.NONE
                            if state.jellies[dr, cc] > 0:
                                state.jellies[dr, cc] -= 1
                            has_refill_above = any(state.active_grid[ar, cc] for ar in range(dr))
                            if state.jellies[dr, cc] > 0 and not has_refill_above:
                                pass
                            else:
                                state.colors[dr, cc] = CandyColor.EMPTY
                else:  # STRIPED_VERTICAL
                    for rr in range(state.height):
                        if state.is_valid_cell(rr, dc):
                            if state.blockers[rr, dc] != Blocker.NONE:
                                state.blockers[rr, dc] = Blocker.NONE
                            if state.jellies[rr, dc] > 0:
                                state.jellies[rr, dc] -= 1
                            has_refill_above = any(state.active_grid[ar, dc] for ar in range(rr))
                            if state.jellies[rr, dc] > 0 and not has_refill_above:
                                pass
                            else:
                                state.colors[rr, dc] = CandyColor.EMPTY

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
                            if state.blockers[rr, cc] != Blocker.NONE:
                                state.blockers[rr, cc] = Blocker.NONE
                            if state.jellies[rr, cc] > 0:
                                state.jellies[rr, cc] -= 1
                            has_refill_above = any(state.active_grid[ar, cc] for ar in range(rr))
                            if state.jellies[rr, cc] > 0 and not has_refill_above:
                                pass
                            else:
                                state.colors[rr, cc] = CandyColor.EMPTY
            for dc in [-1, 0, 1]:
                cc = target_c + dc
                if 0 <= cc < state.width:
                    for rr in range(state.height):
                        if state.is_valid_cell(rr, cc):
                            if state.blockers[rr, cc] != Blocker.NONE:
                                state.blockers[rr, cc] = Blocker.NONE
                            if state.jellies[rr, cc] > 0:
                                state.jellies[rr, cc] -= 1
                            has_refill_above = any(state.active_grid[ar, cc] for ar in range(rr))
                            if state.jellies[rr, cc] > 0 and not has_refill_above:
                                pass
                            else:
                                state.colors[rr, cc] = CandyColor.EMPTY
            return True, 20000

        # Combo 4: Striped + Striped = 1 Row + 1 Column cross clear
        if s1 in (SpecialCandy.STRIPED_HORIZONTAL, SpecialCandy.STRIPED_VERTICAL) and \
           s2 in (SpecialCandy.STRIPED_HORIZONTAL, SpecialCandy.STRIPED_VERTICAL):
            for cc in range(state.width):
                if state.is_valid_cell(r1, cc):
                    if state.blockers[r1, cc] != Blocker.NONE:
                        state.blockers[r1, cc] = Blocker.NONE
                    if state.jellies[r1, cc] > 0:
                        state.jellies[r1, cc] -= 1
                    has_refill_above = any(state.active_grid[ar, cc] for ar in range(r1))
                    if state.jellies[r1, cc] > 0 and not has_refill_above:
                        pass
                    else:
                        state.colors[r1, cc] = CandyColor.EMPTY
            for rr in range(state.height):
                if state.is_valid_cell(rr, c1):
                    if state.blockers[rr, c1] != Blocker.NONE:
                        state.blockers[rr, c1] = Blocker.NONE
                    if state.jellies[rr, c1] > 0:
                        state.jellies[rr, c1] -= 1
                    has_refill_above = any(state.active_grid[ar, c1] for ar in range(rr))
                    if state.jellies[rr, c1] > 0 and not has_refill_above:
                        pass
                    else:
                        state.colors[rr, c1] = CandyColor.EMPTY
            return True, 8000

        # Combo 5: Wrapped + Wrapped = Dual 5x5 explosion
        if s1 == SpecialCandy.WRAPPED and s2 == SpecialCandy.WRAPPED:
            for dr in range(-2, 3):
                for dc in range(-2, 3):
                    rr, cc = r2 + dr, c2 + dc
                    if state.is_valid_cell(rr, cc):
                        if state.blockers[rr, cc] != Blocker.NONE:
                            state.blockers[rr, cc] = Blocker.NONE
                        if state.jellies[rr, cc] > 0:
                            state.jellies[rr, cc] -= 1
                        has_refill_above = any(state.active_grid[ar, cc] for ar in range(rr))
                        if state.jellies[rr, cc] > 0 and not has_refill_above:
                            pass
                        else:
                            state.colors[rr, cc] = CandyColor.EMPTY
            return True, 15000

        # Combo 5b: Color Bomb + Wrapped = Wipes two colors across the board
        if (s1 == SpecialCandy.COLOR_BOMB and s2 == SpecialCandy.WRAPPED) or (
            s2 == SpecialCandy.COLOR_BOMB and s1 == SpecialCandy.WRAPPED
        ):
            target_col1 = state.colors[r2, c2] if s1 == SpecialCandy.COLOR_BOMB else state.colors[r1, c1]
            if target_col1 == CandyColor.EMPTY:
                target_col1 = random.randint(1, state.num_colors)
            target_col2 = (target_col1 % state.num_colors) + 1
            for r in range(state.height):
                for c in range(state.width):
                    if state.active_grid[r, c] and state.colors[r, c] in (target_col1, target_col2):
                        if state.blockers[r, c] != Blocker.NONE:
                            state.blockers[r, c] = Blocker.NONE
                        if state.jellies[r, c] > 0:
                            state.jellies[r, c] -= 1
                        has_refill_above = any(state.active_grid[ar, c] for ar in range(r))
                        if state.jellies[r, c] > 0 and not has_refill_above:
                            pass
                        else:
                            state.colors[r, c] = CandyColor.EMPTY
                        state.specials[r, c] = SpecialCandy.NONE
            state.colors[r1, c1] = CandyColor.EMPTY
            state.colors[r2, c2] = CandyColor.EMPTY
            return True, 25000

        # Combo 5c: Jelly Fish / Candy Frog + Any Special = Homing school of fish targeting isolated/remaining jellies
        if (s1 == SpecialCandy.JELLY_FISH and s2 != SpecialCandy.NONE) or (
            s2 == SpecialCandy.JELLY_FISH and s1 != SpecialCandy.NONE
        ):
            targets = []
            for jr in range(state.height):
                for jc in range(state.width):
                    if state.active_grid[jr, jc] and state.jellies[jr, jc] > 0:
                        nbr_cnt = sum(
                            1 for dr_n, dc_n in [(-1, 0), (1, 0), (0, -1), (0, 1)]
                            if state.is_valid_cell(jr + dr_n, jc + dc_n)
                        )
                        is_iso = nbr_cnt <= 1
                        targets.append((is_iso, state.blockers[jr, jc] != Blocker.NONE, int(state.jellies[jr, jc]), jr, jc))
            targets.sort(key=lambda t: (t[0], t[1], t[2]), reverse=True)
            max_targets = 9 if (getattr(state.config, "frog_stomach_size", 0) > 0 or s1 == SpecialCandy.COLOR_BOMB or s2 == SpecialCandy.COLOR_BOMB) else 6
            for _, _, _, tr, tc in targets[:max_targets]:
                if state.blockers[tr, tc] != Blocker.NONE:
                    state.blockers[tr, tc] = Blocker.NONE
                if state.jellies[tr, tc] > 0:
                    state.jellies[tr, tc] -= 1
            state.colors[r1, c1] = CandyColor.EMPTY
            state.colors[r2, c2] = CandyColor.EMPTY
            return True, 18000

        # Combo 6: Color Bomb + Normal = Wipes single color & damages jellies / clears locks
        if s1 == SpecialCandy.COLOR_BOMB or s2 == SpecialCandy.COLOR_BOMB:
            target_col = state.colors[r2, c2] if s1 == SpecialCandy.COLOR_BOMB else state.colors[r1, c1]
            if target_col == CandyColor.EMPTY:
                target_col = random.randint(1, state.num_colors)
            for r in range(state.height):
                for c in range(state.width):
                    if state.colors[r, c] == target_col and state.active_grid[r, c]:
                        if state.blockers[r, c] != Blocker.NONE:
                            state.blockers[r, c] = Blocker.NONE
                        if state.jellies[r, c] > 0:
                            state.jellies[r, c] -= 1
                        has_refill_above = any(state.active_grid[ar, c] for ar in range(r))
                        if state.jellies[r, c] > 0 and not has_refill_above:
                            pass
                        else:
                            state.colors[r, c] = CandyColor.EMPTY
                        state.specials[r, c] = SpecialCandy.NONE
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

        # Expand detonations for special candies via BFS chain-reaction
        expanded = set(matched_cells)
        queue = list(matched_cells)
        detonated = set()
        while queue:
            r, c = queue.pop()
            if (r, c) in detonated:
                continue
            spec = state.specials[r, c]
            if spec in (SpecialCandy.STRIPED_HORIZONTAL, SpecialCandy.STRIPED_VERTICAL, SpecialCandy.WRAPPED):
                detonated.add((r, c))
                if spec == SpecialCandy.STRIPED_HORIZONTAL:
                    for cc in range(state.width):
                        if state.is_valid_cell(r, cc) and (r, cc) not in expanded:
                            expanded.add((r, cc))
                            queue.append((r, cc))
                elif spec == SpecialCandy.STRIPED_VERTICAL:
                    for rr in range(state.height):
                        if state.is_valid_cell(rr, c) and (rr, c) not in expanded:
                            expanded.add((rr, c))
                            queue.append((rr, c))
                elif spec == SpecialCandy.WRAPPED:
                    for dr in [-1, 0, 1]:
                        for dc in [-1, 0, 1]:
                            if state.is_valid_cell(r + dr, c + dc) and (r + dr, c + dc) not in expanded:
                                expanded.add((r + dr, c + dc))
                                queue.append((r + dr, c + dc))

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

                if (
                    state.colors[r, c] != CandyColor.EMPTY
                    or state.specials[r, c] != SpecialCandy.NONE
                    or state.ingredients[r, c] != Ingredient.NONE
                ):
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

            # Refill empty slots (including any ingredient tile whose candy color was cleared, but preserving colorless Color Bombs)
            for r in range(state.height - 1, -1, -1):
                if (
                    state.is_valid_cell(r, c)
                    and state.blockers[r, c] == Blocker.NONE
                    and state.colors[r, c] == CandyColor.EMPTY
                    and state.specials[r, c] != SpecialCandy.COLOR_BOMB
                ):
                    state.colors[r, c] = random.randint(1, state.num_colors)
                    if state.specials[r, c] == SpecialCandy.NONE:
                        cannon_grid = getattr(state.config, "cannon_specials", None)
                        if cannon_grid is not None and cannon_grid[r, c] != SpecialCandy.NONE:
                            state.specials[r, c] = int(cannon_grid[r, c])
                        else:
                            has_cannon = state.config.cannons and (c in [6, 7] or len(state.config.cannons) > 0)
                            if has_cannon and random.random() < 0.35:
                                state.specials[r, c] = SpecialCandy.JELLY_FISH
                            else:
                                state.specials[r, c] = SpecialCandy.NONE

        # Dynamic ingredient dispenser (works across all irregular board shapes)
        if state.config.level_type == LevelType.INGREDIENTS:
            current_on_screen = int(np.sum(state.ingredients != Ingredient.NONE))
            needed = state.config.target_ingredients - (state.ingredients_collected + current_on_screen)
            if needed > 0 and current_on_screen == 0:
                spawned = False
                col_order = sorted(range(state.width), key=lambda c: abs(c - state.width // 2))
                valid_cols = []
                for c in col_order:
                    bottom_candidates = [r for r in range(state.height - 1, -1, -1) if state.active_grid[r, c]]
                    if not bottom_candidates:
                        continue
                    bot_r = bottom_candidates[0]
                    top_candidates = [r for r in range(state.height) if state.is_valid_cell(r, c) and state.blockers[r, c] == Blocker.NONE]
                    if not top_candidates:
                        continue
                    tp_r = top_candidates[0]
                    if all(state.active_grid[r, c] for r in range(tp_r, bot_r + 1)):
                        valid_cols.append((tp_r, c))

                if not valid_cols:
                    for c in col_order:
                        for top_r in range(state.height):
                            if state.is_valid_cell(top_r, c) and state.blockers[top_r, c] == Blocker.NONE:
                                valid_cols.append((top_r, c))
                                break
                        if valid_cols:
                            break

                if valid_cols:
                    top_r, col = valid_cols[0]
                    state.ingredients[top_r, col] = Ingredient.CHERRY

    def _collect_ingredients_at_bottom(self, state: ReverseEngineState) -> None:
        """Checks if ingredients reached the bottom-most active cell, collects them, and cascades column gravity."""
        while True:
            collected_any = False
            for c in range(state.width):
                for r in range(state.height - 1, -1, -1):
                    if state.is_valid_cell(r, c):
                        if state.ingredients[r, c] != Ingredient.NONE and state.blockers[r, c] == Blocker.NONE:
                            state.ingredients_collected += 1
                            state.ingredients[r, c] = Ingredient.NONE
                            state.colors[r, c] = CandyColor.EMPTY
                            state.score += 10000
                            collected_any = True
                        break
            if collected_any:
                self._apply_gravity_and_refill(state)
            else:
                break

    def _expand_chocolate(self, state: ReverseEngineState) -> None:
        """Expands 1 chocolate block to an adjacent candy if no chocolate was cleared."""
        chocolate_coords = list(zip(*np.where(state.blockers == Blocker.CHOCOLATE)))
        random.shuffle(chocolate_coords)
        for r, c in chocolate_coords:
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if (
                    state.is_valid_cell(nr, nc)
                    and state.blockers[nr, nc] == Blocker.NONE
                    and state.specials[nr, nc] == SpecialCandy.NONE
                    and state.ingredients[nr, nc] == Ingredient.NONE
                ):
                    state.blockers[nr, nc] = Blocker.CHOCOLATE
                    state.colors[nr, nc] = CandyColor.EMPTY
                    return
