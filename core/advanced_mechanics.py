"""
Advanced Reverse-Engineered Game Mechanics for Candy Crush Saga.
Implements exact King production mechanics:
1. Portals / Teleporters (`portals`)
2. Cannons / Ammo Cannons (`cannons`, `ammoCannons`)
3. Evil Spawners / Magic Mixers (`evilSpawnerElements`, `evilSpawnerAmount`, `evilSpawnerInterval`)
4. Complex Blockers: Cake Bombs, Toffee Swirls, Frogs, Popcorn
"""

from enum import IntEnum
from typing import List, Tuple, Dict, Optional, Set, Any
from dataclasses import dataclass, field
import numpy as np
import random

from core.reverse_engine import (
    CandyColor, SpecialCandy, Blocker, Ingredient, Direction, OFFSETS
)


class AdvancedBlockerType(IntEnum):
    NONE = 0
    CHOCOLATE = 1
    LICORICE_SWIRL = 2
    FROSTING = 3
    MAGIC_MIXER = 4
    CAKE_BOMB = 5
    TOFFEE_SWIRL = 6
    POPCORN = 7
    CANDY_FROG = 8
    SOUR_SKULL = 9


@dataclass
class PortalLink:
    """Represents a directional wormhole between two grid coordinates."""
    src_row: int
    src_col: int
    dst_row: int
    dst_col: int
    portal_id: int = 14


class PortalSystem:
    """
    Manages teleportation routing between disjoint board segments.
    In King's format: [ [[src_x, src_y, id], [dst_x, dst_y, id]], ... ]
    x is column (0 to W-1), y is row (0 to H-1).
    """

    def __init__(self, raw_portals: Optional[List[Any]] = None):
        self.links: Dict[Tuple[int, int], Tuple[int, int]] = {}
        self.reverse_links: Dict[Tuple[int, int], Tuple[int, int]] = {}
        if raw_portals:
            self.load_from_config(raw_portals)

    def load_from_config(self, raw_portals: List[Any]) -> None:
        self.links.clear()
        self.reverse_links.clear()
        for pair in raw_portals:
            if not isinstance(pair, list) or len(pair) < 2:
                continue
            src, dst = pair[0], pair[1]
            if isinstance(src, list) and isinstance(dst, list) and len(src) >= 2 and len(dst) >= 2:
                src_c, src_r = int(src[0]), int(src[1])
                dst_c, dst_r = int(dst[0]), int(dst[1])
                self.links[(src_r, src_c)] = (dst_r, dst_c)
                self.reverse_links[(dst_r, dst_c)] = (src_r, src_c)

    def get_destination(self, r: int, c: int) -> Optional[Tuple[int, int]]:
        return self.links.get((r, c))

    def get_source(self, r: int, c: int) -> Optional[Tuple[int, int]]:
        return self.reverse_links.get((r, c))

    def has_portal(self, r: int, c: int) -> bool:
        return (r, c) in self.links or (r, c) in self.reverse_links


class CannonDispenserSystem:
    """
    Manages item dispensing cannons and replenishment ammo reservoirs.
    In King's format:
      - cannons: list of item IDs per dispenser column
      - ammoCannons: remaining ammunition inventory
    """

    def __init__(
        self,
        raw_cannons: Optional[List[Any]] = None,
        raw_ammo: Optional[List[Any]] = None,
        width: int = 9
    ):
        self.width = width
        self.dispensers: Dict[int, List[int]] = {}
        self.ammo_counts: Dict[int, int] = {}
        self._load(raw_cannons, raw_ammo)

    def _load(self, raw_cannons: Optional[List[Any]], raw_ammo: Optional[List[Any]]) -> None:
        if raw_cannons:
            for col_idx, items in enumerate(raw_cannons):
                if isinstance(items, list) and len(items) > 0:
                    self.dispensers[col_idx] = [int(x) for x in items]
                elif isinstance(items, int):
                    self.dispensers[col_idx] = [int(items)]

        if raw_ammo:
            for col_idx, ammo in enumerate(raw_ammo):
                if isinstance(ammo, list) and len(ammo) > 0:
                    self.ammo_counts[col_idx] = len(ammo)
                elif isinstance(ammo, int):
                    self.ammo_counts[col_idx] = int(ammo)
                else:
                    self.ammo_counts[col_idx] = 999  # Unlimited default

    def has_dispenser(self, col: int) -> bool:
        return col in self.dispensers and len(self.dispensers[col]) > 0

    def pop_dispensed_item(self, col: int) -> Optional[int]:
        """Returns the item to spawn (e.g. 26=Licorice Swirl, 71=Striped) or None."""
        if not self.has_dispenser(col):
            return None
        
        # Check ammo
        ammo = self.ammo_counts.get(col, 999)
        if ammo <= 0:
            return None
        
        # 40% probability to dispense special cannon item instead of regular candy
        if random.random() < 0.40:
            items = self.dispensers[col]
            chosen = random.choice(items)
            self.ammo_counts[col] = ammo - 1
            return chosen
        return None


@dataclass
class MagicMixerState:
    """
    Reverse-Engineered Magic Mixer (Evil Spawner).
    - HP: default 4
    - Interval: default 3 moves without damage to trigger
    - Counter: counts down to 0
    - Amount: number of blockers spewed
    - Elements: list of blocker IDs (e.g. [2] Chocolate, [8] Licorice, [9] Frosting, [14] Bombs)
    """
    row: int
    col: int
    hp: int = 4
    interval: int = 3
    countdown: int = 3
    amount: int = 1
    elements: List[int] = field(default_factory=lambda: [2])
    was_damaged_this_turn: bool = False

    def take_hit(self) -> bool:
        """Deals 1 damage. Resets countdown timer. Returns True if destroyed."""
        self.hp = max(0, self.hp - 1)
        self.was_damaged_this_turn = True
        self.countdown = self.interval  # Stun resets countdown
        return self.hp == 0

    def step_turn(self) -> bool:
        """Decrements countdown if not damaged. Returns True if ready to spew."""
        if self.was_damaged_this_turn:
            self.was_damaged_this_turn = False
            return False
        self.countdown -= 1
        if self.countdown <= 0:
            self.countdown = self.interval
            return True
        return False


@dataclass
class CakeBombState:
    """
    Reverse-Engineered Cake Bomb.
    - Occupies a 2x2 grid (top-left r, c)
    - 8 slices (HP = 8)
    - When HP reaches 0: triggers board-wide hammer sweep wipe (1 layer to all cells)
    """
    top_left_r: int
    top_left_c: int
    slices_remaining: int = 8

    def is_cell_part_of_cake(self, r: int, c: int) -> bool:
        return (self.top_left_r <= r <= self.top_left_r + 1 and
                self.top_left_c <= c <= self.top_left_c + 1)

    def take_hit(self, damage: int = 1) -> bool:
        """Reduces slice count. Returns True if completely eliminated."""
        self.slices_remaining = max(0, self.slices_remaining - damage)
        return self.slices_remaining == 0


@dataclass
class ToffeeSwirlState:
    """
    Reverse-Engineered Toffee Swirl.
    - Multi-layer blocker (HP: 1 to 4 layers)
    - Movable / subject to gravity (falls down columns)
    - Takes 1 damage per adjacent match
    """
    row: int
    col: int
    layers: int = 1

    def take_hit(self) -> bool:
        self.layers = max(0, self.layers - 1)
        return self.layers == 0


@dataclass
class PopcornState:
    """
    Reverse-Engineered Popcorn / Bubblegum Pop.
    - HP = 3 (Stage 1: unpopped kernel, Stage 2: heated, Stage 3: bursting)
    - When destroyed: transforms directly into a COLOR BOMB on this tile!
    """
    row: int
    col: int
    hp: int = 3

    def take_hit(self) -> bool:
        self.hp = max(0, self.hp - 1)
        return self.hp == 0


@dataclass
class CandyFrogState:
    """
    Reverse-Engineered Candy Frog.
    - Matches with same color
    - Stomach fullness tracks consumed candies towards capacity (default 6 to 12)
    - When full (Frogtastic), player can target ANY 3x3 square to clear it!
    """
    row: int
    col: int
    color: int
    stomach_capacity: int = 6
    current_fullness: int = 0
    is_frogtastic: bool = False

    def feed(self, count: int = 1) -> bool:
        """Increments fullness. Returns True if entering Frogtastic state."""
        self.current_fullness += count
        if self.current_fullness >= self.stomach_capacity:
            self.is_frogtastic = True
        return self.is_frogtastic

    def jump_and_crush(self, target_r: int, target_c: int, new_color: int) -> Tuple[int, int]:
        """Executes Frogtastic 3x3 landing crush, changes color, and resets fullness."""
        self.row = target_r
        self.col = target_c
        self.color = new_color
        self.current_fullness = 0
        self.is_frogtastic = False
        return (target_r, target_c)


class AdvancedMechanicsManager:
    """
    Unified coordinator managing all advanced mechanics during state transitions:
    - Portals
    - Cannons
    - Magic Mixers
    - Cake Bombs
    - Toffee Swirls
    - Popcorn
    - Frogs
    """

    def __init__(
        self,
        height: int = 9,
        width: int = 9,
        raw_portals: Optional[List[Any]] = None,
        raw_cannons: Optional[List[Any]] = None,
        raw_ammo: Optional[List[Any]] = None
    ):
        self.height = height
        self.width = width
        self.portal_system = PortalSystem(raw_portals)
        self.cannon_system = CannonDispenserSystem(raw_cannons, raw_ammo, width=width)
        
        self.magic_mixers: List[MagicMixerState] = []
        self.cake_bombs: List[CakeBombState] = []
        self.toffee_swirls: Dict[Tuple[int, int], ToffeeSwirlState] = {}
        self.popcorns: Dict[Tuple[int, int], PopcornState] = {}
        self.frog: Optional[CandyFrogState] = None

    def add_magic_mixer(self, r: int, c: int, hp: int = 4, interval: int = 3, amount: int = 1, elements: Optional[List[int]] = None) -> None:
        self.magic_mixers.append(MagicMixerState(
            row=r, col=c, hp=hp, interval=interval, countdown=interval,
            amount=amount, elements=elements or [2]
        ))

    def add_cake_bomb(self, top_left_r: int, top_left_c: int, slices: int = 8) -> None:
        self.cake_bombs.append(CakeBombState(top_left_r, top_left_c, slices_remaining=slices))

    def add_toffee_swirl(self, r: int, c: int, layers: int = 1) -> None:
        self.toffee_swirls[(r, c)] = ToffeeSwirlState(row=r, col=c, layers=layers)

    def add_popcorn(self, r: int, c: int, hp: int = 3) -> None:
        self.popcorns[(r, c)] = PopcornState(row=r, col=c, hp=hp)

    def set_frog(self, r: int, c: int, color: int, stomach_capacity: int = 6) -> None:
        self.frog = CandyFrogState(row=r, col=c, color=color, stomach_capacity=stomach_capacity)

    def damage_adjacent_mechanics(
        self, cleared_coords: Set[Tuple[int, int]]
    ) -> Dict[str, Any]:
        """
        Calculates adjacency damage from match clears to:
        - Magic Mixers
        - Cake Bombs
        - Toffee Swirls
        - Popcorn
        Returns dictionary of triggered board events (e.g. board_wipe, color_bombs_spawned).
        """
        events = {
            "board_wipe": False,
            "popcorn_color_bombs": [],
            "mixers_destroyed": []
        }

        # Find adjacent coordinates
        adjacent_coords = set()
        for r, c in cleared_coords:
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                adjacent_coords.add((r + dr, c + dc))

        # 1. Magic Mixers
        alive_mixers = []
        for mixer in self.magic_mixers:
            if (mixer.row, mixer.col) in adjacent_coords or (mixer.row, mixer.col) in cleared_coords:
                destroyed = mixer.take_hit()
                if destroyed:
                    events["mixers_destroyed"].append((mixer.row, mixer.col))
                else:
                    alive_mixers.append(mixer)
            else:
                alive_mixers.append(mixer)
        self.magic_mixers = alive_mixers

        # 2. Cake Bombs
        alive_cakes = []
        for cake in self.cake_bombs:
            hit = False
            for dr in [0, 1]:
                for dc in [0, 1]:
                    cr, cc = cake.top_left_r + dr, cake.top_left_c + dc
                    if (cr, cc) in adjacent_coords or (cr, cc) in cleared_coords:
                        hit = True
                        break
                if hit:
                    break
            if hit:
                destroyed = cake.take_hit(damage=1)
                if destroyed:
                    events["board_wipe"] = True  # Triggers full board hammer sweep!
                else:
                    alive_cakes.append(cake)
            else:
                alive_cakes.append(cake)
        self.cake_bombs = alive_cakes

        # 3. Popcorn
        remaining_popcorns = {}
        for (pr, pc), popcorn in self.popcorns.items():
            if (pr, pc) in adjacent_coords or (pr, pc) in cleared_coords:
                popped = popcorn.take_hit()
                if popped:
                    events["popcorn_color_bombs"].append((pr, pc))
                else:
                    remaining_popcorns[(pr, pc)] = popcorn
            else:
                remaining_popcorns[(pr, pc)] = popcorn
        self.popcorns = remaining_popcorns

        # 4. Toffee Swirls
        remaining_toffee = {}
        for (tr, tc), toffee in self.toffee_swirls.items():
            if (tr, tc) in adjacent_coords or (tr, tc) in cleared_coords:
                cleared = toffee.take_hit()
                if not cleared:
                    remaining_toffee[(tr, tc)] = toffee
            else:
                remaining_toffee[(tr, tc)] = toffee
        self.toffee_swirls = remaining_toffee

        return events

    def update_evil_spawners(self) -> List[Tuple[int, int, int]]:
        """
        Steps turn countdown for Magic Mixers.
        Returns list of (row, col, blocker_element_id) to spawn on the grid.
        """
        spawns = []
        for mixer in self.magic_mixers:
            ready = mixer.step_turn()
            if ready:
                # Find adjacent valid cells to spew blockers
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nr, nc = mixer.row + dr, mixer.col + dc
                    if 0 <= nr < self.height and 0 <= nc < self.width:
                        elem = random.choice(mixer.elements)
                        spawns.append((nr, nc, elem))
                        if len(spawns) >= mixer.amount:
                            break
        return spawns
