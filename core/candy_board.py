"""
Candy Crush Board & Tile State Definitions.
Encapsulates colors, special candies, blockers, and tensor representations.
"""

from enum import IntEnum
from typing import Tuple, List, Optional, Dict, Any
import numpy as np


class CandyColor(IntEnum):
    EMPTY = 0
    RED = 1
    BLUE = 2
    GREEN = 3
    YELLOW = 4
    ORANGE = 5
    PURPLE = 6


class SpecialType(IntEnum):
    NORMAL = 0
    STRIPED_HORIZONTAL = 1
    STRIPED_VERTICAL = 2
    WRAPPED = 3
    COLOR_BOMB = 4


class BlockerType(IntEnum):
    NONE = 0
    JELLY_L1 = 1
    JELLY_L2 = 2
    CHOCOLATE = 3
    LICORICE = 4
    FROSTING = 5


class Direction(IntEnum):
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3


DIRECTION_OFFSETS = {
    Direction.UP: (-1, 0),
    Direction.DOWN: (1, 0),
    Direction.LEFT: (0, -1),
    Direction.RIGHT: (0, 1),
}


class BoardState:
    """
    Represents the full multi-layer state of a Candy Crush grid.
    
    Layers:
      - colors: (H, W) array of CandyColor
      - specials: (H, W) array of SpecialType
      - blockers: (H, W) array of BlockerType
      - jellies: (H, W) array of jelly layers remaining (0, 1, 2)
    """

    def __init__(self, height: int = 9, width: int = 9, num_colors: int = 6):
        self.height = height
        self.width = width
        self.num_colors = num_colors
        
        self.colors = np.zeros((height, width), dtype=np.int32)
        self.specials = np.zeros((height, width), dtype=np.int32)
        self.blockers = np.zeros((height, width), dtype=np.int32)
        self.jellies = np.zeros((height, width), dtype=np.int32)
        
        self.moves_left: int = 30
        self.score: int = 0
        self.target_score: int = 10000

    def copy(self) -> "BoardState":
        new_state = BoardState(self.height, self.width, self.num_colors)
        new_state.colors = np.copy(self.colors)
        new_state.specials = np.copy(self.specials)
        new_state.blockers = np.copy(self.blockers)
        new_state.jellies = np.copy(self.jellies)
        new_state.moves_left = self.moves_left
        new_state.score = self.score
        new_state.target_score = self.target_score
        return new_state

    def is_valid_coord(self, r: int, c: int) -> bool:
        return 0 <= r < self.height and 0 <= c < self.width

    def to_tensor(self) -> np.ndarray:
        """
        Converts the board state to a multi-channel feature tensor (C, H, W)
        suitable for CNN / ResNet policy & value networks.
        
        Channels:
          - 0..6: One-hot candy colors (0: empty, 1..6: colors)
          - 7..11: One-hot special types (normal, striped_h, striped_v, wrapped, bomb)
          - 12..14: Jellies (0, 1, 2)
          - 15..17: Blockers (chocolate, licorice, frosting)
          - 18: Normalized moves left
          Total: 19 feature channels
        """
        channels = 19
        tensor = np.zeros((channels, self.height, self.width), dtype=np.float32)
        
        # Color channels 0..6
        for color_val in range(7):
            tensor[color_val] = (self.colors == color_val).astype(np.float32)
            
        # Specials channels 7..11
        for s_idx in range(5):
            tensor[7 + s_idx] = (self.specials == s_idx).astype(np.float32)
            
        # Jellies channels 12..14
        for j_val in range(3):
            tensor[12 + j_val] = (self.jellies == j_val).astype(np.float32)
            
        # Blockers channels 15..17
        tensor[15] = (self.blockers == BlockerType.CHOCOLATE).astype(np.float32)
        tensor[16] = (self.blockers == BlockerType.LICORICE).astype(np.float32)
        tensor[17] = (self.blockers == BlockerType.FROSTING).astype(np.float32)
        
        # Normalized moves left channel 18
        tensor[18].fill(min(1.0, max(0.0, self.moves_left / 50.0)))
        
        return tensor

    def __repr__(self) -> str:
        res = [f"BoardState({self.height}x{self.width}, moves={self.moves_left}, score={self.score}):"]
        for r in range(self.height):
            row_str = []
            for c in range(self.width):
                color = self.colors[r, c]
                special = self.specials[r, c]
                if color == 0:
                    row_str.append(" . ")
                elif special == SpecialType.COLOR_BOMB:
                    row_str.append("[*]")
                elif special == SpecialType.STRIPED_HORIZONTAL:
                    row_str.append(f"-{color}-")
                elif special == SpecialType.STRIPED_VERTICAL:
                    row_str.append(f"|{color}|")
                elif special == SpecialType.WRAPPED:
                    row_str.append(f"#{color}#")
                else:
                    row_str.append(f" {color} ")
            res.append(" ".join(row_str))
        return "\n".join(res)
