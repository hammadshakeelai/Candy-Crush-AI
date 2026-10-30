"""
State encoding v2 for AlphaCandy-Zero.

Fixes three real defects of the legacy 20-channel encoder (models/spatial_transformer_net.py):
  1. Channel 19 blended moves_left and objectives into ONE number, so "2 moves / 2 jellies" and
     "20 moves / 20 jellies" were indistinguishable. v2 exposes them as separate globals + pace.
  2. Level type, colour count, score target and ingredient progress were invisible.
  3. No geometry hints for spawners / ingredient exits / portals.

Spatial planes (24, 9x9):
   0-5  colour one-hot (RED..PURPLE)          [colour planes: the network is S6-invariant over these]
   6-9  striped-H, striped-V, wrapped, colour bomb
   10   jelly fish
   11   chocolate   12 licorice   13 frosting   14 any blocker
   15   jelly x1    16 jelly x2+
   17   ingredient
   18   active cell
   19   top spawner cell of each column (first active cell from the top)
   20   ingredient exit cell (bottom-most active cell of each column)
   21   portal entrance  22 portal exit
   23   cell participates in at least one legal swap

Global features (16):
   moves_left/50, moves_left/moves_limit, remaining/initial_remaining, log1p(remaining)/6,
   pace = remaining/(moves_left+1) (clipped /10), level-type one-hot (4), num_colors/6,
   ingredient progress, score/target (clipped at 2), specials on board /10,
   blocker fraction, jelly-cell fraction, moves_limit/60
"""

from __future__ import annotations

from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np

from core.reverse_engine import (
    Blocker,
    Direction,
    LevelType,
    OFFSETS,
    ReverseEngineState,
    SpecialCandy,
)

H = W = 9
NUM_PLANES = 24
NUM_COLOR_PLANES = 6
NUM_GLOBALS = 16
NUM_ACTIONS = 144  # 72 horizontal (9 x 8) + 72 vertical (8 x 9)


# ----------------------------------------------------------------------------- actions
def encode_action(r: int, c: int, d: Direction) -> int:
    if d == Direction.LEFT:
        c, d = c - 1, Direction.RIGHT
    elif d == Direction.UP:
        r, d = r - 1, Direction.DOWN
    if d == Direction.RIGHT:
        return r * (W - 1) + c
    return 72 + r * W + c


def decode_action(idx: int) -> Tuple[int, int, Direction]:
    if idx < 72:
        return idx // (W - 1), idx % (W - 1), Direction.RIGHT
    v = idx - 72
    return v // W, v % W, Direction.DOWN


def action_endpoints() -> Tuple[np.ndarray, np.ndarray]:
    """Flat cell indices (u, v) of the two cells swapped by each canonical action."""
    u = np.zeros(NUM_ACTIONS, dtype=np.int64)
    v = np.zeros(NUM_ACTIONS, dtype=np.int64)
    for a in range(NUM_ACTIONS):
        r, c, d = decode_action(a)
        dr, dc = OFFSETS[d]
        u[a] = r * W + c
        v[a] = (r + dr) * W + (c + dc)
    return u, v


def _flip_permutation() -> np.ndarray:
    """perm[a] = index of action a after mirroring the board left<->right."""
    perm = np.zeros(NUM_ACTIONS, dtype=np.int64)
    for a in range(NUM_ACTIONS):
        r, c, d = decode_action(a)
        if d == Direction.RIGHT:  # swap (r,c)-(r,c+1) -> (r,7-c)-(r,8-c)
            perm[a] = encode_action(r, (W - 2) - c, Direction.RIGHT)
        else:  # swap (r,c)-(r+1,c) -> (r,8-c)-(r+1,8-c)
            perm[a] = encode_action(r, (W - 1) - c, Direction.DOWN)
    return perm


FLIP_PERM = _flip_permutation()
ACTION_U, ACTION_V = action_endpoints()


def legal_mask_from_moves(moves: Iterable[Tuple[int, int, Direction]]) -> np.ndarray:
    mask = np.zeros(NUM_ACTIONS, dtype=bool)
    for r, c, d in moves:
        a = encode_action(r, c, d)
        if 0 <= a < NUM_ACTIONS:
            mask[a] = True
    return mask


# ----------------------------------------------------------------------------- objectives
def initial_objective_count(state: ReverseEngineState) -> int:
    return max(1, int(state.remaining_objectives_count()))


def _portal_cells(portals) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]:
    """Best-effort parser for King portal records (format varies across level files)."""
    ins, outs = [], []
    if not portals:
        return ins, outs
    for p in portals:
        try:
            if isinstance(p, dict):
                a = p.get("entrance") or p.get("in") or p.get("from")
                b = p.get("exit") or p.get("out") or p.get("to")
                if isinstance(a, dict):
                    a = (a.get("y", a.get("row")), a.get("x", a.get("col")))
                if isinstance(b, dict):
                    b = (b.get("y", b.get("row")), b.get("x", b.get("col")))
            else:
                a, b = p[0], p[1]
                if isinstance(a, dict):
                    a = (a.get("y"), a.get("x"))
                if isinstance(b, dict):
                    b = (b.get("y"), b.get("x"))
            ins.append((int(a[0]), int(a[1])))
            outs.append((int(b[0]), int(b[1])))
        except Exception:
            continue
    return ins, outs


# ----------------------------------------------------------------------------- encoder
def encode_state(
    state: ReverseEngineState,
    init_objectives: int,
    legal_moves: Optional[Sequence[Tuple[int, int, Direction]]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    planes = np.zeros((NUM_PLANES, H, W), dtype=np.float32)
    h, w = min(state.height, H), min(state.width, W)
    colors = state.colors[:h, :w]
    specials = state.specials[:h, :w]
    blockers = state.blockers[:h, :w]
    jellies = state.jellies[:h, :w]
    active = state.active_grid[:h, :w].astype(bool)

    for k in range(6):
        planes[k, :h, :w] = colors == (k + 1)
    planes[6, :h, :w] = specials == SpecialCandy.STRIPED_HORIZONTAL
    planes[7, :h, :w] = specials == SpecialCandy.STRIPED_VERTICAL
    planes[8, :h, :w] = specials == SpecialCandy.WRAPPED
    planes[9, :h, :w] = specials == SpecialCandy.COLOR_BOMB
    planes[10, :h, :w] = specials == SpecialCandy.JELLY_FISH
    planes[11, :h, :w] = blockers == Blocker.CHOCOLATE
    planes[12, :h, :w] = blockers == Blocker.LICORICE
    planes[13, :h, :w] = blockers == Blocker.FROSTING
    planes[14, :h, :w] = blockers != Blocker.NONE
    planes[15, :h, :w] = jellies == 1
    planes[16, :h, :w] = jellies >= 2
    planes[17, :h, :w] = state.ingredients[:h, :w] != 0
    planes[18, :h, :w] = active

    for c in range(w):
        col = np.where(active[:, c])[0]
        if col.size:
            planes[19, col[0], c] = 1.0
            planes[20, col[-1], c] = 1.0

    ins, outs = _portal_cells(getattr(state.config, "portals", None))
    for (r, c) in ins:
        if 0 <= r < H and 0 <= c < W:
            planes[21, r, c] = 1.0
    for (r, c) in outs:
        if 0 <= r < H and 0 <= c < W:
            planes[22, r, c] = 1.0

    if legal_moves:
        for r, c, d in legal_moves:
            dr, dc = OFFSETS[d]
            if 0 <= r < H and 0 <= c < W:
                planes[23, r, c] = 1.0
            if 0 <= r + dr < H and 0 <= c + dc < W:
                planes[23, r + dr, c + dc] = 1.0

    cfg = state.config
    remaining = float(state.remaining_objectives_count())
    moves_left = float(max(0, state.moves_left))
    moves_limit = float(max(1, getattr(cfg, "moves_limit", 30)))
    g = np.zeros(NUM_GLOBALS, dtype=np.float32)
    g[0] = moves_left / 50.0
    g[1] = moves_left / moves_limit
    g[2] = min(2.0, remaining / max(1.0, float(init_objectives)))
    g[3] = np.log1p(remaining) / 6.0
    g[4] = min(10.0, remaining / (moves_left + 1.0)) / 10.0
    lt = int(getattr(cfg, "level_type", LevelType.SCORE))
    if 0 <= lt < 4:
        g[5 + lt] = 1.0
    g[9] = float(state.num_colors) / 6.0
    tgt_ing = max(1, int(getattr(cfg, "target_ingredients", 0) or 0))
    g[10] = float(state.ingredients_collected) / tgt_ing if cfg.target_ingredients else 0.0
    tgt_score = max(1, int(getattr(cfg, "target_score", 1) or 1))
    g[11] = min(2.0, float(state.score) / tgt_score)
    n_active = max(1.0, float(active.sum()))
    g[12] = float((specials != 0).sum()) / 10.0
    g[13] = float((blockers != 0).sum()) / n_active
    g[14] = float((jellies > 0).sum()) / n_active
    g[15] = moves_limit / 60.0
    return planes, g


def flip_planes(planes: np.ndarray) -> np.ndarray:
    """Left-right mirror (gravity is vertical, so this is an exact game symmetry)."""
    return np.ascontiguousarray(planes[..., ::-1])


def flip_policy(vec: np.ndarray) -> np.ndarray:
    """Mirror a (..., 144) action vector: out[..., FLIP_PERM[a]] = vec[..., a]."""
    out = np.empty_like(vec)
    out[..., FLIP_PERM] = vec
    return out
