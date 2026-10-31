"""
Clairvoyant determinized beam-search oracle & topological level debugger for AlphaCandy-Zero.

Purpose (see alphacandy_zero/__init__.py):
  1. Separate engine/level-loader bugs from agent weakness:
     - Detects physically unreachable objectives (e.g. jellies on inactive cells) and
       1x1 isolated island cells (cells with zero active orthogonal neighbours, such as
       Level 279's (8,0), (8,4), (8,8), which can NEVER be swapped directly and can only be
       cleared by a remote Striped laser, Color Bomb, or Jelly Fish).
  2. Provide an upper-bound solver ("is this level winnable at all in ReverseCandyEngine?")
     using determinized width-B beam search with 64-bit Zobrist transposition deduplication
     and island-laser / special-combo targeting.
  3. Export golden winning trajectories in the exact replay-buffer format consumed by
     `alphacandy_zero.train`, bootstrapping expert iteration on hard bottleneck levels.

Usage:
  python -m alphacandy_zero.oracle --levels 235,272,279 --beam 24 --seeds 12 --out runs/oracle_300.pt
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from alphacandy_zero.encoding import (
    NUM_ACTIONS,
    encode_action,
    encode_state,
    initial_objective_count,
    legal_mask_from_moves,
)
from alphacandy_zero.search import final_bin
from core.reverse_engine import (
    Blocker,
    Direction,
    LevelConfig,
    LevelType,
    OFFSETS,
    ReverseCandyEngine,
    ReverseEngineState,
    SpecialCandy,
)


# ----------------------------------------------------------------------------- topology audit
@dataclass
class LevelDiagnosis:
    level_id: int
    level_type: str
    moves_limit: int
    active_cells: int
    initial_objectives: int
    required_pace: float
    void_objectives: int
    isolated_objective_cells: List[Tuple[int, int]]
    dead_columns_without_spawner: List[int]
    chocolate_cells: int
    frosting_cells: int
    licorice_cells: int
    solvable_in_principle: bool
    notes: List[str]


def diagnose_level(cfg: LevelConfig) -> LevelDiagnosis:
    active = cfg.active_grid.astype(bool)
    jellies = cfg.initial_jellies
    ingr = cfg.initial_ingredients
    blk = cfg.initial_blockers

    void_obj = int(np.sum(jellies[~active]) + np.sum(ingr[~active] > 0))
    isolated: List[Tuple[int, int]] = []
    for r in range(cfg.height):
        for c in range(cfg.width):
            if not active[r, c] or (jellies[r, c] == 0 and ingr[r, c] == 0):
                continue
            nbr_count = sum(
                1 for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1))
                if 0 <= r + dr < cfg.height and 0 <= c + dc < cfg.width and active[r + dr, c + dc]
            )
            # Both 0-neighbor isolated islands and 1-neighbor cul-de-sac dead-ends
            if nbr_count <= 1:
                isolated.append((r, c))

    dead_cols = [c for c in range(cfg.width) if not np.any(active[:, c])]
    state0 = ReverseEngineState(cfg)
    init_obj = int(state0.remaining_objectives_count())
    pace = init_obj / max(1, cfg.moves_limit)

    notes: List[str] = []
    if void_obj > 0:
        notes.append(f"BUG: {void_obj} objective(s) placed on inactive void cells!")
    if isolated:
        notes.append(
            f"ISOLATED_ISLANDS: {len(isolated)} objective cell(s) {isolated} have 0 active neighbours; "
            "direct swaps impossible — must clear via remote Striped laser / Color Bomb / Jelly Fish."
        )
    if pace >= 4.0 and cfg.level_type != LevelType.SCORE:
        notes.append(f"HIGH_PACE_FORTRESS: {pace:.2f} objectives/move requires mass special combos.")
    if int(np.sum(blk == Blocker.CHOCOLATE)) > 0:
        notes.append(f"CHOCOLATE_CONTAGION: {int(np.sum(blk == Blocker.CHOCOLATE))} initial chocolate spawners.")

    return LevelDiagnosis(
        level_id=cfg.level_id,
        level_type=cfg.level_type.name,
        moves_limit=cfg.moves_limit,
        active_cells=int(active.sum()),
        initial_objectives=init_obj,
        required_pace=round(pace, 3),
        void_objectives=void_obj,
        isolated_objective_cells=isolated,
        dead_columns_without_spawner=dead_cols,
        chocolate_cells=int(np.sum(blk == Blocker.CHOCOLATE)),
        frosting_cells=int(np.sum(blk == Blocker.FROSTING)),
        licorice_cells=int(np.sum(blk == Blocker.LICORICE)),
        solvable_in_principle=(void_obj == 0),
        notes=notes,
    )


# ----------------------------------------------------------------------------- zobrist hash
def _board_hash(state: ReverseEngineState) -> int:
    return hash((
        state.colors.tobytes(),
        state.specials.tobytes(),
        state.jellies.tobytes(),
        state.blockers.tobytes(),
        state.ingredients.tobytes(),
    ))


# ----------------------------------------------------------------------------- oracle scorer
def oracle_potential(
    state: ReverseEngineState,
    init_obj: int,
    isolated_cells: Sequence[Tuple[int, int]],
) -> float:
    """Dense tactical potential guiding the clairvoyant beam search."""
    if state.has_won():
        return 1e6 + state.moves_left * 1000.0
    if state.has_lost():
        return -1e5 - state.remaining_objectives_count() * 100.0

    rem = float(state.remaining_objectives_count())
    score = (init_obj - rem) * 120.0

    # 1. Extra reward for clearing isolated 1x1 island & cul-de-sac jellies (e.g. Level 279, 975, 1205)
    iso_alive_rows = set()
    iso_alive_cols = set()
    for r, c in isolated_cells:
        if state.jellies[r, c] == 0:
            score += 3500.0  # Decisive reward for breaking the bottleneck cell!
        else:
            score -= 1200.0  # Heavy penalty for leaving bottleneck cells uncleared!
            iso_alive_rows.add(r)
            iso_alive_cols.add(c)

    # 2. Special candies & laser alignment with remaining / isolated jellies
    spec = state.specials
    cb_coords = np.argwhere(spec == SpecialCandy.COLOR_BOMB)
    sh_coords = np.argwhere(spec == SpecialCandy.STRIPED_HORIZONTAL)
    sv_coords = np.argwhere(spec == SpecialCandy.STRIPED_VERTICAL)
    wr_coords = np.argwhere(spec == SpecialCandy.WRAPPED)
    jf_coords = np.argwhere(spec == SpecialCandy.JELLY_FISH)

    score += len(cb_coords) * 220.0 + len(jf_coords) * 160.0 + len(wr_coords) * 90.0
    score += (len(sh_coords) + len(sv_coords)) * 75.0

    # Reward vertical stripes in isolated columns and horizontal stripes in isolated rows
    for r, c in sv_coords:
        if int(c) in iso_alive_cols:
            score += 650.0
        elif np.any(state.jellies[:, c] > 0) or np.any(state.ingredients[:, c] > 0):
            score += 45.0
    for r, c in sh_coords:
        if int(r) in iso_alive_rows:
            score += 650.0
        elif np.any(state.jellies[r, :] > 0):
            score += 35.0

    # Reward adjacent special-special pairs ready to detonate next turn
    spec_mask = spec != SpecialCandy.NONE
    adj_pairs = int(np.sum(spec_mask[:, :-1] & spec_mask[:, 1:]) + np.sum(spec_mask[:-1, :] & spec_mask[1:, :]))
    score += adj_pairs * 320.0

    # Reward Color Bombs adjacent to candies that match isolated objective colors
    if len(cb_coords) > 0 and len(isolated_cells) > 0:
        iso_alive_colors = {int(state.colors[ir, ic]) for ir, ic in isolated_cells if state.jellies[ir, ic] > 0}
        for cbr, cbc in cb_coords:
            for ar, ac in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = int(cbr + ar), int(cbc + ac)
                if state.is_valid_cell(nr, nc) and int(state.colors[nr, nc]) in iso_alive_colors:
                    score += 850.0

    # 3. Ingredient downward gravity progress & unobstructed column lanes (e.g. Level 272)
    if state.config.level_type == LevelType.INGREDIENTS:
        ing_pos = np.argwhere(state.ingredients > 0)
        for r, c in ing_pos:
            score += float(r) * 140.0
            # Penalize blockers or candy distance directly below the ingredient in column c
            blockers_below = int(np.sum(state.blockers[r + 1 :, c] > 0))
            score -= blockers_below * 95.0

    # 4. Blocker suppression (Chocolate & Frosting)
    choco = int(np.sum(state.blockers == Blocker.CHOCOLATE))
    frost = int(np.sum(state.blockers == Blocker.FROSTING))
    score -= choco * 110.0 + frost * 22.0
    if state.chocolate_destroyed_this_turn:
        score += 95.0

    # 5. Score progress: if no jellies remain on a JELLY/ORDER level, reaching target_score is the sole win condition!
    if state.config.level_type == LevelType.JELLY and rem == 0:
        score += state.score / 80.0
    elif state.config.level_type == LevelType.SCORE:
        score += state.score / 80.0
    else:
        score += min(120.0, state.score / 2500.0)
    return float(score)


def _is_special_combo_swap(
    state: ReverseEngineState,
    r: int,
    c: int,
    d: Direction,
    isolated_cells: Sequence[Tuple[int, int]] = (),
) -> float:
    """Immediate bonus/penalty for executing or preserving special combinations."""
    dr, dc = OFFSETS[d]
    r2, c2 = r + dr, c + dc
    s1, s2 = int(state.specials[r, c]), int(state.specials[r2, c2])
    if s1 == SpecialCandy.COLOR_BOMB and s2 == SpecialCandy.COLOR_BOMB:
        return 800.0
    if (s1 == SpecialCandy.COLOR_BOMB and s2 != SpecialCandy.NONE) or (
        s2 == SpecialCandy.COLOR_BOMB and s1 != SpecialCandy.NONE
    ):
        return 520.0
    # Wrapped + Striped Cross-Cannon Combination
    is_wrapped_striped = (
        (s1 == SpecialCandy.WRAPPED and s2 in (SpecialCandy.STRIPED_HORIZONTAL, SpecialCandy.STRIPED_VERTICAL))
        or (s2 == SpecialCandy.WRAPPED and s1 in (SpecialCandy.STRIPED_HORIZONTAL, SpecialCandy.STRIPED_VERTICAL))
    )
    if is_wrapped_striped:
        if isolated_cells:
            # 3-row horizontal and 3-column vertical blast
            for ir, ic in isolated_cells:
                if state.jellies[ir, ic] > 0 and (abs(r - ir) <= 1 or abs(c - ic) <= 1):
                    return 1450.0
        return 650.0

    if s1 != SpecialCandy.NONE and s2 != SpecialCandy.NONE:
        return 320.0
    if s1 == SpecialCandy.COLOR_BOMB or s2 == SpecialCandy.COLOR_BOMB:
        target_col = int(state.colors[r2, c2] if s1 == SpecialCandy.COLOR_BOMB else state.colors[r, c])
        iso_alive = [(ir, ic) for ir, ic in isolated_cells if state.jellies[ir, ic] > 0]
        if iso_alive:
            hits = sum(1 for ir, ic in iso_alive if int(state.colors[ir, ic]) == target_col)
            if hits > 0:
                return 1200.0 * hits
            if state.moves_left <= 5:
                return 250.0  # Endgame desperation: detonate to trigger massive board wipe & cascade
            return -150.0  # preserve Color Bomb until it can snipe an isolated island or pair with a special!
        return 90.0

    # Direct Striped Laser Snipe on isolated objective cells (bidirectional swap check)
    if isolated_cells:
        for ir, ic in isolated_cells:
            if state.jellies[ir, ic] > 0:
                if (s1 == SpecialCandy.STRIPED_HORIZONTAL and (r == ir or r2 == ir)) or (
                    s2 == SpecialCandy.STRIPED_HORIZONTAL and (r == ir or r2 == ir)
                ):
                    return 1100.0
                if (s1 == SpecialCandy.STRIPED_VERTICAL and (c == ic or c2 == ic)) or (
                    s2 == SpecialCandy.STRIPED_VERTICAL and (c == ic or c2 == ic)
                ):
                    return 1100.0

    return 0.0


# ----------------------------------------------------------------------------- beam oracle
@dataclass
class _BeamNode:
    state: ReverseEngineState
    score: float
    rng_seed: int
    history: List[Tuple[ReverseEngineState, List[Tuple[int, int, Direction]], int]]


def solve_with_oracle(
    cfg: LevelConfig,
    beam_width: int = 16,
    branch_per_node: int = 10,
    max_seeds: int = 8,
    base_seed: int = 42,
) -> Dict[str, Any]:
    """Runs determinized width-B beam search across up to `max_seeds` initial boards."""
    t0 = time.time()
    diag = diagnose_level(cfg)
    engine = ReverseCandyEngine()
    best_overall: Optional[_BeamNode] = None
    best_seed = base_seed

    for s_idx in range(max_seeds):
        seed = (cfg.level_id * 10007 + s_idx * 7919 + base_seed) % 1000003
        random.seed(seed)
        np.random.seed(seed)
        root = ReverseEngineState(cfg)
        engine.initialize_board(root, seed=seed)
        init_obj = initial_objective_count(root)

        beam: List[_BeamNode] = [
            _BeamNode(
                state=root,
                score=oracle_potential(root, init_obj, diag.isolated_objective_cells),
                rng_seed=seed * 31 + 7,
                history=[],
            )
        ]

        for step in range(cfg.moves_limit):
            next_beam: List[_BeamNode] = []
            seen_hashes = set()

            for node in beam:
                if node.state.has_won():
                    next_beam.append(node)
                    continue
                if node.state.has_lost():
                    continue

                st = node.state
                legal = engine.get_legal_moves(st)
                if not legal:
                    st = st.copy()
                    random.seed(node.rng_seed)
                    np.random.seed(node.rng_seed % (2**32 - 1))
                    if not engine.shuffle_board(st):
                        continue
                    legal = engine.get_legal_moves(st)
                    if not legal:
                        continue

                # Prioritize special-involving swaps, ingredient-column swaps, and bottom-row swaps
                if len(legal) > branch_per_node:
                    ing_cols = set(np.where(np.any(st.ingredients > 0, axis=0))[0].tolist())
                    pri_moves = []
                    reg_moves = []
                    for mv in legal:
                        r, c, d = mv
                        dr, dc = OFFSETS[d]
                        r2, c2 = r + dr, c + dc
                        if (
                            st.specials[r, c] != 0
                            or st.specials[r2, c2] != 0
                            or c in ing_cols
                            or c2 in ing_cols
                            or r >= 6
                        ):
                            pri_moves.append(mv)
                        else:
                            reg_moves.append(mv)
                    # Sort priority moves so bottom-most moves come first
                    pri_moves.sort(key=lambda m: -m[0])
                    keep_reg = max(2, branch_per_node - len(pri_moves))
                    if len(reg_moves) > keep_reg:
                        stride = max(1, len(reg_moves) // keep_reg)
                        reg_moves = reg_moves[::stride][:keep_reg]
                    candidates = (pri_moves + reg_moves)[: branch_per_node + 6]
                else:
                    candidates = legal

                for r, c, d in candidates:
                    a_idx = encode_action(r, c, d)
                    child_seed = (node.rng_seed * 1103515245 + a_idx * 12345 + step) & 0x7FFFFFFF
                    random.seed(child_seed)
                    np.random.seed(child_seed)

                    ch = st.copy()
                    combo_bonus = _is_special_combo_swap(st, r, c, d, diag.isolated_objective_cells)
                    engine.execute_move(ch, r, c, d)

                    if ch.has_won():
                        win_node = _BeamNode(
                            state=ch,
                            score=1e6 + ch.moves_left * 1000.0,
                            rng_seed=child_seed,
                            history=node.history + [(st, legal, a_idx)],
                        )
                        return _package_oracle_result(cfg, diag, win_node, init_obj, seed, s_idx + 1, time.time() - t0)

                    bh = _board_hash(ch)
                    if bh in seen_hashes:
                        continue
                    seen_hashes.add(bh)

                    pot = oracle_potential(ch, init_obj, diag.isolated_objective_cells) + combo_bonus
                    next_beam.append(
                        _BeamNode(
                            state=ch,
                            score=pot,
                            rng_seed=child_seed,
                            history=node.history + [(st, legal, a_idx)],
                        )
                    )

            if not next_beam:
                break
            next_beam.sort(key=lambda n: -n.score)
            beam = next_beam[:beam_width]
            if best_overall is None or beam[0].score > best_overall.score:
                best_overall = beam[0]
                best_seed = seed

    assert best_overall is not None
    init_obj = max(1, diag.initial_objectives)
    return _package_oracle_result(cfg, diag, best_overall, init_obj, best_seed, max_seeds, time.time() - t0)


def _package_oracle_result(
    cfg: LevelConfig,
    diag: LevelDiagnosis,
    node: _BeamNode,
    init_obj: int,
    seed: int,
    attempts_used: int,
    elapsed: float,
) -> Dict[str, Any]:
    won = bool(node.state.has_won())
    final_rem = int(node.state.remaining_objectives_count())
    final_jelly = (
        (node.state.jellies[:9, :9] > 0).reshape(-1)
        if node.state.jellies.shape == (9, 9)
        else np.zeros(81, bool)
    )
    samples = []
    for st, legal, a_idx in node.history:
        planes, g = encode_state(st, init_obj, legal)
        pi = np.zeros(NUM_ACTIONS, np.float32)
        pi[a_idx] = 1.0
        jel_now = (st.jellies[:9, :9] > 0).reshape(-1) if st.jellies.shape == (9, 9) else np.zeros(81, bool)
        samples.append({
            "planes": planes.astype(np.uint8),
            "globals": g,
            "mask": legal_mask_from_moves(legal),
            "pi": pi,
            "z": 1.0 if won else 0.0,
            "fbin": final_bin(final_rem, int(st.remaining_objectives_count()), won),
            "jmask": jel_now,
            "jtgt": jel_now & (~final_jelly),
        })
    return {
        "level_id": cfg.level_id,
        "won": won,
        "attempts_used": attempts_used,
        "winning_seed": seed if won else None,
        "moves_used": len(node.history),
        "moves_limit": cfg.moves_limit,
        "moves_remaining": node.state.moves_left,
        "final_score": node.state.score,
        "objectives_remaining": final_rem,
        "initial_objectives": init_obj,
        "elapsed_sec": round(elapsed, 2),
        "diagnosis": diag.__dict__,
        "samples": samples,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--levels", default="235,272,279")
    ap.add_argument("--beam", type=int, default=20)
    ap.add_argument("--branch", type=int, default=12)
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    from alphacandy_zero.train import parse_levels
    from core.king_level_loader import KingLevelLoader

    loader = KingLevelLoader()
    levels = parse_levels(args.levels)
    all_samples = []
    summary = []

    for lvl in levels:
        cfg = loader.load_level(lvl)
        res = solve_with_oracle(cfg, beam_width=args.beam, branch_per_node=args.branch, max_seeds=args.seeds)
        if res["won"]:
            all_samples.extend(res["samples"])
        rec = {k: v for k, v in res.items() if k != "samples"}
        summary.append(rec)
        print(json.dumps(rec), flush=True)

    if args.out and all_samples:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        torch.save(all_samples, args.out)
        print(f"[oracle] saved {len(all_samples)} golden expert transitions to {args.out}")


if __name__ == "__main__":
    main()
