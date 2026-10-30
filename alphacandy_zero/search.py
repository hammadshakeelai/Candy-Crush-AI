"""
Gumbel AlphaZero search over the TRUE stochastic ReverseCandyEngine.

References
  * Danihelka, Guez, Schrittwieser, Silver (ICLR 2022) "Policy improvement by planning with Gumbel"
    - Gumbel-top-k sampling without replacement + Sequential Halving at the root
    - completed-Q improved policy target  pi' = softmax(logits + sigma(completedQ))
    - guaranteed policy improvement even with very few simulations (we use 16-64)
  * Antonoglou et al. (ICLR 2022) "Stochastic MuZero": chance outcomes must be averaged, never
    trusted from a single sample. Here every simulation re-executes the move on a fresh copy, so
    each visit is an independent draw of the refill RNG and Q(a) is a Monte-Carlo chance average.

Why this replaces the legacy GrandmasterConsensusAgent lookahead:
    the legacy agent scored each candidate from ONE stochastic sample with a hand-tuned linear
    score (consensus*10 + score/100 + objectives*25) and at most depth 2. One lucky refill could
    flip the decision; that is the main reason it needed up to 80 "seed rotations" to win a level.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from alphacandy_zero.encoding import (
    NUM_ACTIONS,
    decode_action,
    encode_action,
    encode_state,
    flip_planes,
    flip_policy,
    initial_objective_count,
    legal_mask_from_moves,
)
from alphacandy_zero.network import FINAL_BIN_EDGES
from core.reverse_engine import Direction, ReverseCandyEngine, ReverseEngineState


# ----------------------------------------------------------------------------- evaluation
class Evaluator:
    """Batched network evaluation; optional left-right flip ensemble (exact symmetry)."""

    def __init__(self, net: Optional[torch.nn.Module], device: str = "cpu", flip_ensemble: bool = False):
        self.net = net
        self.device = device
        self.flip_ensemble = flip_ensemble
        if net is not None:
            net.eval()

    @torch.no_grad()
    def __call__(
        self,
        states: Sequence[ReverseEngineState],
        init_obj: int,
        legal_lists: Sequence[Optional[Sequence[Tuple[int, int, Direction]]]],
    ) -> Tuple[np.ndarray, np.ndarray]:
        n = len(states)
        if self.net is None or n == 0:
            return np.zeros((n, NUM_ACTIONS), np.float32), np.full(n, 0.5, np.float32)
        planes = np.zeros((n, 24, 9, 9), np.float32)
        glob = np.zeros((n, 16), np.float32)
        masks = np.zeros((n, NUM_ACTIONS), bool)
        for i, (s, lg) in enumerate(zip(states, legal_lists)):
            planes[i], glob[i] = encode_state(s, init_obj, lg)
            masks[i] = legal_mask_from_moves(lg) if lg else True
        p = torch.from_numpy(planes).to(self.device)
        g = torch.from_numpy(glob).to(self.device)
        out = self.net(p, g)
        logits = out["policy_logits"].float().cpu().numpy()
        win = torch.sigmoid(out["win_logit"]).squeeze(-1).float().cpu().numpy()
        if self.flip_ensemble:
            out2 = self.net(torch.from_numpy(flip_planes(planes)).to(self.device), g)
            logits = 0.5 * (logits + flip_policy(out2["policy_logits"].float().cpu().numpy()))
            win = 0.5 * (win + torch.sigmoid(out2["win_logit"]).squeeze(-1).float().cpu().numpy())
        logits = np.where(masks, logits, -1e9).astype(np.float32)
        return logits, win.astype(np.float32)


def heuristic_value(state: ReverseEngineState, init_obj: int, init_moves: int) -> float:
    """Pace + tactical micro-potential heuristic used to bootstrap before the value net is trained.

    Projects the clearing rate achieved so far over the remaining moves and augments it with:
      1. Fractional ingredient drop depth (so sinking an ingredient from r=1 -> r=7 is rewarded
         before it exits at r=8).
      2. Active special candy inventory & row/column laser alignment with remaining jellies
         (critical for isolated 1x1 island cells like Level 279's (8,0), (8,4), (8,8)).
      3. Blocker suppression bonus (clearing frosting/chocolate that chokes the board).
    Returns a pseudo win-probability in (0, 1).
    """
    if state.has_won():
        return 1.0
    remaining = float(state.remaining_objectives_count())
    moves_left = float(max(0, state.moves_left))
    if moves_left <= 0:
        return 0.0

    # 1. Fractional progress from ingredient sinking & blocker clearing
    sub_progress = 0.0
    ing_rows, _ = np.where(state.ingredients > 0)
    if ing_rows.size > 0:
        sub_progress += float(np.sum(ing_rows / max(1.0, float(state.height - 1)))) * 0.65

    # 2. Special candy potential & laser alignment with remaining jellies
    spec = state.specials
    n_cb = int(np.sum(spec == 4))  # COLOR_BOMB
    n_fish = int(np.sum(spec == 5))  # JELLY_FISH
    n_wrap = int(np.sum(spec == 3))  # WRAPPED
    n_sh = int(np.sum(spec == 1))  # STRIPED_HORIZONTAL
    n_sv = int(np.sum(spec == 2))  # STRIPED_VERTICAL
    spec_bonus = n_cb * 1.2 + n_fish * 0.9 + n_wrap * 0.45 + (n_sh + n_sv) * 0.35

    if np.any(state.jellies > 0):
        j_rows = np.any(state.jellies > 0, axis=1)
        j_cols = np.any(state.jellies > 0, axis=0)
        sh_rows = np.any(spec == 1, axis=1)
        sv_cols = np.any(spec == 2, axis=0)
        spec_bonus += 0.65 * float(np.sum(j_rows & sh_rows) + np.sum(j_cols & sv_cols))

    # 3. Blocker penalty (fewer blockers = higher board openness)
    init_blk = float(np.sum(state.config.initial_blockers > 0))
    cur_blk = float(np.sum(state.blockers > 0))
    blk_cleared = (init_blk - cur_blk) * 0.20

    used = max(1.0, float(init_moves) - moves_left)
    effective_cleared = max(0.0, float(init_obj) - remaining + sub_progress + spec_bonus + blk_cleared)
    capacity = effective_cleared / used * moves_left
    z = 2.5 * (capacity - remaining) / (math.sqrt(max(remaining, 1.0)) + 1.0)
    return float(1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, z)))))


def terminal_or_none(state: ReverseEngineState, init_obj: int) -> Optional[float]:
    if state.has_won():
        return 1.0
    if state.moves_left <= 0:
        # tiny shaping so that among losses, closer misses rank higher (never exceeds a real win chance)
        return 0.02 * (1.0 - min(1.0, state.remaining_objectives_count() / max(1, init_obj)))
    return None


# ----------------------------------------------------------------------------- search
@dataclass
class SearchConfig:
    num_simulations: int = 32
    max_considered: int = 8
    c_visit: float = 50.0
    c_scale: float = 0.1
    net_weight: float = 1.0  # lambda: leaf = lambda*net + (1-lambda)*heuristic (anneal 0 -> 1)
    leaf_rollout: int = 0  # greedy policy steps after the root action before evaluating


@dataclass
class SearchResult:
    action: int
    move: Tuple[int, int, Direction]
    pi_target: np.ndarray
    root_value: float
    q: Dict[int, float]
    visits: Dict[int, int]


class GumbelSearcher:
    def __init__(self, engine: ReverseCandyEngine, evaluator: Evaluator, cfg: SearchConfig):
        self.engine = engine
        self.evaluate = evaluator
        self.cfg = cfg

    def _sigma(self, q: np.ndarray, max_visits: int) -> np.ndarray:
        return (self.cfg.c_visit + max_visits) * self.cfg.c_scale * q

    def _leaf_values(self, children: List[ReverseEngineState], init_obj: int, init_moves: int) -> np.ndarray:
        vals = np.zeros(len(children), np.float32)
        pending: List[int] = []
        for i, ch in enumerate(children):
            t = terminal_or_none(ch, init_obj)
            if t is None:
                pending.append(i)
            else:
                vals[i] = t
        # optional greedy policy rollout (batched)
        for _ in range(self.cfg.leaf_rollout):
            if not pending:
                break
            legal = [self.engine.get_legal_moves(children[i]) for i in pending]
            logits, _ = self.evaluate([children[i] for i in pending], init_obj, legal)
            still = []
            for j, i in enumerate(pending):
                if not legal[j]:
                    if not self.engine.shuffle_board(children[i]):
                        vals[i] = 0.0
                        continue
                    still.append(i)
                    continue
                r, c, d = decode_action(int(np.argmax(logits[j])))
                self.engine.execute_move(children[i], r, c, d)
                t = terminal_or_none(children[i], init_obj)
                if t is None:
                    still.append(i)
                else:
                    vals[i] = t
            pending = still
        if pending:
            states = [children[i] for i in pending]
            lam = self.cfg.net_weight
            if lam > 0:
                legal = [None] * len(states)
                _, win = self.evaluate(states, init_obj, legal)
            else:
                win = np.zeros(len(states), np.float32)
            for j, i in enumerate(pending):
                h = heuristic_value(states[j], init_obj, init_moves) if lam < 1 else 0.0
                vals[i] = lam * float(win[j]) + (1.0 - lam) * h
        return vals

    def search(
        self,
        state: ReverseEngineState,
        legal: Sequence[Tuple[int, int, Direction]],
        init_obj: int,
        init_moves: int,
        add_noise: bool = True,
    ) -> SearchResult:
        logits_all, root_win = self.evaluate([state], init_obj, [legal])
        logits_all = logits_all[0]
        legal_idx = sorted({encode_action(r, c, d) for r, c, d in legal})
        legal_moves = {encode_action(r, c, d): (r, c, d) for r, c, d in legal}
        root_v = float(root_win[0]) if self.cfg.net_weight > 0 else heuristic_value(state, init_obj, init_moves)

        la = np.array([logits_all[a] for a in legal_idx], np.float64)
        la = la - la.max()
        gumbel = np.random.gumbel(size=len(legal_idx)) if add_noise else np.zeros(len(legal_idx))
        m = min(self.cfg.max_considered, len(legal_idx))
        order = np.argsort(-(la + gumbel))
        remaining = [int(k) for k in order[:m]]

        q_sum = np.zeros(len(legal_idx))
        n_vis = np.zeros(len(legal_idx), np.int64)

        if len(legal_idx) > 1:
            phases = max(1, math.ceil(math.log2(m))) if m > 1 else 1
            budget_left = self.cfg.num_simulations
            for ph in range(phases):
                if not remaining or budget_left <= 0:
                    break
                per = max(1, self.cfg.num_simulations // (phases * len(remaining)))
                children, owners = [], []
                for k in remaining:
                    r, c, d = legal_moves[legal_idx[k]]
                    for _ in range(per):
                        ch = state.copy()
                        self.engine.execute_move(ch, r, c, d)
                        children.append(ch)
                        owners.append(k)
                budget_left -= len(children)
                vals = self._leaf_values(children, init_obj, init_moves)
                for k, v in zip(owners, vals):
                    q_sum[k] += v
                    n_vis[k] += 1
                if len(remaining) > 1:
                    q = np.where(n_vis > 0, q_sum / np.maximum(n_vis, 1), 0.0)
                    score = gumbel + la + self._sigma(q, int(n_vis.max()))
                    remaining.sort(key=lambda k: -score[k])
                    remaining = remaining[: max(1, len(remaining) // 2)]

        q = np.where(n_vis > 0, q_sum / np.maximum(n_vis, 1), 0.0)
        max_n = int(n_vis.max()) if n_vis.size else 0
        if len(legal_idx) == 1:
            best = 0
        else:
            final = gumbel + la + self._sigma(q, max_n)
            best = max(remaining, key=lambda k: final[k])

        # completed-Q improved policy target (Danihelka et al., eq. 33 "v_mix")
        prior = np.exp(la)
        prior /= prior.sum()
        visited = n_vis > 0
        if visited.any():
            pv = (prior[visited] * q[visited]).sum() / max(prior[visited].sum(), 1e-8)
            n_tot = n_vis.sum()
            v_mix = (root_v + n_tot * pv) / (1.0 + n_tot)
        else:
            v_mix = root_v
        completed = np.where(visited, q, v_mix)
        tgt_logits = la + self._sigma(completed, max_n)
        tgt = np.exp(tgt_logits - tgt_logits.max())
        tgt /= tgt.sum()
        pi_target = np.zeros(NUM_ACTIONS, np.float32)
        for k, a in enumerate(legal_idx):
            pi_target[a] = tgt[k]

        a_best = legal_idx[best]
        return SearchResult(
            action=a_best,
            move=legal_moves[a_best],
            pi_target=pi_target,
            root_value=root_v,
            q={legal_idx[k]: float(q[k]) for k in range(len(legal_idx)) if n_vis[k] > 0},
            visits={legal_idx[k]: int(n_vis[k]) for k in range(len(legal_idx)) if n_vis[k] > 0},
        )


# ----------------------------------------------------------------------------- full game
def final_bin(final_remaining: int, now_remaining: int, won: bool) -> int:
    if won or final_remaining <= 0:
        return 0
    frac = final_remaining / max(1, now_remaining)
    for i, edge in enumerate(FINAL_BIN_EDGES[1:], start=1):
        if frac <= edge:
            return i
    return len(FINAL_BIN_EDGES)  # > 1.0 (objectives grew, e.g. chocolate)


def play_game(
    engine: ReverseCandyEngine,
    searcher: GumbelSearcher,
    config,
    seed: int,
    start: Optional[Dict] = None,
    record: bool = True,
    add_noise: bool = True,
    snapshot_moves_left: Tuple[int, ...] = (3, 6, 10),
) -> Dict:
    """Plays one game. `start` = {"state": ReverseEngineState, "init_obj": int} for restarts."""
    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))
    if start is not None:
        state = start["state"].copy()
        init_obj = int(start["init_obj"])
    else:
        state = ReverseEngineState(config)
        engine.initialize_board(state, seed=seed)
        random.seed(seed * 7 + 1)
        init_obj = initial_objective_count(state)
    init_moves = int(config.moves_limit)

    recs: List[Tuple] = []
    snaps: List[Dict] = []
    shuffles = 0
    while not state.has_won() and state.moves_left > 0:
        legal = engine.get_legal_moves(state)
        if not legal:
            if not engine.shuffle_board(state) or shuffles > 20:
                break
            shuffles += 1
            continue
        res = searcher.search(state, legal, init_obj, init_moves, add_noise=add_noise)
        if record:
            planes, g = encode_state(state, init_obj, legal)
            recs.append((
                planes.astype(np.uint8),  # all planes are binary
                g,
                legal_mask_from_moves(legal),
                res.pi_target,
                (state.jellies[:9, :9] > 0).reshape(-1).copy() if state.jellies.shape == (9, 9) else None,
                int(state.remaining_objectives_count()),
            ))
        if state.moves_left in snapshot_moves_left:
            snaps.append({"state": state.copy(), "init_obj": init_obj})
        r, c, d = res.move
        engine.execute_move(state, r, c, d)

    won = bool(state.has_won())
    final_rem = int(state.remaining_objectives_count())
    final_jelly = (state.jellies[:9, :9] > 0).reshape(-1) if state.jellies.shape == (9, 9) else None
    samples = []
    for planes, g, mask, pi, jel_now, rem_now in recs:
        if jel_now is not None and final_jelly is not None:
            jmask = jel_now
            jtgt = jel_now & (~final_jelly)
        else:
            jmask = np.zeros(81, bool)
            jtgt = np.zeros(81, bool)
        samples.append({
            "planes": planes, "globals": g, "mask": mask, "pi": pi,
            "z": 1.0 if won else 0.0,
            "fbin": final_bin(final_rem, rem_now, won),
            "jmask": jmask, "jtgt": jtgt,
        })
    return {
        "won": won,
        "final_remaining": final_rem,
        "init_obj": init_obj,
        "moves_used": init_moves - max(0, state.moves_left) if start is None else None,
        "samples": samples,
        "snapshots": snaps if not won else [],
    }
