"""
AlphaCandy-Zero expert-iteration trainer.

Loop (repeat until the time budget runs out):
  1. ACT    : N CPU workers play self-play games with Gumbel search in the true simulator.
              Level choice = Prioritized Level Replay (Jiang, Grefenstette, Rocktaschel ICML 2021):
              levels whose current win rate is neither 0 nor 1 have the most learning signal.
              With prob p_restart a game restarts from a saved NEAR-MISS endgame state of that level
              (Go-Explore style, Ecoffet et al. Nature 2021) - this is targeted practice of exactly
              the clutch positions where we lose.
  2. LEARN  : train the net on a replay buffer with
                 policy  CE(pi_search_improved, pi_net)          (expert iteration target)
                 value   BCE(win, z)
                 aux     CE(final-objectives bin) + masked BCE(per-cell jelly cleared)  (KataGo-style)
              + exact left-right mirror augmentation. Colour symmetry is built into the net.
  3. ANNEAL : leaf value = lambda*net + (1-lambda)*pace-heuristic, lambda 0 -> 1 over warm-up, so
              the very first iteration is already a strong heuristic search, not random play.

Usage (Kaggle 4 vCPU, ~11h):
  python -m alphacandy_zero.train --levels 1-2500 --focus-file campaign/unsolved_levels_1_2500.json \
      --preset cpu --sims 32 --games-per-iter 96 --time-budget-hours 11 --out /kaggle/working/zero
Smoke test (local, <1 min):
  python -m alphacandy_zero.train --levels 1-5 --preset tiny --sims 8 --games-per-iter 4 \
      --iterations 1 --workers 2 --max-moves-override 6 --out /tmp/zero_smoke
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from collections import deque
from multiprocessing import get_context
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F

from alphacandy_zero.encoding import FLIP_PERM, NUM_ACTIONS
from alphacandy_zero.network import AlphaCandyZeroNet, count_params

# ----------------------------------------------------------------------------- worker side
_W: Dict = {}


def _worker_init(preset: str, max_moves_override: int):
    torch.set_num_threads(1)
    from core.king_level_loader import KingLevelLoader
    from core.reverse_engine import ReverseCandyEngine

    _W["loader"] = KingLevelLoader()
    _W["engine"] = ReverseCandyEngine()
    _W["net"] = AlphaCandyZeroNet(preset).eval()
    _W["version"] = -1
    _W["max_moves_override"] = max_moves_override
    _W["cfg_cache"] = {}


def _get_cfg(level_id: int):
    cache = _W["cfg_cache"]
    if level_id not in cache:
        cfg = _W["loader"].load_level(level_id)
        if _W["max_moves_override"] > 0:
            cfg.moves_limit = min(cfg.moves_limit, _W["max_moves_override"])
        cache[level_id] = cfg
    return cache[level_id]


def _worker_play(task: Dict) -> Dict:
    from alphacandy_zero.search import Evaluator, GumbelSearcher, SearchConfig, play_game

    if task["version"] != _W["version"] and task.get("weights"):
        sd = torch.load(task["weights"], map_location="cpu", weights_only=True)
        _W["net"].load_state_dict(sd)
        _W["net"].eval()
        _W["version"] = task["version"]
    scfg = SearchConfig(**task["search"])
    searcher = GumbelSearcher(_W["engine"], Evaluator(_W["net"]), scfg)
    t0 = time.time()
    try:
        out = play_game(
            _W["engine"], searcher, _get_cfg(task["level"]), task["seed"],
            start=task.get("start"), record=task.get("record", True), add_noise=task.get("noise", True),
        )
    except Exception as e:  # never kill the fleet because of one engine edge case
        return {"level": task["level"], "error": repr(e), "won": False, "samples": [], "snapshots": []}
    out["level"] = task["level"]
    out["restart"] = task.get("start") is not None
    out["sec"] = time.time() - t0
    return out


# ----------------------------------------------------------------------------- level sampler
class PrioritizedLevelReplay:
    def __init__(self, levels: List[int], focus: Optional[List[int]] = None, focus_frac: float = 0.5,
                 beta: float = 0.3, rho: float = 0.3):
        self.levels = list(levels)
        self.focus = [l for l in (focus or []) if l in set(levels)]
        self.focus_frac = focus_frac if self.focus else 0.0
        self.beta, self.rho = beta, rho
        self.p_win = {l: None for l in self.levels}
        self.last_seen = {l: 0 for l in self.levels}
        self.games = {l: 0 for l in self.levels}
        self.t = 0
        self.snapshots: Dict[int, deque] = {}

    def _score(self, l: int) -> float:
        p = self.p_win[l]
        if p is None:
            return 1.0  # unseen levels first
        return p * (1 - p) + 0.05 * (1 - p)  # learning potential, slight bias toward unsolved

    def sample(self, k: int) -> List[int]:
        out = []
        for _ in range(k):
            pool = self.focus if (self.focus and random.random() < self.focus_frac) else self.levels
            scores = np.array([self._score(l) for l in pool])
            ranks = np.empty(len(pool))
            ranks[np.argsort(-scores)] = np.arange(1, len(pool) + 1)
            w = (1.0 / ranks) ** (1.0 / self.beta)
            w /= w.sum()
            stale = np.array([self.t - self.last_seen[l] for l in pool], dtype=np.float64) + 1.0
            stale /= stale.sum()
            p = (1 - self.rho) * w + self.rho * stale
            out.append(int(pool[np.random.choice(len(pool), p=p / p.sum())]))
        return out

    def update(self, level: int, won: bool, restart: bool):
        self.t += 1
        self.last_seen[level] = self.t
        if restart:
            return  # restarts are practice, they do not measure the full-level win rate
        self.games[level] += 1
        p = self.p_win[level]
        self.p_win[level] = float(won) if p is None else 0.8 * p + 0.2 * float(won)

    def add_snapshots(self, level: int, snaps: List[Dict], final_remaining: int, init_obj: int):
        # keep only near misses (finished with <= 35% of objectives left)
        if not snaps or final_remaining > max(2, 0.35 * init_obj):
            return
        dq = self.snapshots.setdefault(level, deque(maxlen=8))
        dq.extend(snaps)

    def state_dict(self) -> Dict:
        return {"p_win": self.p_win, "games": self.games}


# ----------------------------------------------------------------------------- learner
def _collate(batch: List[Dict], device: str, flip_prob: float = 0.5):
    planes = np.stack([b["planes"] for b in batch]).astype(np.float32)
    glob = np.stack([b["globals"] for b in batch]).astype(np.float32)
    mask = np.stack([b["mask"] for b in batch])
    pi = np.stack([b["pi"] for b in batch]).astype(np.float32)
    z = np.array([b["z"] for b in batch], np.float32)
    fbin = np.array([b["fbin"] for b in batch], np.int64)
    jm = np.stack([b["jmask"] for b in batch]).astype(np.float32)
    jt = np.stack([b["jtgt"] for b in batch]).astype(np.float32)
    flip = np.random.rand(len(batch)) < flip_prob
    if flip.any():
        planes[flip] = planes[flip][..., ::-1]
        for arr in (mask, pi):
            tmp = np.empty_like(arr[flip])
            tmp[:, FLIP_PERM] = arr[flip]
            arr[flip] = tmp
        jm[flip] = jm[flip].reshape(-1, 9, 9)[..., ::-1].reshape(-1, 81)
        jt[flip] = jt[flip].reshape(-1, 9, 9)[..., ::-1].reshape(-1, 81)
    t = lambda a: torch.from_numpy(np.ascontiguousarray(a)).to(device)
    return t(planes), t(glob), t(mask), t(pi), t(z), t(fbin), t(jm), t(jt)


def train_steps(net, opt, buffer: deque, steps: int, batch_size: int, device: str) -> Dict[str, float]:
    net.train()
    agg = {"loss": 0.0, "policy": 0.0, "value": 0.0, "fbin": 0.0, "jelly": 0.0}
    if len(buffer) < batch_size:
        return agg
    buf = list(buffer)
    for _ in range(steps):
        batch = random.sample(buf, batch_size)
        planes, glob, mask, pi, z, fbin, jm, jt = _collate(batch, device)
        out = net(planes, glob, mask)
        logp = F.log_softmax(out["policy_logits"], dim=-1)
        l_pol = -(pi * logp).sum(-1).mean()
        l_val = F.binary_cross_entropy_with_logits(out["win_logit"].squeeze(-1), z)
        l_fb = F.cross_entropy(out["final_bins"], fbin)
        jl = F.binary_cross_entropy_with_logits(out["jelly_logits"], jt, reduction="none")
        l_j = (jl * jm).sum() / jm.sum().clamp(min=1.0)
        loss = l_pol + l_val + 0.5 * l_fb + 0.5 * l_j
        if not torch.isfinite(loss):
            continue
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        for k, v in zip(agg, (loss, l_pol, l_val, l_fb, l_j)):
            agg[k] += float(v.detach().item()) / steps
    net.eval()
    return agg


# ----------------------------------------------------------------------------- utils
def parse_levels(spec: str) -> List[int]:
    out: List[int] = []
    for part in spec.split(","):
        part = part.strip()
        if "-" in part:
            a, b = part.split("-")
            out.extend(range(int(a), int(b) + 1))
        elif part:
            out.append(int(part))
    return out


def load_focus(path: Optional[str]) -> List[int]:
    if not path or not os.path.exists(path):
        return []
    data = json.load(open(path))
    if isinstance(data, dict):
        for key in ("unsolved", "unsolved_levels", "levels"):
            if key in data:
                data = data[key]
                break
    return [int(x if not isinstance(x, dict) else x.get("level", x.get("level_id"))) for x in data]


def atomic_save(obj, path: str):
    tmp = path + ".tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path)


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--levels", default="1-300")
    ap.add_argument("--focus-file", default=None)
    ap.add_argument("--focus-frac", type=float, default=0.5)
    ap.add_argument("--preset", default="cpu")
    ap.add_argument("--sims", type=int, default=32)
    ap.add_argument("--considered", type=int, default=8)
    ap.add_argument("--leaf-rollout", type=int, default=0)
    ap.add_argument("--games-per-iter", type=int, default=96)
    ap.add_argument("--iterations", type=int, default=10**6)
    ap.add_argument("--warmup-iters", type=int, default=4)
    ap.add_argument("--p-restart", type=float, default=0.3)
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 4)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--replay-ratio", type=float, default=4.0)
    ap.add_argument("--buffer", type=int, default=150_000)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--time-budget-hours", type=float, default=11.0)
    ap.add_argument("--max-moves-override", type=int, default=0, help="debug only: cap moves per level")
    ap.add_argument("--resume", default=None)
    ap.add_argument("--out", default="runs/alphacandy_zero")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    levels = parse_levels(args.levels)
    focus = load_focus(args.focus_file)
    plr = PrioritizedLevelReplay(levels, focus, args.focus_frac)

    net = AlphaCandyZeroNet(args.preset).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-4)
    start_iter = 0
    if args.resume and os.path.exists(args.resume):
        ck = torch.load(args.resume, map_location=device, weights_only=False)
        net.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        start_iter = ck.get("iteration", 0) + 1
        for k, v in ck.get("plr", {}).get("p_win", {}).items():
            if int(k) in plr.p_win:
                plr.p_win[int(k)] = v
        print(f"[resume] from {args.resume} at iteration {start_iter}")
    print(f"[AlphaCandy-Zero] preset={args.preset} params={count_params(net):,} device={device} "
          f"levels={len(levels)} focus={len(plr.focus)} workers={args.workers}")

    weights_path = os.path.join(args.out, "actor_weights.pt")
    buffer: deque = deque(maxlen=args.buffer)
    log_path = os.path.join(args.out, "metrics.jsonl")
    t_start = time.time()
    budget = args.time_budget_hours * 3600

    ctx = get_context("spawn")
    with ctx.Pool(args.workers, initializer=_worker_init, initargs=(args.preset, args.max_moves_override)) as pool:
        for it in range(start_iter, start_iter + args.iterations):
            if time.time() - t_start > budget:
                print("[budget] time budget reached, stopping")
                break
            t_it = time.time()
            atomic_save({k: v.detach().cpu() for k, v in net.state_dict().items()}, weights_path)
            lam = min(1.0, it / max(1, args.warmup_iters))
            search = dict(num_simulations=args.sims, max_considered=args.considered,
                          net_weight=lam, leaf_rollout=args.leaf_rollout)
            tasks = []
            for lvl in plr.sample(args.games_per_iter):
                start = None
                snaps = plr.snapshots.get(lvl)
                if snaps and random.random() < args.p_restart:
                    start = random.choice(list(snaps))
                tasks.append({"level": lvl, "seed": random.randrange(1, 2**31 - 1), "version": it,
                              "weights": weights_path, "search": search, "start": start})
            results = pool.map(_worker_play, tasks, chunksize=1)

            wins = full = restarts = rwins = errors = 0
            new = 0
            for r in results:
                if "error" in r:
                    errors += 1
                    continue
                plr.update(r["level"], r["won"], r["restart"])
                if r["restart"]:
                    restarts += 1
                    rwins += int(r["won"])
                else:
                    full += 1
                    wins += int(r["won"])
                    plr.add_snapshots(r["level"], r["snapshots"], r["final_remaining"], r["init_obj"])
                buffer.extend(r["samples"])
                new += len(r["samples"])

            steps = max(1, int(math.ceil(new * args.replay_ratio / args.batch_size)))
            losses = train_steps(net, opt, buffer, steps, args.batch_size, device)

            seen = [p for p in plr.p_win.values() if p is not None]
            rec = {
                "iter": it, "lambda": lam, "games": len(results), "full_games": full,
                "full_win_rate": wins / max(1, full), "restart_games": restarts,
                "restart_win_rate": rwins / max(1, restarts), "errors": errors,
                "samples": new, "buffer": len(buffer), "train_steps": steps,
                "levels_seen": len(seen), "mean_p_win_seen": float(np.mean(seen)) if seen else 0.0,
                "sec": round(time.time() - t_it, 1), "elapsed_h": round((time.time() - t_start) / 3600, 3),
                **{f"loss_{k}": round(v, 4) for k, v in losses.items()},
            }
            print(json.dumps(rec), flush=True)
            with open(log_path, "a") as f:
                f.write(json.dumps(rec) + "\n")
            atomic_save({"model": net.state_dict(), "opt": opt.state_dict(), "iteration": it,
                         "preset": args.preset, "plr": plr.state_dict()},
                        os.path.join(args.out, "checkpoint.pt"))
    print("[done]")


if __name__ == "__main__":
    main()
