"""
Honest multi-seed evaluation protocol & warm-start checkpoint bridge for AlphaCandy-Zero.

Why this module exists (see alphacandy_zero/__init__.py):
  1. Honest evaluation without seed-hunting inflation:
     - Playing a level across 80 random seeds and stopping at the first win measures pass@80,
       NOT the agent's true single-attempt win rate.
     - `evaluate_levels` fixes K deterministic seeds per level (e.g. K=5) and reports BOTH:
         * `mean_single_seed_win_rate` (true 1-attempt expected win rate across all K seeds)
         * `pass_at_k` (fraction of levels solved on at least 1 of the K seeds)
         * `mean_objectives_cleared_frac` (partial progress on unsolved levels)
  2. Warm-start bridge from legacy 20-channel checkpoints (`LegacyMasterEvaluator`):
     - Allows `GumbelSearcher` to use our pre-trained `checkpoints/alphacandy_large_net_master.pt`
       prior immediately while `AlphaCandyZeroNet` is being trained.

Usage:
  python -m alphacandy_zero.evaluate --levels 235,272,279 --seeds 3 --sims 24 --preset cpu
"""

from __future__ import annotations

import argparse
import json
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from alphacandy_zero.encoding import NUM_ACTIONS, encode_action, legal_mask_from_moves
from alphacandy_zero.network import AlphaCandyZeroNet
from alphacandy_zero.search import Evaluator, GumbelSearcher, SearchConfig, play_game
from core.reverse_engine import Direction, ReverseCandyEngine, ReverseEngineState


class LegacyMasterEvaluator(Evaluator):
    """Adapts pre-trained 20-channel AlphaCandyLargeNet into GumbelSearcher's Evaluator API."""

    def __init__(self, checkpoint_path: str = "checkpoints/alphacandy_large_net_master.pt", device: str = "cpu"):
        super().__init__(net=None, device=device, flip_ensemble=False)
        from models.spatial_transformer_net import AlphaCandyLargeNet, encode_spatial_board_tensor

        self._encode20 = encode_spatial_board_tensor
        self.legacy_net = AlphaCandyLargeNet().to(device)
        ck = torch.load(checkpoint_path, map_location=device, weights_only=False)
        sd = ck.get("model_state_dict", ck.get("state_dict", ck))
        self.legacy_net.load_state_dict(sd, strict=False)
        self.legacy_net.eval()

    @torch.no_grad()
    def __call__(
        self,
        states: Sequence[ReverseEngineState],
        init_obj: int,
        legal_lists: Sequence[Optional[Sequence[Tuple[int, int, Direction]]]],
    ) -> Tuple[np.ndarray, np.ndarray]:
        n = len(states)
        if n == 0:
            return np.zeros((0, NUM_ACTIONS), np.float32), np.zeros(0, np.float32)
        t20 = np.stack([self._encode20(s) for s in states]).astype(np.float32)
        masks = np.zeros((n, NUM_ACTIONS), bool)
        for i, lg in enumerate(legal_lists):
            masks[i] = legal_mask_from_moves(lg) if lg else True
        x = torch.from_numpy(t20).to(self.device)
        logits, win = self.legacy_net(x)
        logits_np = np.where(masks, logits.float().cpu().numpy(), -1e9).astype(np.float32)
        win_np = win.squeeze(-1).float().cpu().numpy().astype(np.float32)
        return logits_np, win_np


def evaluate_levels(
    levels: Sequence[int],
    evaluator: Evaluator,
    search_cfg: SearchConfig,
    seeds_per_level: int = 3,
    base_seed: int = 1000,
) -> Dict[str, Any]:
    from core.king_level_loader import KingLevelLoader

    loader = KingLevelLoader()
    engine = ReverseCandyEngine()
    searcher = GumbelSearcher(engine, evaluator, search_cfg)

    per_level: List[Dict[str, Any]] = []
    total_games = 0
    total_wins = 0
    levels_passed = 0
    t0 = time.time()

    for lvl in levels:
        cfg = loader.load_level(lvl)
        lvl_wins = 0
        rem_fracs = []
        winning_seed = None
        best_moves = None

        for k in range(seeds_per_level):
            seed = base_seed + lvl * 101 + k * 37
            out = play_game(engine, searcher, cfg, seed=seed, record=False, add_noise=False)
            total_games += 1
            if out["won"]:
                lvl_wins += 1
                total_wins += 1
                if winning_seed is None:
                    winning_seed = seed
                    best_moves = out["moves_used"]
            rem_fracs.append(1.0 - min(1.0, out["final_remaining"] / max(1, out["init_obj"])))

        if lvl_wins > 0:
            levels_passed += 1
        per_level.append({
            "level_id": lvl,
            "level_type": cfg.level_type.name,
            "moves_limit": cfg.moves_limit,
            "seeds_tested": seeds_per_level,
            "wins": lvl_wins,
            "single_seed_win_rate": round(lvl_wins / max(1, seeds_per_level), 4),
            "pass_at_k": lvl_wins > 0,
            "mean_objective_clearance": round(float(np.mean(rem_fracs)), 4),
            "winning_seed": winning_seed,
            "moves_used_on_win": best_moves,
        })

    return {
        "levels_count": len(levels),
        "seeds_per_level": seeds_per_level,
        "num_simulations": search_cfg.num_simulations,
        "mean_single_seed_win_rate": round(total_wins / max(1, total_games), 4),
        "pass_at_k_rate": round(levels_passed / max(1, len(levels)), 4),
        "levels_passed": levels_passed,
        "elapsed_sec": round(time.time() - t0, 2),
        "per_level": per_level,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--levels", default="1,25,50")
    ap.add_argument("--preset", default="cpu")
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--use-legacy-master", action="store_true")
    ap.add_argument("--sims", type=int, default=16)
    ap.add_argument("--considered", type=int, default=6)
    ap.add_argument("--net-weight", type=float, default=0.3)
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--flip-ensemble", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    from alphacandy_zero.train import parse_levels

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.use_legacy_master:
        ev: Evaluator = LegacyMasterEvaluator(device=device)
    else:
        net = AlphaCandyZeroNet(args.preset).to(device)
        if args.checkpoint:
            ck = torch.load(args.checkpoint, map_location=device, weights_only=False)
            net.load_state_dict(ck.get("model", ck))
        ev = Evaluator(net, device=device, flip_ensemble=args.flip_ensemble)

    scfg = SearchConfig(
        num_simulations=args.sims,
        max_considered=args.considered,
        net_weight=args.net_weight if (args.checkpoint or args.use_legacy_master) else 0.0,
    )
    report = evaluate_levels(parse_levels(args.levels), ev, scfg, seeds_per_level=args.seeds)
    print(json.dumps(report, indent=2))
    if args.out:
        with open(args.out, "w") as f:
            json.dump(report, f, indent=2)


if __name__ == "__main__":
    main()
