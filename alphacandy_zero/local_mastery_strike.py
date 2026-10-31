#!/usr/bin/env python3
"""
Local Mastery Strike Solver
===========================
Executes multi-seed Clairvoyant Zobrist Beam Search across up to 10 local CPU cores
targeting all remaining unsolved fortresses in Levels 1–2,500.

Features:
- Multi-layer isolated island & 2x2 non-matchable box health tracking
- Preserved Color Bomb & Preset Special candy physics
- Direct Striped Laser Snipe & Color Bomb + Striped / Jelly Fish combo hierarchy
- Automatic removal of conquered fortresses from campaign/unsolved_levels_1_2500.json
- Golden trajectory harvesting to runs/fortress_golden_trajectories.pt
"""

import sys
import time
import json
from pathlib import Path
import multiprocessing as mp
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.king_level_loader import KingLevelLoader
from alphacandy_zero.oracle import solve_with_oracle

UNSOLVED_PATH = Path("campaign/unsolved_levels_1_2500.json")
PROGRESS_PATH = Path("campaign/progress.json")
OUT_PATH = Path("runs/local_mastery_strike_results.json")
GOLDEN_PATH = Path("runs/fortress_golden_trajectories.pt")
TITAN_DEMO_PATH = Path("data/expert_demonstrations_omega.pt")

DEFAULT_TARGETS = [
    606, 1351, 1349, 975, 1205, 953, 656, 967, 1390, 1488,
    1794, 1958, 2007, 2035, 2245, 2273, 2358, 2359, 2406, 2411,
    2437, 2454, 2464, 2471, 2475, 2486,
]
BEAM_WIDTH = 24
BRANCH_PER_NODE = 10
MAX_SEEDS = 30


def _solve_single(lid: int):
    t0 = time.time()
    try:
        loader = KingLevelLoader()
        cfg = loader.load_level(lid)
        res = solve_with_oracle(
            cfg,
            beam_width=BEAM_WIDTH,
            branch_per_node=BRANCH_PER_NODE,
            max_seeds=MAX_SEEDS,
            base_seed=42,
        )
        elapsed = time.time() - t0
        return {
            "level_id": lid,
            "won": bool(res.get("won", False)),
            "moves_used": res.get("moves_used"),
            "moves_limit": res.get("moves_limit"),
            "remaining_objectives": res.get("objectives_remaining"),
            "initial_objectives": res.get("initial_objectives"),
            "winning_seed": res.get("winning_seed"),
            "elapsed_sec": round(elapsed, 2),
            "samples": res.get("samples", []) if res.get("won") else [],
            "titan_samples": res.get("titan_samples", []) if res.get("won") else [],
            "error": None,
        }
    except Exception as e:
        return {
            "level_id": lid,
            "won": False,
            "elapsed_sec": round(time.time() - t0, 2),
            "samples": [],
            "titan_samples": [],
            "error": str(e),
        }


def main():
    if UNSOLVED_PATH.exists():
        try:
            unsolved_data = json.loads(UNSOLVED_PATH.read_text())
            target_levels = unsolved_data.get("unsolved_levels", [])
        except Exception:
            target_levels = DEFAULT_TARGETS
    else:
        target_levels = DEFAULT_TARGETS

    # If all 26 primary fortresses in unsolved_levels_1_2500.json are conquered,
    # check campaign/progress.json for any unrecorded wins in 1..2500 or advance to 2501..5000
    if not target_levels and PROGRESS_PATH.exists():
        try:
            prog = json.loads(PROGRESS_PATH.read_text())
            cl = prog.get("completed_levels", {})
            unrecorded_1_2500 = sorted([int(k) for k, v in cl.items() if int(k) <= 2500 and not v.get("won")])
            if unrecorded_1_2500:
                target_levels = unrecorded_1_2500
            else:
                # Advance into Horizon 2501..5000 in continuous blocks of 100 levels
                solved_above_2500 = {int(k) for k, v in cl.items() if int(k) > 2500 and v.get("won")}
                target_levels = [lvl for lvl in range(2501, 5001) if lvl not in solved_above_2500][:100]
        except Exception:
            pass

    if not target_levels:
        print("🏆 [MasteryStrike] 100.0% of target horizons are already conquered!")
        return

    num_workers = min(10, mp.cpu_count())
    print(
        f"[MasteryStrike] Launching {num_workers}-core assault on {len(target_levels)} fortresses: {target_levels}",
        flush=True,
    )
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    results = []
    won_levels = []
    golden_samples = []
    if GOLDEN_PATH.exists():
        try:
            golden_samples = list(torch.load(GOLDEN_PATH, weights_only=False))
        except Exception:
            golden_samples = []

    with mp.Pool(processes=num_workers) as pool:
        for r in pool.imap_unordered(_solve_single, target_levels):
            samples = r.pop("samples", [])
            titan_samples = r.pop("titan_samples", [])
            results.append(r)
            lid = r["level_id"]
            if r["won"]:
                won_levels.append(lid)
                if samples:
                    golden_samples.extend(samples)
                    torch.save(golden_samples, GOLDEN_PATH)
                if titan_samples:
                    try:
                        import numpy as np
                        new_s = torch.tensor(np.stack([x["state"] for x in titan_samples], axis=0), dtype=torch.float32)
                        new_a = torch.tensor([x["action"] for x in titan_samples], dtype=torch.long)
                        new_w = torch.tensor([x["win_target"] for x in titan_samples], dtype=torch.float32)
                        new_m = torch.tensor([x["move_ratio"] for x in titan_samples], dtype=torch.float32)
                        if TITAN_DEMO_PATH.exists():
                            prev = torch.load(TITAN_DEMO_PATH, weights_only=False)
                            new_s = torch.cat([prev["states"], new_s], dim=0)
                            new_a = torch.cat([prev["actions"], new_a], dim=0)
                            new_w = torch.cat([prev["win_targets"], new_w], dim=0)
                            new_m = torch.cat([prev["move_ratios"], new_m], dim=0)
                        TITAN_DEMO_PATH.parent.mkdir(parents=True, exist_ok=True)
                        torch.save(
                            {"states": new_s, "actions": new_a, "win_targets": new_w, "move_ratios": new_m},
                            TITAN_DEMO_PATH,
                        )
                    except Exception:
                        pass
                # Update unsolved registry and master progress.json immediately
                if UNSOLVED_PATH.exists():
                    try:
                        curr = json.loads(UNSOLVED_PATH.read_text()).get("unsolved_levels", [])
                        updated = [x for x in curr if x != lid]
                        UNSOLVED_PATH.write_text(
                            json.dumps({"count": len(updated), "unsolved_levels": updated}, indent=2) + "\n"
                        )
                    except Exception:
                        pass
                if PROGRESS_PATH.exists():
                    try:
                        prog = json.loads(PROGRESS_PATH.read_text())
                        cl = prog.setdefault("completed_levels", {})
                        cl[str(lid)] = {
                            "won": True,
                            "moves": r["moves_used"],
                            "moves_limit": r["moves_limit"],
                            "winning_seed": r["winning_seed"],
                            "solver": "AlphaCandy-Zero Clairvoyant Zobrist Beam + Titan",
                        }
                        won_1_2500 = sum(1 for k, v in cl.items() if int(k) <= 2500 and v.get("won"))
                        prog["levels_1_2500_won"] = max(prog.get("levels_1_2500_won", 0), won_1_2500)
                        prog["levels_1_2500_unsolved"] = 2500 - prog["levels_1_2500_won"]
                        prog["levels_1_2500_win_rate_pct"] = round(100.0 * prog["levels_1_2500_won"] / 2500.0, 2)
                        prog["stats"] = {
                            "total_won": sum(1 for v in cl.values() if v.get("won")),
                            "total_tested": len(cl),
                            "win_rate": round(100.0 * sum(1 for v in cl.values() if v.get("won")) / max(1, len(cl)), 2),
                        }
                        PROGRESS_PATH.write_text(json.dumps(prog, indent=2))
                    except Exception:
                        pass
                print(
                    f"🎉 [WON] Level {lid} in {r['elapsed_sec']}s! Moves: {r['moves_used']}/{r['moves_limit']} (Remaining fortresses: {len(target_levels) - len(won_levels)})",
                    flush=True,
                )
            else:
                print(
                    f"❌ [LOSS] Level {lid} in {r['elapsed_sec']}s. Rem: {r.get('remaining_objectives')}/{r.get('initial_objectives')}",
                    flush=True,
                )

            with open(OUT_PATH, "w") as f:
                json.dump(
                    {
                        "total_targets": len(target_levels),
                        "evaluated_count": len(results),
                        "won_count": len(won_levels),
                        "won_levels": sorted(won_levels),
                        "results": results,
                    },
                    f,
                    indent=2,
                )

    print(
        f"\n[MasteryStrike] Assault complete! Conquered {len(won_levels)}/{len(target_levels)} fortresses: {sorted(won_levels)}",
        flush=True,
    )


if __name__ == "__main__":
    main()
