---
name: alphacandy-zero-oracle
description: >-
  Run the AlphaCandy-Zero Clairvoyant Solvability Oracle (`alphacandy_zero.oracle`)
  to audit King level blueprints for topological traps (isolated 1x1 island cells,
  void objectives, chocolate spawners), prove level solvability via determinized
  width-B Zobrist beam search, and harvest golden winning trajectories.
---

# AlphaCandy-Zero Clairvoyant Oracle Skill

Use this skill whenever an agent fails repeatedly on specific King levels (e.g., `[235, 272, 279]` or any level in `campaign/unsolved_levels_1_2500.json`) to separate **engine/level-loader bugs** from **policy/search weakness**.

## 1. Topological Level Audit & Solvability Proof

Run the oracle on target levels using the workspace virtual environment:

```bash
PYTHONPATH=. /home/hammad/Antigravity_Workspace/Candy-Crush-AI/venv/bin/python3 -m alphacandy_zero.oracle \
  --levels 235,272,279 \
  --beam 16 \
  --branch 10 \
  --seeds 6 \
  --out runs/oracle_golden.pt
```

## 2. Interpreting `diagnosis` Telemetry

- `void_objectives > 0`: Indicates a level-loader bug in [`core/king_level_loader.py`](file:///home/hammad/Antigravity_Workspace/Candy-Crush-AI/core/king_level_loader.py) where jellies or ingredients were placed on inactive cutout cells (`active_grid == False`).
- `isolated_objective_cells`: Lists $1\times 1$ island cells with zero orthogonal active neighbors (such as Level 279's `[(8, 0), (8, 4), (8, 8)]`). Normal swaps cannot touch these cells; they require remote `STRIPED_VERTICAL` / `STRIPED_HORIZONTAL` lasers, `COLOR_BOMB`, or `JELLY_FISH`.
- `required_pace >= 4.0`: High-density blocker/jelly fortress (such as Level 235's `5.06` jellies/move) requiring consecutive `COLOR_BOMB` + `STRIPED`/`WRAPPED` combinations.

## 3. Bootstrapping `AlphaCandyZeroNet` with Oracle Trajectories

The `--out runs/oracle_golden.pt` file saves transitions in the exact replay-buffer dictionary format (`planes, globals, mask, pi, z, fbin, jmask, jtgt`) used by [`alphacandy_zero/train.py`](file:///home/hammad/Antigravity_Workspace/Candy-Crush-AI/alphacandy_zero/train.py).
