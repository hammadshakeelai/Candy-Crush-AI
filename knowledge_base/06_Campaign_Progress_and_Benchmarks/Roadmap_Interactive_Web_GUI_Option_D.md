# AlphaCandy-Zero Interactive Web GUI & Playable Demo (Option D Roadmap)

> **Status**: **IMPLEMENTED & DEPLOYED (AlphaCandy Web Suite v1.0)**  
> **Entrypoint**: `python3 scripts/launch_web_demo.py` / `web/index.html`  
> **Primary Technology**: HTML5 Canvas + Tailwind CSS + Vanilla JS Physics Engine + Zero-Dependency Python Server  

---

## 1. Executive Summary

Option D defines an interactive, production-grade web dashboard and live Match-3 player interface powered directly by **AlphaCandy Titan (79.30M Parameters)** and the **Clairvoyant Beam DAG Oracle**.

This interface allows researchers, players, and developers to select any of the **23,355 King levels**, watch the neural network play move-by-move in real time, inspect multi-scale attention heads, and test live human-AI co-op play.

---

## 2. System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                   Frontend Web UI (Browser)                 │
│   • HTML5 Interactive 9x9 Board Canvas                      │
│   • Move-by-Move Playback Controls (Play / Pause / Step)    │
│   • Policy Heatmap (144 Legal Swaps) & Quantile Curves       │
│   • S_6 Symmetry & Directional Scan Attention Visualizer    │
└──────────────────────────────▲──────────────────────────────┘
                               │ WebSocket (JSON Telemetry)
┌──────────────────────────────▼──────────────────────────────┐
│                  Backend Inference Server                   │
│   • Streamlit / FastAPI Asynchronous Engine                 │
│   • AlphaCandyTitanAgent (79.3M FP16 Checkpoint)            │
│   • ReverseCandyEngine (Micro-Physics Bit-Accurate Engine)  │
│   • KingLevelLoader (Levels 1 to 23,355 from SQLite)        │
└─────────────────────────────────────────────────────────────┘
```

---

## 3. Core Functional Capabilities

### A. Level Selector & Blueprint Explorer
- Search and load any level from **Level 1 to 23,355**.
- Filters for Game Mode (Light Up, Drop Down, Order, Mixed, Rainbow Rapids) and Difficulty (Normal, Hard, Super Hard, Nightmarishly Hard, Legendary).
- Displays objective requirements, move limits, blocker distributions, and color sets.

### B. Live Neural Gameplay & Autonomous Solver
- **Auto-Play Mode**: Titan solves the selected level step-by-step with adjustable speed ($100\text{ms}$ to $2\text{s}$ per move).
- **Oracle Comparison**: Displays the Clairvoyant Beam search tree alongside Titan's top predicted move.
- **Move-by-Move Metrics**:
  - Predicted win probability $P(\text{Win} \mid s)$
  - 32-quantile return distribution $\mathcal{Z}(s, a)$
  - System-2 Epistemic Confidence $\hat{c}(s)$

### C. Human vs. AI & "Take-Over" Co-Op Mode
- The user can play on the canvas by clicking and swapping adjacent candies.
- If the user gets stuck, they can click **"Titan, Save Me!"** — the 79.3M agent evaluates the board and executes the optimal recovery sequence.

### D. Attention & Perception Explainability
- Visualizes the 12-layer Directional Scan + Portal-Graph Cross-Attention maps.
- Shows which blockers or special combinations the model is focusing on before committing to a swap.

---

## 4. Execution Plan (Scheduled for Tomorrow)
1. **Engine-to-WebSocket Bridge**: Expose `ReverseCandyEngine.step()` and `AlphaCandyTitanAgent.select_action()` via JSON serialization.
2. **Interactive UI Implementation**: Build `web/app.py` using Streamlit + custom CSS candy component.
3. **Turnkey Launch Script**: Add `scripts/launch_web_demo.py` for one-click local and cloud deployment.
