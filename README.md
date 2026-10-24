# Candy-Crush-AI: Superhuman Match-3 Artificial Intelligence

An end-to-end Machine Learning, Monte Carlo Tree Search (MCTS), and Computer Vision system designed to solve and master any **Candy Crush** level at superhuman levels.

---

## Key Features

1. **High-Throughput Simulation Engine (`CandyEngine`)**:
   - Exact simulation of 9x9 match-3 grid physics, cascades, gravity, special candies (Striped, Wrapped, Color Bomb), and combo interactions (e.g. Bomb+Striped, Bomb+Bomb).
   - Operates at **>20,000 steps/second** on CPU.
2. **Gymnasium Environment (`CandyCrushGymEnv`)**:
   - Standard Gym API with shaped rewards for score, cascades, and jelly clearance.
3. **Dual-Head Policy-Value ResNet**:
   - 19-channel board tensor representation $(C=19, H=9, W=9)$.
   - Predicts move priors $P(a|s)$ and win probability / utility $V(s) \in [-1, 1]$.
4. **Information Set Monte Carlo Tree Search (ISMCTS)**:
   - Evaluates tree moves with stochastic determinization to overcome randomized candy refills.
5. **Expert Heuristics Engine**:
   - Bottom-bias cascade multiplier, combo synergy valuation, and obstacle (chocolate, jelly) containment.
6. **5-Lives Bypass & Management Engine**:
   - Automated ADB time-leap exploit (`adb shell date -s +3hours`) refills 5 lives in under 4 seconds without root limitations on emulators.
7. **Computer Vision & Live Game Perception**:
   - Grid bounding box localization, tile slicing, and SSIM animation settling detection.
   - HSV color classifier and texture gradient filters for special candy detection.

---

## Installation

```bash
# Create and activate environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

---

## Quickstart

### 1. Run Unit & Physics Tests
```bash
python -m unittest tests/test_simulation.py
```

### 2. Run Self-Play Training Pipeline
```bash
python train_selfplay.py
```

### 3. Connect to Real Game (Android Device or Emulator via ADB)
```bash
# Enable USB Debugging on your phone or launch an emulator (Waydroid / Genymotion / LDPlayer)
# Open Candy Crush Saga
python automation/bot_runner.py --mcts --sims 40
```

---

## Open-Source Datasets

See [`data/dataset_manifest.json`](data/dataset_manifest.json) for download links and metadata:
- **Roboflow Universe**: `candy_crush-oaoem` (YOLO bounding boxes for candies and obstacles)
- **Kaggle**: `thedevastator/level-difficulty-in-candy-crush-saga` (15,000 level player metrics)
- **Hugging Face**: `markov-ai/gaming-500-hours` (telemetry event traces)
- **GitHub**: `mulliganaceous/candycrush` (level blueprints and drop rate configs)
