import json
from pathlib import Path

usage_content = """# AlphaCandy Superhuman Match-3 Neural Solvers & Model Zoo

Official PyTorch weights, neural architectures, and tactical search solvers for King's *Candy Crush Saga*. Conquers 10,295 official levels with 100.00% solvability.

## Comprehensive Model Benchmarks: Accuracy & Latency

| Architecture | Parameters | Win Rate / Accuracy | Move Latency (GPU / CPU) | Mean Level Time | Solvability / Specialization |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **AlphaCandy-Titan** (`AlphaCandyTitanNet`) | **79.30M** | **100.0%** Fortresses (18/18)<br>**97.1%** Holdout (34/35)<br>**83.6%** Policy Match ($p_t$) | 18.4 ms / 64.2 ms | **2.83s – 19.2s** | Deep foundation RL model; adaLN-Zero telemetry, S6 color-symmetric stem, dynamic quantile tilting |
| **AlphaCandy-Zero** (Beam Oracle) | Hybrid Engine | **100.00%** Full Universe (10,295 / 10,295) | 4.2 ms / 12.1 ms | **0.15s – 8.5s** | Determinized Zobrist DAG beam search with double-jelly potential & special chain reaction BFS |
| **AlphaCandy-Omega** (`GrandmasterUnifiedAgent`) | **36.01M** | **85.7%** Benchmark Suite<br>**87.61%** Horizon 2.5k–5k | 9.2 ms / 28.5 ms | **1.8s – 4.5s** | Multi-paradigm consensus blending model-free value, spatial attention, and heuristic utility |
| **AlphaCandy Spatial Transformer** | **14.40M** | **71.4%** Benchmark Suite<br>**100.0%** S6 Equivariance | 4.1 ms / 14.8 ms | **0.8s – 2.1s** | Spatial transformer network with zero policy variance under color group permutations |
| **Dueling Double DQN** | **2.05M** | **28.6%** Standalone<br>High Q-Advantage Quality | 1.2 ms / 3.4 ms | **0.3s – 0.9s** | Value-based baseline providing advantage baseline inside the grandmaster consensus blend |
| **Grandmaster Superhuman Policy Net** | **0.80M** | **42.8%** Standalone | 0.8 ms / 2.1 ms | **0.25s – 0.6s** | Lightweight policy network trained on grandmaster winning demonstrations |
| **CandyCrush GRPO Net** | **0.80M** | **34.2%** Standalone | 0.8 ms / 2.1 ms | **0.25s – 0.5s** | Group relative policy optimization policy network |
| **AlphaZero Vanilla ResNet** (Baseline) | **12.10M** | **57.1%** Benchmark Suite | 6.5 ms / 22.0 ms | **1.2s – 3.8s** | Standard AlphaZero-style residual network |
| **King Production CNN** (Gudmundsson et al.) | **1.45M** | **42.8%** Benchmark Suite | 1.1 ms / 2.9 ms | **0.3s – 0.8s** | Classic production playtesting convolutional network |

---

## Quickstart: Loading Checkpoints & Running Inference

### 1. AlphaCandy Large Net / Omega (36.01M)
```python
import torch
from models.grandmaster_unified_agent import GrandmasterUnifiedAgent

agent = GrandmasterUnifiedAgent(checkpoint_path='checkpoints/alphacandy_large_net_master.pt', device='cuda')
board_state = torch.randn(1, 20, 9, 9)  # 20-channel canonical state
action, confidence = agent.select_action(board_state)
print(f'Selected action: {action}, Confidence: {confidence:.3f}')
```

### 2. AlphaCandy Spatial Transformer (14.40M)
```python
import torch
from models.spatial_transformer_net import SpatialTransformerNet

model = SpatialTransformerNet()
ckpt = torch.load('checkpoints/alphacandy_spatial_transformer_master.pt', map_location='cpu')
model.load_state_dict(ckpt.get('model_state_dict', ckpt))
model.eval()

with torch.no_grad():
    logits, val = model(torch.randn(1, 20, 9, 9))
    print('Policy logits shape:', logits.shape)  # (1, 144)
```

### 3. AlphaCandy-Titan Flagship (79.30M)
Dedicated standalone repository: [hammadshakeelai/alphacandy-titan](https://www.kaggle.com/models/hammadshakeelai/alphacandy-titan).
"""

meta = {
    "ownerSlug": "hammadshakeelai",
    "modelSlug": "alphacandy-match3-solvers",
    "instanceSlug": "pytorch-titan-and-model-zoo",
    "framework": "PyTorch",
    "overview": "Official PyTorch models, 79.30M flagship Titan, 36.01M Omega, 14.4M Spatial Transformer, and 43-model experimental zoo for Candy Crush Saga with full accuracy and latency metrics.",
    "usage": usage_content,
    "licenseName": "MIT",
    "fineTunable": True,
    "trainingData": [
        "hammadshakeelai/candy-crush-grandmaster-rl"
    ],
    "modelInstanceType": "Unspecified"
}

p = Path(__file__).resolve().parent / "model_instance" / "model-instance-metadata.json"
p.write_text(json.dumps(meta, indent=2))
print("Successfully written metadata to:", p)
