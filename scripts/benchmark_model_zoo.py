#!/usr/bin/env python3
"""
Comprehensive Benchmark & Profiler for AlphaCandy 42-Model Zoo
==============================================================
Profiles parameter count, forward inference latency, output shapes,
and architectural family for all 42 models in MODEL_ZOO_REGISTRY.
"""

import time
import sys
from pathlib import Path
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from models.zoo import MODEL_ZOO_REGISTRY, load_zoo_model_class, create_zoo_model

def profile_model_zoo():
    print("=" * 80)
    print("🍬 ALPHACANDY 42-MODEL ZOO COMPREHENSIVE BENCHMARK & AUDIT")
    print("=" * 80)

    device = torch.device("cpu")
    dummy_x = torch.randn(1, 20, 9, 9, device=device)
    dummy_telemetry = torch.zeros(1, 16, device=device)

    results = []
    total_params_all = 0

    for idx, (name, class_path) in enumerate(MODEL_ZOO_REGISTRY.items(), 1):
        try:
            cls = load_zoo_model_class(name)
            
            # Instantiate model
            try:
                model = cls()
            except TypeError:
                try:
                    model = cls(in_channels=20)
                except TypeError:
                    model = cls(20)

            model.to(device)
            model.eval()

            # Count parameters
            params = sum(p.numel() for p in model.parameters())
            total_params_all += params

            # Warmup
            with torch.no_grad():
                try:
                    _ = model(dummy_x)
                except TypeError:
                    _ = model(dummy_x, dummy_telemetry)
                except Exception:
                    pass

            # Timed inference (5 runs)
            latencies = []
            with torch.no_grad():
                for _ in range(5):
                    t0 = time.perf_counter()
                    try:
                        out = model(dummy_x)
                    except TypeError:
                        out = model(dummy_x, dummy_telemetry)
                    dt = (time.perf_counter() - t0) * 1000.0  # ms
                    latencies.append(dt)

            med_latency = sorted(latencies)[len(latencies) // 2]
            
            out_summary = "dict" if isinstance(out, dict) else (tuple(out.shape) if hasattr(out, "shape") else type(out).__name__)

            results.append({
                "idx": idx,
                "name": name,
                "module": class_path.split(".")[-2],
                "params": params,
                "latency_ms": med_latency,
                "output": str(out_summary),
                "status": "PASS"
            })
            print(f"[{idx:02d}/42] {name:<36} | {params:>10,} params | {med_latency:>6.2f} ms | PASS")
        except Exception as e:
            results.append({
                "idx": idx,
                "name": name,
                "module": class_path.split(".")[-2] if "." in class_path else "",
                "params": 0,
                "latency_ms": 0.0,
                "output": "ERROR",
                "status": f"FAIL: {str(e)[:40]}"
            })
            print(f"[{idx:02d}/42] {name:<36} | FAILED: {e}")

    print("=" * 80)
    print(f"Total Architectures: {len(results)} | Total Trainable Parameters: {total_params_all:,}")
    print("=" * 80)
    return results, total_params_all

if __name__ == "__main__":
    profile_model_zoo()
