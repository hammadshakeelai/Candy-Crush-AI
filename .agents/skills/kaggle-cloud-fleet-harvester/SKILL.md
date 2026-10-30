---
name: kaggle-cloud-fleet-harvester
description: >-
  Manage, monitor, and harvest telemetry from Kaggle 4-vCPU and Dual-T4 GPU
  cloud kernels in under 2 seconds using `kagglesdk` without downloading
  bulky `.pt` checkpoints.
---

# Kaggle Cloud Fleet Harvester Skill

Use this skill to check status and harvest `.json`, `.md`, and `.log` outputs from active Kaggle kernels in `<2 seconds`.

## 1. Check Kernel Status

```bash
/home/hammad/Antigravity_Workspace/Candy-Crush-AI/venv/bin/kaggle kernels status hammadshakeelai/<kernel-slug>
```

## 2. Fast `<2s` Telemetry Harvester (Skips Bulky `.pt` Checkpoints)

Never run `kaggle kernels output` directly when `/kaggle/working/checkpoints` contains large `.pt` files. Instead, filter for `.json`, `.md`, and `.log` via `kagglesdk`:

```bash
/home/hammad/Antigravity_Workspace/Candy-Crush-AI/venv/bin/python3 -c '
import os, urllib.request
from kagglesdk import KaggleClient
from kagglesdk.kernels.types.kernels_api_service import ApiListKernelSessionOutputRequest

client = KaggleClient()
req = ApiListKernelSessionOutputRequest()
req.user_name = "hammadshakeelai"
req.kernel_slug = "alphacandy-mastery-residual-300-perfect"
res = client.kernels.kernels_api_client.list_kernel_session_output(req)
for f in (res.files or []):
    if any(f.file_name.endswith(ext) for ext in (".json", ".md", ".log")):
        urllib.request.urlretrieve(f.url, os.path.basename(f.file_name))
'
```
