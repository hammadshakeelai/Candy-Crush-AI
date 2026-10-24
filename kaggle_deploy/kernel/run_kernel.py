import os
import sys

print("=================================================================")
print("  STARTING KAGGLE GPU TRAINER FOR CANDY CRUSH SUPERHUMAN AI      ")
print("=================================================================")

src_root = None
config_path = None

for root, dirs, files in os.walk("/kaggle/input"):
    if "main_gpu.py" in files and src_root is None:
        src_root = root
    if "configs_v1.208.json" in files and config_path is None:
        config_path = os.path.join(root, "configs_v1.208.json")

print(f"Detected Source root: {src_root}")
print(f"Detected Config path: {config_path}")

if config_path:
    os.environ["KING_CONFIG_PATH"] = config_path

work_dir = "/kaggle/working"
os.chdir(work_dir)
sys.path.insert(0, work_dir)

if src_root:
    sys.path.append(src_root)

    # 1. Ensure train_grandmaster_net.py has typing imports
    tg_path = os.path.join(src_root, "train_grandmaster_net.py")
    if os.path.exists(tg_path):
        with open(tg_path, "r", encoding="utf-8") as f:
            content = f.read()
        if "from typing import" not in content or "List" not in content or "checkpoint_out" not in content:
            print("[INFO] Hot-patching train_grandmaster_net.py...")
            patched = "from typing import List, Tuple, Dict, Optional, Any\n" + content
            patched = patched.replace(
                "output_model: str = \"checkpoints/candy_crush_grandmaster_v2.pt\"",
                "checkpoint_out: Optional[str] = None, output_model: Optional[str] = None"
            ).replace(
                "torch.save(model.state_dict(), output_model)",
                "save_path = checkpoint_out or output_model or 'checkpoints/candy_crush_grandmaster_v2.pt'\n    torch.save(model.state_dict(), save_path)"
            )
            with open(os.path.join(work_dir, "train_grandmaster_net.py"), "w", encoding="utf-8") as f:
                f.write(patched)

    # 2. Ensure train_grpo.py has typing imports
    grpo_path = os.path.join(src_root, "train_grpo.py")
    if os.path.exists(grpo_path):
        with open(grpo_path, "r", encoding="utf-8") as f:
            grpo_content = f.read()
        if "Optional" not in grpo_content.split("from typing import")[1].split("\n")[0] if "from typing import" in grpo_content else True:
            print("[INFO] Hot-patching train_grpo.py with Optional...")
            patched_grpo = grpo_content.replace(
                "from typing import List, Tuple, Dict, Any\n",
                "from typing import List, Tuple, Dict, Any, Optional\n"
            )
            if "from typing import" not in patched_grpo:
                patched_grpo = "from typing import List, Tuple, Dict, Any, Optional\n" + patched_grpo
            with open(os.path.join(work_dir, "train_grpo.py"), "w", encoding="utf-8") as f:
                f.write(patched_grpo)

    import main_gpu
    main_gpu.main()
else:
    print("FATAL: Source root not found under /kaggle/input!")
    for root, dirs, files in os.walk("/kaggle/input"):
        print(root, files)
