import os
import sys
import shutil

print("=================================================================")
print("  STARTING KAGGLE GPU TRAINER FOR CANDY CRUSH SUPERHUMAN AI      ")
print("=================================================================")

input_dir = "/kaggle/input"
ds_dir = None

if os.path.exists(input_dir):
    subdirs = os.listdir(input_dir)
    print(f"Available input datasets in {input_dir}: {subdirs}")
    for d in subdirs:
        if "candy" in d.lower():
            ds_dir = os.path.join(input_dir, d)
            break
    if ds_dir is None and len(subdirs) > 0:
        ds_dir = os.path.join(input_dir, subdirs[0])

work_dir = "/kaggle/working/candy_crush"

if ds_dir and os.path.exists(ds_dir):
    print(f"Mounting files from {ds_dir} to {work_dir}...")
    shutil.copytree(ds_dir, work_dir, dirs_exist_ok=True)
else:
    print(f"Dataset directory not found, using current directory.")
    work_dir = "/kaggle/working"

os.chdir(work_dir)
sys.path.insert(0, work_dir)

print(f"Working directory: {os.getcwd()}")
print(f"Directory contents: {os.listdir(work_dir)}")

import main_gpu
main_gpu.main()
