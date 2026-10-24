import os
import sys
import shutil

print("=================================================================")
print("  STARTING KAGGLE GPU TRAINER FOR CANDY CRUSH SUPERHUMAN AI      ")
print("=================================================================")

ds_dir = "/kaggle/input/candy-crush-grandmaster-rl"
work_dir = "/kaggle/working/candy_crush"

print(f"Mounting files from {ds_dir} to {work_dir}...")
shutil.copytree(ds_dir, work_dir, dirs_exist_ok=True)

os.chdir(work_dir)
sys.path.insert(0, work_dir)

import main_gpu
main_gpu.main()
