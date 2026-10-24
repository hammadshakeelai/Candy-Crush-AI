"""
Automated Downloader and Preprocessing Script for Open-Source Candy Crush Datasets.
Supports Roboflow Universe YOLO exports, Kaggle telemetry datasets, and Hugging Face traces.
"""

import os
import json
import argparse


def download_roboflow_dataset(api_key: str, workspace: str = "demo-uahd9", project: str = "candy_crush-oaoem", version: int = 1):
    """
    Downloads Candy Crush object detection dataset from Roboflow Universe.
    """
    try:
        from roboflow import Roboflow
        print(f"Connecting to Roboflow: {workspace}/{project} v{version}...")
        rf = Roboflow(api_key=api_key)
        proj = rf.workspace(workspace).project(project)
        dataset = proj.version(version).download("yolov8")
        print(f"Dataset successfully downloaded to: {dataset.location}")
    except ImportError:
        print("Please install roboflow: pip install roboflow")
    except Exception as e:
        print(f"Download failed: {e}")


def download_kaggle_difficulty_dataset():
    """
    Downloads Candy Crush level difficulty & player progression dataset from Kaggle.
    """
    try:
        import kaggle
        print("Downloading Kaggle dataset: thedevastator/level-difficulty-in-candy-crush-saga...")
        os.makedirs("data/kaggle", exist_ok=True)
        kaggle.api.dataset_download_files(
            "thedevastator/level-difficulty-in-candy-crush-saga",
            path="data/kaggle",
            unzip=True
        )
        print("Kaggle dataset downloaded to data/kaggle/")
    except ImportError:
        print("Please install kaggle (pip install kaggle) and ensure ~/.kaggle/kaggle.json exists.")
    except Exception as e:
        print(f"Kaggle download failed: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download Candy Crush Datasets")
    parser.add_argument("--roboflow_key", type=str, default=None, help="Roboflow API key")
    parser.add_argument("--kaggle", action="store_true", help="Download Kaggle dataset")
    args = parser.parse_args()

    if args.roboflow_key:
        download_roboflow_dataset(args.roboflow_key)
    if args.kaggle:
        download_kaggle_difficulty_dataset()
    if not args.roboflow_key and not args.kaggle:
        print("Please provide --roboflow_key <KEY> or --kaggle. See data/dataset_manifest.json for dataset links.")
