"""
Standalone runner script for fine-tuning experiment (System C or System B).
Usage: python -m scripts.run_training --system C (or B)
"""

import argparse
import json
import torch
from pathlib import Path

from src.data.vizwiz_dataset import load_annotations
from src.data.dataset_cache import DatasetCacheManager
from src.models.finetune_engine import run_finetuning_experiment

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--system", type=str, required=True, choices=["B", "C"], help="System B or C")
    args = parser.parse_args()

    system_name = args.system.upper()
    print(f"Initializing run for System {system_name}...", flush=True)

    with open("results/train_split_images.json", "r") as f:
        train_fns = set(json.load(f))
    with open("results/val_split_images.json", "r") as f:
        val_fns = set(json.load(f))

    all_train = [s for s in load_annotations("data/raw/Annotations/Annotations/train.json") if s.image_filename in train_fns]
    all_val = [s for s in load_annotations("data/raw/Annotations/Annotations/val.json") if s.image_filename in val_fns]

    train_cache = DatasetCacheManager("data/cache/train_images")
    val_cache = DatasetCacheManager("data/cache/val_images")

    config = {
        "seed": 42,
        "mode": "class_agnostic",
        "backbone": "mobilenet_v3",
        "num_classes": 2,
        "min_size": 512,
        "max_size": 640,
        "batch_size": 2,
        "grad_accum_steps": 4,
        "total_epochs": 30,
        "stage1_epochs": 5,
        "val_every": 2,
        "stage1_lr": 1e-3,
        "stage2_heads_lr": 2.5e-4,
        "stage2_backbone_lr": 2e-5,
        "weight_decay": 1e-4,
        "early_stopping_patience": 5  # 5 validation cycles = 10 epochs
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    results = run_finetuning_experiment(
        system_name=system_name,
        train_samples=all_train,
        val_samples=all_val,
        train_cache=train_cache,
        val_cache=val_cache,
        config=config,
        device=device,
        results_dir="results",
        checkpoint_dir="models/checkpoints",
        notebook_path="notebooks/privacylens_training.ipynb"
    )

    print(f"\nExperiment System {system_name} Complete!", flush=True)
    print(f"Best Val mAP@0.5: {results['best_val_map_50']:.4f} at Epoch {results['best_epoch']}", flush=True)

if __name__ == "__main__":
    main()
