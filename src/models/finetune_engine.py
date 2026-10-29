import os
import sys
import time
import json
from pathlib import Path
import torch
import torchvision
import matplotlib.pyplot as plt
from typing import List, Dict, Any, Optional
from torch.utils.data import DataLoader

from src.data.vizwiz_dataset import VizWizSample
from src.data.dataset_cache import DatasetCacheManager
from src.data.torch_dataset import VizWizDetectionDataset, collate_fn
from src.models.detector_model import create_model
from src.evaluation.coco_evaluator import evaluate_model_map


def append_metrics_jsonl(metrics_file: Path, record: Dict[str, Any]):
    """Appends one flushed JSON line per epoch."""
    metrics_file.parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")
        f.flush()


def update_notebook_monitor(
    notebook_path: Path,
    system_name: str,
    metrics_file: Path,
    total_epochs: int,
    best_map50: float,
    best_epoch: int
):
    """
    Updates the monitor cell in the notebook with live text table & statistics,
    ensuring each epoch result is permanently saved in the jupyter notebook file.
    """
    if not notebook_path.exists():
        return

    try:
        with open(notebook_path, "r", encoding="utf-8") as f:
            nb = json.load(f)

        # Read last 15 epochs from metrics.jsonl
        records = []
        if metrics_file.exists():
            with open(metrics_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        records.append(json.loads(line))

        last_15 = records[-15:]
        lines = [
            f"=== Live Training Monitor: System {system_name} ===",
            f"Epochs Completed: {len(records)}/{total_epochs} | Best Val mAP@0.5: {best_map50:.4f} (Epoch {best_epoch})",
            "",
            f"{'Epoch':<6} | {'Stage':<6} | {'Train Loss':<11} | {'Val mAP@0.5':<12} | {'Val mAP@[.5:.95]':<16} | {'Val Recall':<12} | {'Time':<8}",
            "-" * 80
        ]
        for r in last_15:
            lines.append(
                f"{r.get('epoch', '-'):<6} | "
                f"{r.get('stage', '-'):<6} | "
                f"{r.get('train_loss', 0.0):<11.4f} | "
                f"{r.get('val_map_50', 0.0):<12.4f} | "
                f"{r.get('val_map', 0.0):<16.4f} | "
                f"{r.get('val_recall_50', 0.0):<12.4f} | "
                f"{r.get('epoch_time_sec', 0.0):<7.1f}s"
            )

        # Find or create monitor cell
        monitor_cell = None
        for cell in nb.get("cells", []):
            if cell.get("metadata", {}).get("privacy_lens_monitor") == True:
                monitor_cell = cell
                break

        if monitor_cell is None:
            monitor_cell = {
                "cell_type": "code",
                "execution_count": 99,
                "metadata": {"privacy_lens_monitor": True},
                "source": [
                    "# Live Monitor Cell (auto-updated by training engine)\n",
                    "import json\n",
                    "from pathlib import Path\n",
                    f"metrics_path = Path('{metrics_file.as_posix()}')\n",
                    "if metrics_path.exists():\n",
                    "    with open(metrics_path, 'r') as f:\n",
                    "        for line in f:\n",
                    "            print(line.strip())\n"
                ],
                "outputs": []
            }
            nb["cells"].append(monitor_cell)

        monitor_cell["outputs"] = [{
            "name": "stdout",
            "output_type": "stream",
            "text": [l + "\n" for l in lines]
        }]

        with open(notebook_path, "w", encoding="utf-8") as f:
            json.dump(nb, f, indent=2)
    except Exception as e:
        print(f"Warning: Failed to update notebook monitor cell: {e}", flush=True)


def run_finetuning_experiment(
    system_name: str,
    train_samples: List[VizWizSample],
    val_samples: List[VizWizSample],
    train_cache: DatasetCacheManager,
    val_cache: DatasetCacheManager,
    config: Dict[str, Any],
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    results_dir: str = "results",
    checkpoint_dir: str = "models/checkpoints",
    notebook_path: Optional[str] = "notebooks/privacylens_training.ipynb"
) -> Dict[str, Any]:
    """
    Executes fine-tuning for System B (heads only) or System C (full fine-tuning).
    - Validates every 2 epochs.
    - Early stopping and best checkpoint selection on val mAP@0.5.
    - Appends per-epoch metrics to results/<run>/metrics.jsonl.
    - Auto-updates the Jupyter Notebook monitor cell each epoch.
    - Supports auto-resume from latest checkpoint.
    """
    run_dir = Path(results_dir) / f"system_{system_name.lower()}"
    run_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = Path(checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = Path(results_dir) / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    metrics_file = run_dir / "metrics.jsonl"
    best_checkpoint_path = ckpt_dir / f"privacylens_best_{system_name.lower()}.pt"
    latest_checkpoint_path = ckpt_dir / f"privacylens_latest_{system_name.lower()}.pt"
    nb_path = Path(notebook_path) if notebook_path else None

    total_epochs = config.get("total_epochs", 30)
    stage1_epochs = config.get("stage1_epochs", 5) if system_name.upper() == "C" else total_epochs
    val_every = config.get("val_every", 2)
    batch_size = config.get("batch_size", 2)
    grad_accum_steps = config.get("grad_accum_steps", 4)
    patience = config.get("early_stopping_patience", 5)  # 5 validation checks = 10 epochs

    train_dataset = VizWizDetectionDataset(train_samples, train_cache, mode=config.get("mode", "class_agnostic"))
    val_dataset = VizWizDetectionDataset(val_samples, val_cache, mode=config.get("mode", "class_agnostic"))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    # Initialize model with deterministic seed
    torch.manual_seed(config.get("seed", 42))
    model = create_model(
        backbone_type=config.get("backbone", "mobilenet_v3"),
        num_classes=config.get("num_classes", 2),
        min_size=config.get("min_size", 512),
        max_size=config.get("max_size", 640),
        box_score_thresh=0.01
    ).to(device)

    # Stage 1: Freeze backbone
    for param in model.backbone.parameters():
        param.requires_grad = False

    head_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(head_params, lr=config.get("stage1_lr", 1e-3), weight_decay=config.get("weight_decay", 1e-4))
    scaler = torch.amp.GradScaler('cuda') if torch.cuda.is_available() else None

    start_epoch = 1
    history = []
    best_val_map50 = -1.0
    best_epoch = 0
    checks_without_improvement = 0

    # Auto-resume check
    if latest_checkpoint_path.exists():
        print(f"Resuming System {system_name} from {latest_checkpoint_path}...", flush=True)
        ckpt = torch.load(latest_checkpoint_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        history = ckpt.get("history", [])
        best_val_map50 = ckpt.get("best_val_map50", -1.0)
        best_epoch = ckpt.get("best_epoch", 0)
        checks_without_improvement = ckpt.get("checks_without_improvement", 0)

        # Check if resumed at stage 2
        if system_name.upper() == "C" and start_epoch > stage1_epochs:
            for param in model.backbone.parameters():
                param.requires_grad = True
            param_groups = [
                {"params": model.backbone.parameters(), "lr": config.get("stage2_backbone_lr", 5e-5)},
                {"params": [p for n, p in model.named_parameters() if not n.startswith("backbone")], "lr": config.get("stage2_heads_lr", 5e-4)}
            ]
            optimizer = torch.optim.AdamW(param_groups, weight_decay=config.get("weight_decay", 1e-4))
            if "optimizer_state_dict" in ckpt:
                try:
                    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
                except Exception:
                    pass
        elif "optimizer_state_dict" in ckpt:
            try:
                optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            except Exception:
                pass

    print(f"\n==================== Launching System {system_name} Fine-Tuning ====================")
    print(f"Total Epochs: {total_epochs} | Validate Every: {val_every} epochs | Batch Size: {batch_size} (Accum: {grad_accum_steps})")
    print(f"Train Images: {len(train_samples)} | Val Images: {len(val_samples)}")
    print(f"{'Epoch':<6} | {'Stage':<6} | {'Train Loss':<11} | {'Val mAP@0.5':<12} | {'Val mAP@[.5:.95]':<16} | {'Val Recall':<12} | {'Epoch Time':<10}")
    print("-" * 85, flush=True)

    total_train_start = time.time()

    for epoch in range(start_epoch, total_epochs + 1):
        epoch_start = time.time()
        stage = 1 if (system_name.upper() == "B" or epoch <= stage1_epochs) else 2

        # Transition to Stage 2 for System C
        if system_name.upper() == "C" and epoch == stage1_epochs + 1:
            print("\n>>> System C Stage 2: Unfreezing backbone with differential learning rate...", flush=True)
            for param in model.backbone.parameters():
                param.requires_grad = True
            param_groups = [
                {"params": model.backbone.parameters(), "lr": config.get("stage2_backbone_lr", 5e-5)},
                {"params": [p for n, p in model.named_parameters() if not n.startswith("backbone")], "lr": config.get("stage2_heads_lr", 5e-4)}
            ]
            optimizer = torch.optim.AdamW(param_groups, weight_decay=config.get("weight_decay", 1e-4))

        model.train()
        train_loss = 0.0
        # Use AMP in Stage 1, switch to stable FP32 in Stage 2
        use_amp = (scaler is not None) and (stage == 1)

        for step, (images, targets) in enumerate(train_loader):
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items() if k in ["boxes", "labels"]} for t in targets]

            if use_amp:
                with torch.amp.autocast('cuda'):
                    loss_dict = model(images, targets)
                    total_batch_loss = sum(loss_dict.values())
                    losses = total_batch_loss / grad_accum_steps

                if torch.isfinite(total_batch_loss):
                    scaler.scale(losses).backward()
                    train_loss += total_batch_loss.item()
                else:
                    print(f"Warning: Non-finite loss encountered at step {step}, skipping batch.", flush=True)

                if (step + 1) % grad_accum_steps == 0 or (step + 1) == len(train_loader):
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
            else:
                loss_dict = model(images, targets)
                total_batch_loss = sum(loss_dict.values())
                losses = total_batch_loss / grad_accum_steps
                if torch.isfinite(total_batch_loss):
                    losses.backward()
                    train_loss += total_batch_loss.item()
                else:
                    print(f"Warning: Non-finite loss encountered at step {step}, skipping batch.", flush=True)

                if (step + 1) % grad_accum_steps == 0 or (step + 1) == len(train_loader):
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                    optimizer.step()
                    optimizer.zero_grad()

        avg_train_loss = train_loss / len(train_loader)

        # Validation check every `val_every` epochs or on final epoch
        is_val_epoch = (epoch % val_every == 0) or (epoch == total_epochs)
        if is_val_epoch:
            val_eval = evaluate_model_map(model, val_loader, device=device, score_thresh=0.01)
            val_map50 = val_eval["map_50"]
            val_map = val_eval["map"]
            val_recall50 = val_eval["mar_100"]
        else:
            val_map50 = history[-1]["val_map_50"] if history else 0.0
            val_map = history[-1]["val_map"] if history else 0.0
            val_recall50 = history[-1]["val_recall_50"] if history else 0.0

        epoch_time = time.time() - epoch_start
        val_str_50 = f"{val_map50:.4f}" if is_val_epoch else f"({val_map50:.4f})"
        val_str_map = f"{val_map:.4f}" if is_val_epoch else f"({val_map:.4f})"
        val_str_rec = f"{val_recall50:.4f}" if is_val_epoch else f"({val_recall50:.4f})"

        print(
            f"{epoch:<6} | Stage {stage:<2} | {avg_train_loss:<11.4f} | "
            f"{val_str_50:<12} | {val_str_map:<16} | {val_str_rec:<12} | {epoch_time:<9.1f}s",
            flush=True
        )

        record = {
            "epoch": epoch,
            "stage": stage,
            "train_loss": round(avg_train_loss, 4),
            "val_map_50": round(val_map50, 4),
            "val_map": round(val_map, 4),
            "val_recall_50": round(val_recall50, 4),
            "epoch_time_sec": round(epoch_time, 2),
            "validated": is_val_epoch
        }
        history.append(record)
        append_metrics_jsonl(metrics_file, record)

        # Checkpointing
        torch.save({
            "epoch": epoch,
            "stage": stage,
            "system": system_name,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_val_map50": best_val_map50,
            "best_epoch": best_epoch,
            "checks_without_improvement": checks_without_improvement,
            "history": history,
            "config": config
        }, latest_checkpoint_path)

        if is_val_epoch:
            if val_map50 > best_val_map50:
                best_val_map50 = val_map50
                best_epoch = epoch
                checks_without_improvement = 0
                torch.save({
                    "epoch": epoch,
                    "stage": stage,
                    "system": system_name,
                    "model_state_dict": model.state_dict(),
                    "val_map_50": val_map50,
                    "val_metrics": val_eval,
                    "config": config
                }, best_checkpoint_path)
            else:
                checks_without_improvement += 1
                if checks_without_improvement >= patience:
                    print(f"\n[Early Stopping] System {system_name}: No improvement in val mAP@0.5 for {patience} validation cycles ({patience*val_every} epochs).", flush=True)
                    if nb_path:
                        update_notebook_monitor(nb_path, system_name, metrics_file, total_epochs, best_val_map50, best_epoch)
                    break

        if nb_path:
            update_notebook_monitor(nb_path, system_name, metrics_file, total_epochs, best_val_map50, best_epoch)

    total_time = time.time() - total_train_start
    print("=" * 85)
    print(f"System {system_name} finished in {total_time/60:.2f} min. Best val mAP@0.5: {best_val_map50:.4f} (Epoch {best_epoch})", flush=True)

    # Save summary curves
    try:
        epochs_ran = [h["epoch"] for h in history]
        train_losses = [h["train_loss"] for h in history]
        val_map50s = [h["val_map_50"] for h in history]

        fig, ax1 = plt.subplots(figsize=(9, 5))
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Train Loss", color="tab:blue")
        ax1.plot(epochs_ran, train_losses, color="tab:blue", marker="o", lw=2, label="Train Loss")
        ax1.tick_params(axis="y", labelcolor="tab:blue")

        ax2 = ax1.twinx()
        ax2.set_ylabel("Validation mAP@0.50", color="tab:red")
        ax2.plot(epochs_ran, val_map50s, color="tab:red", marker="s", lw=2, label="Val mAP@0.50")
        ax2.tick_params(axis="y", labelcolor="tab:red")

        plt.title(f"System {system_name} Training Curves (Loss & Val mAP@0.5)")
        fig.tight_layout()
        curve_path = fig_dir / f"system_{system_name.lower()}_curves.png"
        plt.savefig(curve_path, dpi=150)
        plt.close()
    except Exception as e:
        print(f"Warning: Failed to save curve plot: {e}", flush=True)

    return {
        "system": system_name,
        "best_val_map_50": best_val_map50,
        "best_epoch": best_epoch,
        "best_checkpoint": str(best_checkpoint_path),
        "history": history
    }
