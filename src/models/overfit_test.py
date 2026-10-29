import os
import sys
import time
import json
import torch
import matplotlib.pyplot as plt
from typing import List, Dict, Any
from torch.utils.data import DataLoader

from src.data.vizwiz_dataset import VizWizSample
from src.data.dataset_cache import DatasetCacheManager
from src.data.torch_dataset import VizWizDetectionDataset, collate_fn
from src.models.detector_model import create_model
from src.evaluation.coco_evaluator import evaluate_model_map

def run_overfit_test(
    samples: List[VizWizSample],
    cache_mgr: DatasetCacheManager,
    epochs: int = 120,
    lr: float = 0.0005,
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    fig_save_path: str = "results/figures/overfit_loss_curve.png",
    results_save_path: str = "results/overfit_sanity_results.json"
) -> Dict[str, Any]:
    """
    Overfit Sanity Test: Trains on exactly 8 samples for up to ~120-150 epochs.
    Evaluates on the SAME 8 samples.
    Gate: Recall@0.5 (mar_100 or map_50) >= 0.90 and loss clearly decreasing.
    """
    assert len(samples) == 8, f"Expected exactly 8 samples for overfit test, got {len(samples)}"
    os.makedirs(os.path.dirname(fig_save_path), exist_ok=True)
    os.makedirs(os.path.dirname(results_save_path), exist_ok=True)
    
    dataset = VizWizDetectionDataset(samples, cache_mgr, mode="class_agnostic")
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=collate_fn)
    eval_loader = DataLoader(dataset, batch_size=2, shuffle=False, collate_fn=collate_fn)
    
    model = create_model(
        backbone_type="mobilenet_v3",
        num_classes=2,
        min_size=512,
        max_size=640,
        box_score_thresh=0.01,
        custom_anchors=True
    ).to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    loss_history = []
    
    print(f"Starting Overfit Sanity Test (8 images, {epochs} epochs) on {device}...")
    start_time = time.time()
    
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        
        for images, targets in dataloader:
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items() if k in ["boxes", "labels"]} for t in targets]
            
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            
            optimizer.zero_grad()
            losses.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            optimizer.step()
            
            epoch_loss += losses.item()
            
        avg_loss = epoch_loss / len(dataloader)
        loss_history.append(avg_loss)
        
        if epoch % 20 == 0 or epoch == epochs:
            print(f"Epoch [{epoch:03d}/{epochs}] - Loss: {avg_loss:.4f}")
            
    train_duration = time.time() - start_time
    
    # Evaluate on the SAME 8 images using both TorchMetrics mAP and matching recall at IoU 0.5
    model.eval()
    all_preds = []
    all_gts = []
    with torch.no_grad():
        for images, targets in eval_loader:
            images = [img.to(device) for img in images]
            preds = model(images)
            for p, t in zip(preds, targets):
                p_boxes = p["boxes"].cpu().numpy().tolist()
                p_scores = p["scores"].cpu().numpy().tolist()
                p_labels = p["labels"].cpu().numpy().tolist()
                all_preds.append({
                    "image_id": str(t["image_id"].item()),
                    "boxes": p_boxes,
                    "scores": p_scores,
                    "labels": p_labels
                })
                all_gts.append({
                    "image_id": str(t["image_id"].item()),
                    "boxes": t["boxes"].cpu().numpy().tolist(),
                    "labels": t["labels"].cpu().numpy().tolist()
                })
                
    from src.evaluation.detection_metrics import evaluate_detections
    direct_metrics_050 = evaluate_detections(all_preds, all_gts, iou_threshold=0.50)
    metrics = evaluate_model_map(model, eval_loader, device=device, score_thresh=0.01)
    
    # Plot loss curve
    plt.figure(figsize=(8, 4.5))
    plt.plot(range(1, epochs + 1), loss_history, label="Training Loss", color="#1f77b4", lw=2)
    plt.title(f"Section 3: Overfit Sanity Test Loss Curve (8 samples, {epochs} epochs)")
    plt.xlabel("Epoch")
    plt.ylabel("Total Loss (RPN + ROI)")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_save_path, dpi=150)
    plt.close()
    
    recall_at_05 = direct_metrics_050["recall"]
    gate_passed = (recall_at_05 >= 0.90 or metrics["map_50"] >= 0.90) and (loss_history[-1] < loss_history[0])
    
    results = {
        "epochs": epochs,
        "train_duration_sec": round(train_duration, 2),
        "initial_loss": round(loss_history[0], 4),
        "final_loss": round(loss_history[-1], 4),
        "loss_reduction_pct": round((1.0 - (loss_history[-1] / loss_history[0])) * 100, 2),
        "recall_at_iou_0.50": round(recall_at_05, 4),
        "precision_at_iou_0.50": round(direct_metrics_050["precision"], 4),
        "eval_metrics": metrics,
        "gate_passed": bool(gate_passed),
        "figure_path": fig_save_path
    }
    
    with open(results_save_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
        
    return results
