import torch
from typing import List, Dict, Any
from torchmetrics.detection.mean_ap import MeanAveragePrecision

def evaluate_model_map(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    score_thresh: float = 0.01
) -> Dict[str, float]:
    """
    Evaluates detector using TorchMetrics MeanAveragePrecision (COCO standard).
    Model is placed in eval mode; predictions are collected under torch.no_grad().
    """
    model.eval()
    metric = MeanAveragePrecision(box_format="xyxy", iou_type="bbox")
    
    with torch.no_grad():
        for images, targets in dataloader:
            images = [img.to(device) for img in images]
            predictions = model(images)
            
            # Format predictions
            preds_formatted = []
            for pred in predictions:
                p_boxes = pred["boxes"].cpu()
                p_scores = pred["scores"].cpu()
                p_labels = pred["labels"].cpu()
                
                # Filter by score threshold
                keep = p_scores >= score_thresh
                preds_formatted.append({
                    "boxes": p_boxes[keep],
                    "scores": p_scores[keep],
                    "labels": p_labels[keep]
                })
                
            # Format targets
            targets_formatted = []
            for target in targets:
                targets_formatted.append({
                    "boxes": target["boxes"].cpu(),
                    "labels": target["labels"].cpu()
                })
                
            metric.update(preds_formatted, targets_formatted)
            
    results = metric.compute()
    
    # Extract clean floats
    return {
        "map_50": float(results["map_50"].item()),
        "map": float(results["map"].item()),
        "map_75": float(results["map_75"].item()),
        "mar_100": float(results["mar_100"].item()),
        "map_small": float(results["map_small"].item()),
        "map_medium": float(results["map_medium"].item()),
        "map_large": float(results["map_large"].item())
    }
