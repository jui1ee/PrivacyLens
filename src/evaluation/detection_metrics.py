import numpy as np
from typing import List, Dict, Tuple, Any

def compute_iou(box1: List[float], box2: List[float]) -> float:
    """Computes Intersection over Union (IoU) of two boxes [x1, y1, x2, y2]."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    box1_area = max(0.0, box1[2] - box1[0]) * max(0.0, box1[3] - box1[1])
    box2_area = max(0.0, box2[2] - box2[0]) * max(0.0, box2[3] - box2[1])

    union_area = box1_area + box2_area - inter_area
    if union_area <= 0.0:
        return 0.0
    return inter_area / union_area

def evaluate_detections(
    predictions: List[Dict[str, Any]],
    ground_truths: List[Dict[str, Any]],
    iou_threshold: float = 0.5
) -> Dict[str, Any]:
    """
    Evaluates object detection performance across images.
    predictions: List of dicts per image:
        {'image_id': str, 'boxes': [[x1, y1, x2, y2], ...], 'scores': [float, ...], 'labels': [int or str, ...]}
    ground_truths: List of dicts per image:
        {'image_id': str, 'boxes': [[x1, y1, x2, y2], ...], 'labels': [int or str, ...]}
    """
    gt_by_img = {gt['image_id']: gt for gt in ground_truths}
    
    total_gt = sum(len(gt['boxes']) for gt in ground_truths)
    total_pred = sum(len(pred['boxes']) for pred in predictions)
    
    true_positives = 0
    false_positives = 0
    
    # Per-image matching
    for pred_dict in predictions:
        img_id = pred_dict['image_id']
        gt_dict = gt_by_img.get(img_id, {'boxes': [], 'labels': []})
        gt_boxes = list(gt_dict['boxes'])
        gt_matched = [False] * len(gt_boxes)
        
        # Sort predictions by score descending if available
        scores = pred_dict.get('scores', [1.0] * len(pred_dict['boxes']))
        sorted_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        
        for idx in sorted_indices:
            p_box = pred_dict['boxes'][idx]
            best_iou = 0.0
            best_gt_idx = -1
            
            for g_idx, g_box in enumerate(gt_boxes):
                if not gt_matched[g_idx]:
                    iou = compute_iou(p_box, g_box)
                    if iou > best_iou:
                        best_iou = iou
                        best_gt_idx = g_idx
                        
            if best_iou >= iou_threshold and best_gt_idx != -1:
                true_positives += 1
                gt_matched[best_gt_idx] = True
            else:
                false_positives += 1
                
    false_negatives = total_gt - true_positives
    precision = true_positives / total_pred if total_pred > 0 else 0.0
    recall = true_positives / total_gt if total_gt > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return {
        "iou_threshold": iou_threshold,
        "total_ground_truth": total_gt,
        "total_predictions": total_pred,
        "true_positives": true_positives,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4)
    }
