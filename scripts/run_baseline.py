import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import json
import argparse
import time
from typing import List, Dict, Any

from src.data.vizwiz_dataset import load_annotations, VIZWIZ_PRIV_CLASSES
from src.data.dataset_cache import DatasetCacheManager
from src.models.pretrained_baseline import PretrainedVisionBaseline
from src.evaluation.detection_metrics import evaluate_detections
from src.evaluation.visualize import create_side_by_side_comparison

def run_baseline_pipeline(
    val_annotation_path: str = "data/raw/Annotations/Annotations/val.json",
    remote_archive_url: str = "https://vizwiz.cs.colorado.edu/dataset_downloads/VizWiz-Priv-v1.0/HoleMean_Images.zip",
    cache_dir: str = "data/cache/val_images",
    output_dir: str = "results/baseline",
    max_eval_samples: int = 25,
    confidence_thresh: float = 0.3
):
    os.makedirs(output_dir, exist_ok=True)
    vis_dir = os.path.join(output_dir, "visualizations")
    os.makedirs(vis_dir, exist_ok=True)
    
    print("=" * 60)
    print("PRIVACYLENS: PHASE 2 BASELINE EVALUATION PIPELINE")
    print("=" * 60)
    print(f"Loading annotations from: {val_annotation_path}")
    all_val_samples = load_annotations(val_annotation_path)
    
    # Filter to private samples that actually have regions for detection evaluation
    private_val_samples = [s for s in all_val_samples if s.is_private == 1 and len(s.regions) > 0]
    print(f"Total validation samples: {len(all_val_samples)}")
    print(f"Total private samples with regions: {len(private_val_samples)}")
    
    eval_subset = private_val_samples[:max_eval_samples]
    print(f"Evaluating baseline on {len(eval_subset)} real validation samples...")
    
    cache_mgr = DatasetCacheManager(cache_dir=cache_dir, remote_url=remote_archive_url)
    baseline_model = PretrainedVisionBaseline(confidence_threshold=confidence_thresh)
    
    ground_truths = []
    predictions = []
    sample_records = []
    
    start_time = time.time()
    
    for i, sample in enumerate(eval_subset):
        print(f"[{i+1}/{len(eval_subset)}] Processing {sample.image_filename} (Question: {sample.question or 'N/A'})...")
        
        # Load real image on demand
        img = cache_mgr.get_image(sample.image_filename)
        
        # Model prediction
        pred = baseline_model.predict(img)
        
        # Ground truth format
        gt_boxes = [r.bbox for r in sample.regions]
        gt_labels = [r.class_name for r in sample.regions]
        gt_dict = {
            "image_id": sample.image_id,
            "boxes": gt_boxes,
            "labels": gt_labels
        }
        ground_truths.append(gt_dict)
        
        # Prediction format
        pred_dict = {
            "image_id": sample.image_id,
            "boxes": pred["boxes"],
            "scores": pred["scores"],
            "labels": pred["labels"]
        }
        predictions.append(pred_dict)
        
        # Save side-by-side visualization for inspection
        gt_vis_regions = [{"bbox": r.bbox, "polygon": r.polygon, "label": r.class_name} for r in sample.regions]
        pred_vis_regions = [{"bbox": b, "score": s, "label": l} for b, s, l in zip(pred["boxes"], pred["scores"], pred["labels"])]
        
        vis_img = create_side_by_side_comparison(
            image=img,
            gt_regions=gt_vis_regions,
            pred_regions=pred_vis_regions,
            title_gt=f"GT ({len(gt_vis_regions)} regions)",
            title_pred=f"COCO Baseline ({len(pred_vis_regions)} preds)"
        )
        vis_path = os.path.join(vis_dir, f"{sample.image_id}_comparison.jpg")
        vis_img.save(vis_path, quality=90)
        
        sample_records.append({
            "image_id": sample.image_id,
            "image_filename": sample.image_filename,
            "question": sample.question,
            "gt_regions_count": len(sample.regions),
            "pred_boxes_count": len(pred["boxes"]),
            "visualization_path": vis_path
        })
        
    elapsed = time.time() - start_time
    
    # Compute quantitative metrics at IoU 0.25 and 0.50
    metrics_025 = evaluate_detections(predictions, ground_truths, iou_threshold=0.25)
    metrics_050 = evaluate_detections(predictions, ground_truths, iou_threshold=0.50)
    
    summary = {
        "evaluation_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_images_evaluated": len(eval_subset),
        "total_inference_time_sec": round(elapsed, 2),
        "avg_time_per_image_sec": round(elapsed / len(eval_subset), 3),
        "metrics_iou_0.25": metrics_025,
        "metrics_iou_0.50": metrics_050,
        "sample_records": sample_records
    }
    
    results_path = os.path.join(output_dir, "baseline_results.json")
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
        
    print("\n" + "=" * 60)
    print("BASELINE EVALUATION RESULTS")
    print("=" * 60)
    print(f"Evaluated: {len(eval_subset)} real validation images in {elapsed:.1f}s")
    print(f"Metrics (IoU @ 0.50): Precision={metrics_050['precision']:.4f}, Recall={metrics_050['recall']:.4f}, F1={metrics_050['f1']:.4f}")
    print(f"Metrics (IoU @ 0.25): Precision={metrics_025['precision']:.4f}, Recall={metrics_025['recall']:.4f}, F1={metrics_025['f1']:.4f}")
    print(f"Ground Truth Sensitive Regions: {metrics_050['total_ground_truth']}")
    print(f"Baseline Predicted Boxes: {metrics_050['total_predictions']}")
    print(f"True Positives: {metrics_050['true_positives']}")
    print(f"False Negatives (Missed Sensitive Content): {metrics_050['false_negatives']}")
    print(f"Saved full JSON summary to: {results_path}")
    print(f"Saved {len(sample_records)} side-by-side visualizations to: {vis_dir}")
    print("=" * 60)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=25, help="Number of real validation samples to evaluate")
    parser.add_argument("--conf", type=float, default=0.3, help="Confidence threshold")
    args = parser.parse_args()
    
    run_baseline_pipeline(max_eval_samples=args.samples, confidence_thresh=args.conf)
