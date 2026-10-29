import pytest
import os
from PIL import Image
from src.data.vizwiz_dataset import polygon_to_bbox, load_annotations
from src.evaluation.detection_metrics import compute_iou, evaluate_detections
from src.evaluation.visualize import create_side_by_side_comparison

def test_polygon_to_bbox():
    poly = [[10.0, 20.0], [50.0, 20.0], [50.0, 80.0], [10.0, 80.0]]
    bbox = polygon_to_bbox(poly)
    assert bbox == [10.0, 20.0, 50.0, 80.0]

def test_compute_iou():
    box1 = [0.0, 0.0, 10.0, 10.0]
    box2 = [5.0, 0.0, 15.0, 10.0]
    # Intersection = 5 * 10 = 50, Union = 100 + 100 - 50 = 150 -> IoU = 1/3
    iou = compute_iou(box1, box2)
    assert abs(iou - (1.0 / 3.0)) < 1e-4

def test_evaluate_detections():
    preds = [{'image_id': 'img1', 'boxes': [[0.0, 0.0, 10.0, 10.0]], 'scores': [0.9]}]
    gts = [{'image_id': 'img1', 'boxes': [[0.0, 0.0, 10.0, 10.0]], 'labels': ['Face']}]
    metrics = evaluate_detections(preds, gts, iou_threshold=0.5)
    assert metrics['true_positives'] == 1
    assert metrics['false_negatives'] == 0
    assert metrics['precision'] == 1.0
    assert metrics['recall'] == 1.0

def test_visualization_output():
    img = Image.new("RGB", (100, 100), (255, 255, 255))
    gt = [{'bbox': [10, 10, 50, 50], 'label': 'Test'}]
    pred = [{'bbox': [20, 20, 60, 60], 'score': 0.8, 'label': 'TestPred'}]
    canvas = create_side_by_side_comparison(img, gt, pred)
    assert canvas.size[0] > 200
