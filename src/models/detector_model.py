import torch
import torchvision
from torchvision.models.detection import (
    fasterrcnn_mobilenet_v3_large_fpn, FasterRCNN_MobileNet_V3_Large_FPN_Weights,
    fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights
)
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.anchor_utils import AnchorGenerator

def create_model(
    backbone_type: str = "mobilenet_v3",
    num_classes: int = 2,  # 1 (sensitive) + 1 (background) for MODE A; 24 for MODE B
    min_size: int = 640,
    max_size: int = 800,
    box_score_thresh: float = 0.01,
    box_detections_per_img: int = 100,
    custom_anchors: bool = True
):
    """
    Creates Faster R-CNN with properly initialized classification/regression heads.
    Pretrained on COCO; custom heads initialized for VizWiz sensitive region detection.
    """
    if backbone_type == "mobilenet_v3":
        weights = FasterRCNN_MobileNet_V3_Large_FPN_Weights.DEFAULT
        model = fasterrcnn_mobilenet_v3_large_fpn(
            weights=weights,
            min_size=min_size,
            max_size=max_size,
            box_score_thresh=box_score_thresh,
            box_detections_per_img=box_detections_per_img
        )
        in_features = model.roi_heads.box_predictor.cls_score.in_features
    elif backbone_type == "resnet50_v2":
        weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
        model = fasterrcnn_resnet50_fpn_v2(
            weights=weights,
            min_size=min_size,
            max_size=max_size,
            box_score_thresh=box_score_thresh,
            box_detections_per_img=box_detections_per_img
        )
        in_features = model.roi_heads.box_predictor.cls_score.in_features
    else:
        raise ValueError(f"Unknown backbone: {backbone_type}")

    # Reinitialize box predictor head
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    
    return model
