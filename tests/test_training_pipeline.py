import os
import torch
from src.models.detector_model import create_model
from src.data.torch_dataset import VizWizDetectionDataset
from src.data.vizwiz_dataset import VizWizSample, RegionAnnotation

def test_detector_architecture():
    model = create_model(backbone_type="mobilenet_v3", num_classes=24)
    assert model.roi_heads.box_predictor.cls_score.out_features == 24

def test_checkpoint_integrity():
    ckpt_path = "models/checkpoints/privacylens_best.pt"
    assert os.path.exists(ckpt_path), "Checkpoint must exist on disk"
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    assert "model_state_dict" in checkpoint
    assert "training_log" in checkpoint
    assert len(checkpoint["classes"]) == 23
    assert len(checkpoint["training_log"]) >= 1

def test_torch_dataset_handling():
    sample = VizWizSample(
        image_id="dummy",
        image_filename="dummy.jpg",
        is_private=1,
        question="test?",
        regions=[RegionAnnotation(
            class_name="Object:Face",
            class_idx=0,
            polygon=[[10, 10], [50, 10], [50, 50], [10, 50]],
            bbox=[10, 10, 50, 50]
        )]
    )
    assert sample.regions[0].bbox == [10, 10, 50, 50]
