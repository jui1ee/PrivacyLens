import os
from typing import List, Dict, Any
from PIL import Image
import torch
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights

class PretrainedVisionBaseline:
    """
    Pretrained object detection baseline using standard COCO-pretrained weights.
    Evaluates zero-shot transfer capability on VizWiz-Priv sensitive classes.
    """
    
    def __init__(self, device: str = "cuda" if torch.cuda.is_available() else "cpu", confidence_threshold: float = 0.25):
        self.device = device
        self.confidence_threshold = confidence_threshold
        print(f"Loading Pretrained COCO Faster R-CNN on {self.device}...")
        self.weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
        self.model = fasterrcnn_resnet50_fpn(weights=self.weights)
        self.model.to(self.device)
        self.model.eval()
        self.transforms = self.weights.transforms()
        self.coco_classes = self.weights.meta["categories"]
        
        # Privacy-relevant COCO classes (e.g. person/face, tv/screen, laptop, book, cell phone)
        self.privacy_relevant_coco = {
            "person", "tv", "laptop", "cell phone", "book"
        }

    def predict(self, image: Image.Image) -> Dict[str, Any]:
        """
        Runs inference on a PIL image.
        Returns:
            boxes: [[x1, y1, x2, y2], ...]
            scores: [float, ...]
            labels: [str, ...]
            is_sensitive_candidate: [bool, ...]
        """
        w, h = image.size
        img_tensor = torchvision.transforms.functional.to_tensor(image).to(self.device)
        
        with torch.no_grad():
            preds = self.model([img_tensor])[0]
            
        boxes = preds["boxes"].cpu().numpy()
        scores = preds["scores"].cpu().numpy()
        label_indices = preds["labels"].cpu().numpy()
        
        filtered_boxes = []
        filtered_scores = []
        filtered_labels = []
        
        for box, score, idx in zip(boxes, scores, label_indices):
            if score >= self.confidence_threshold:
                cls_name = self.coco_classes[idx]
                filtered_boxes.append([float(box[0]), float(box[1]), float(box[2]), float(box[3])])
                filtered_scores.append(float(score))
                filtered_labels.append(cls_name)
                
        return {
            "boxes": filtered_boxes,
            "scores": filtered_scores,
            "labels": filtered_labels
        }
