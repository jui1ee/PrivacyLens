import os
import json
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any, Optional
from PIL import Image

VIZWIZ_PRIV_CLASSES = [
    "Object:Face",
    "Text:Miscellaneous Papers",
    "Text:Computer Screen",
    "Text:Letter",
    "Text:Other",
    "Object:Framed Photo",
    "Text:Suspicious",
    "Text:Pill Bottle/Box",
    "Text:Credit Card",
    "Text:Business Card",
    "Text:Newspaper",
    "Object:Face Reflection",
    "Text:License Plate",
    "Object:Other",
    "Text:Receipt",
    "Object:Pregnancy Test Result",
    "Object:Suspicious",
    "Text:Clothing",
    "Text:Street Sign",
    "Object:Tattoo",
    "Text:Menu",
    "Text:Poster",
    "Text:Book"
]

CLASS_TO_IDX = {cls_name: idx for idx, cls_name in enumerate(VIZWIZ_PRIV_CLASSES)}
IDX_TO_CLASS = {idx: cls_name for idx, cls_name in enumerate(VIZWIZ_PRIV_CLASSES)}

@dataclass
class RegionAnnotation:
    class_name: str
    class_idx: int
    polygon: List[List[float]]
    bbox: List[float]  # [x_min, y_min, x_max, y_max]

@dataclass
class VizWizSample:
    image_id: str
    image_filename: str
    is_private: int
    question: Optional[str]
    regions: List[RegionAnnotation]
    image_width: Optional[int] = None
    image_height: Optional[int] = None

def polygon_to_bbox(polygon: List[List[float]], clip_bounds: Optional[Tuple[int, int]] = None) -> List[float]:
    """Converts a polygon [[x1, y1], [x2, y2], ...] into an axis-aligned bounding box [x_min, y_min, x_max, y_max]."""
    xs = [float(pt[0]) for pt in polygon]
    ys = [float(pt[1]) for pt in polygon]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    
    if clip_bounds is not None:
        w, h = clip_bounds
        x_min = max(0.0, min(float(w), x_min))
        y_min = max(0.0, min(float(h), y_min))
        x_max = max(0.0, min(float(w), x_max))
        y_max = max(0.0, min(float(h), y_max))
        
    return [float(x_min), float(y_min), float(x_max), float(y_max)]

def load_annotations(annotation_path: str) -> List[VizWizSample]:
    """Loads and parses a VizWiz-Priv annotation JSON file."""
    if not os.path.exists(annotation_path):
        raise FileNotFoundError(f"Annotation file not found: {annotation_path}")
        
    with open(annotation_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)
        
    samples = []
    for entry in raw_data:
        image_fn = entry["image"]
        is_private = entry.get("private", 0)
        question = entry.get("question", None)
        
        region_annots = []
        for r in entry.get("private_regions", []):
            cls_name = r.get("class", "Object:Other")
            cls_idx = CLASS_TO_IDX.get(cls_name, CLASS_TO_IDX["Object:Other"])
            poly = r.get("polygon", [])
            if poly and len(poly) >= 3:
                bbox = polygon_to_bbox(poly)
                # Valid box check
                if bbox[2] > bbox[0] and bbox[3] > bbox[1]:
                    region_annots.append(RegionAnnotation(
                        class_name=cls_name,
                        class_idx=cls_idx,
                        polygon=poly,
                        bbox=bbox
                    ))
                
        samples.append(VizWizSample(
            image_id=os.path.splitext(image_fn)[0],
            image_filename=image_fn,
            is_private=is_private,
            question=question,
            regions=region_annots
        ))
        
    return samples
