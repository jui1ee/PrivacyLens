import os
from typing import List, Dict, Any, Tuple, Optional
import torch
import torchvision
from PIL import Image, ImageDraw, ImageFilter
import numpy as np

from src.models.detector_model import create_model
from src.pipeline.ocr_engine import extract_ocr_regions
from src.pipeline.query_analyzer import analyze_query_relevance

def compute_box_iou(b1: List[float], b2: List[float]) -> float:
    x1 = max(b1[0], b2[0])
    y1 = max(b1[1], b2[1])
    x2 = min(b1[2], b2[2])
    y2 = min(b1[3], b2[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    a1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
    a2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
    union = a1 + a2 - inter
    return inter / union if union > 0 else 0.0

class PrivacyPipeline:
    def __init__(self, checkpoint_path: str = "models/checkpoints/privacylens_best.pt", device: Optional[str] = None):
        self.device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))
        self.checkpoint_path = checkpoint_path
        self.model = self._load_detector(checkpoint_path)

    def _load_detector(self, path: str):
        model = create_model(backbone_type="mobilenet_v3", num_classes=2, box_score_thresh=0.01).to(self.device)
        if os.path.exists(path):
            ckpt = torch.load(path, map_location=self.device)
            state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
            model.load_state_dict(state_dict)
            print(f"Loaded detector weights from {path}")
        else:
            print(f"Warning: Checkpoint {path} not found. Running initialized detector.")
        model.eval()
        return model

    def set_checkpoint(self, new_checkpoint_path: str):
        self.checkpoint_path = new_checkpoint_path
        self.model = self._load_detector(new_checkpoint_path)

    def run_detection(self, image: Image.Image, score_thresh: float = 0.3) -> List[Dict[str, Any]]:
        w, h = image.size
        img_t = torchvision.transforms.functional.to_tensor(image).to(self.device)
        with torch.no_grad():
            preds = self.model([img_t])[0]

        boxes = preds["boxes"].cpu().numpy()
        scores = preds["scores"].cpu().numpy()
        labels = preds["labels"].cpu().numpy()

        detected = []
        for b, s, l in zip(boxes, scores, labels):
            if s >= score_thresh:
                detected.append({
                    "box": [float(b[0]), float(b[1]), float(b[2]), float(b[3])],
                    "score": float(s),
                    "source": "detector",
                    "label": "Sensitive Region"
                })
        return detected

    def analyze_full(
        self,
        image: Image.Image,
        query: str = "",
        score_thresh: float = 0.3,
        user_keep_zones: Optional[List[List[float]]] = None
    ) -> Dict[str, Any]:
        """
        Executes end-to-end task-aware visual privacy pipeline:
        1. Detector model inference for sensitive regions.
        2. OCR extraction for text and regex sensitive identifiers.
        3. Query relevance analysis: determines if text is crucial for answering the query.
        4. Keep Zone reconciliation: user keep zones & question-derived proposals.
        5. Action policy determination: REDACT, PRESERVE, or FLAG.
        """
        w, h = image.size
        detector_regions = self.run_detection(image, score_thresh=score_thresh)
        ocr_regions = extract_ocr_regions(image)
        ocr_analyzed = analyze_query_relevance(query, ocr_regions)

        all_regions = []

        # Process detector regions
        for d in detector_regions:
            all_regions.append({
                "id": f"det_{len(all_regions)}",
                "box": d["box"],
                "score": d["score"],
                "source": "Model Detector",
                "text": None,
                "crucial_for_query": False,
                "reason": "Visual sensitive pattern detected by trained model",
                "action": "REDACT"
            })

        # Process OCR regions
        for o in ocr_analyzed:
            is_crucial = o.get("crucial_for_query", False)
            pattern = o.get("pattern_match")
            text = o.get("text", "")

            if pattern:
                reason = f"Sensitive pattern: {pattern} ('{text}')"
                action = "FLAG" if is_crucial else "REDACT"
            elif is_crucial:
                reason = f"Crucial for answering question (matches query tokens: {', '.join(o.get('matched_keywords', []))})"
                action = "PRESERVE"
            else:
                reason = f"Detected text: '{text}'"
                action = "PRESERVE"

            all_regions.append({
                "id": f"ocr_{len(all_regions)}",
                "box": o["box"],
                "score": o.get("confidence", 0.9),
                "source": "OCR Engine",
                "text": text,
                "crucial_for_query": is_crucial,
                "reason": reason,
                "action": action
            })

        # Apply User Keep Zones (overrides REDACT to PRESERVE or FLAG)
        if user_keep_zones:
            for reg in all_regions:
                for kz in user_keep_zones:
                    if compute_box_iou(reg["box"], kz) > 0.1:
                        if reg["action"] == "REDACT":
                            reg["action"] = "FLAG"
                            reg["reason"] += " | Overlaps user Keep Zone (requires confirmation)"

        return {
            "image_size": [w, h],
            "query": query,
            "regions": all_regions,
            "counts": {
                "total": len(all_regions),
                "redact": sum(1 for r in all_regions if r["action"] == "REDACT"),
                "preserve": sum(1 for r in all_regions if r["action"] == "PRESERVE"),
                "flag": sum(1 for r in all_regions if r["action"] == "FLAG")
            }
        }

    def render_redacted_image(
        self,
        image: Image.Image,
        regions: List[Dict[str, Any]],
        style: str = "Opaque Black",
        padding: int = 4
    ) -> Image.Image:
        """Applies non-destructive redaction, exporting a new image without altering original."""
        img_out = image.copy().convert("RGB")
        w, h = img_out.size

        for r in regions:
            if r.get("action") == "REDACT":
                b = r["box"]
                x1 = max(0, int(b[0]) - padding)
                y1 = max(0, int(b[1]) - padding)
                x2 = min(w, int(b[2]) + padding)
                y2 = min(h, int(b[3]) + padding)

                if x2 > x1 and y2 > y1:
                    if style == "Opaque Black":
                        draw = ImageDraw.Draw(img_out)
                        draw.rectangle([x1, y1, x2, y2], fill=(0, 0, 0))
                    elif style == "Pixelate":
                        crop = img_out.crop((x1, y1, x2, y2))
                        crop_small = crop.resize((max(1, (x2 - x1) // 12), max(1, (y2 - y1) // 12)), Image.Resampling.NEAREST)
                        crop_pixelated = crop_small.resize((x2 - x1, y2 - y1), Image.Resampling.NEAREST)
                        img_out.paste(crop_pixelated, (x1, y1))
                    elif style == "Gaussian Blur":
                        crop = img_out.crop((x1, y1, x2, y2))
                        blurred = crop.filter(ImageFilter.GaussianBlur(radius=15))
                        img_out.paste(blurred, (x1, y1))

        return img_out
