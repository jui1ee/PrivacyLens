"""
Question-aware redaction pipeline using Florence-2 grounding & OCR,
with fallback/hybrid support for Faster R-CNN detector checkpoints.
"""

import os
import re
import sys
import json
import torch
from PIL import Image, ImageDraw
from typing import List, Dict, Any, Tuple, Optional
from unittest.mock import patch

# Use Florence-2-base by default for 4GB VRAM safety (GTX 1650), or large if specified
DEFAULT_MID = "microsoft/Florence-2-base"
DEFAULT_DETECTOR_CKPT = "models/checkpoints/privacylens_best.pt"
KEEP_OVERLAP = 0.3

dev = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16 if dev == "cuda" else torch.float32

_FLORENCE_MODEL = None
_FLORENCE_PROC = None

PII_PATTERNS = [
    r"[\w.+-]+@[\w-]+\.[\w.-]+",                      # email
    r"\+?\d[\d\s().-]{8,}\d",                         # phone / long numbers
    r"\b(?:\d[ -]?){13,16}\b",                        # card numbers
    r"\b(?:credit\s*card|debit\s*card|mastercard|visa|amex|american\s*express|discover)\b", # card headers
    r"\b(?:card\s*holder|cardholder|exp(?:iry)?\s*date|valid\s*thru|cvv|cvc)\b",           # card fields
    r"\b\d{3}-\d{2}-\d{4}\b",                         # SSN-like
    r"\b\d{1,3}(?:\.\d{1,3}){3}\b",                   # IP address
    r"\b\d{5,6}\b",                                   # PIN / zip codes
    r"\b(?:street|st\.|road|rd\.|avenue|ave|lane|nagar|colony|sector|apt|flat)\b",
    r"\b(?:password|passwd|otp|account no|a/c|iban|ifsc|aadhaar|pan)\b",
    r"\b(?:dollar|dollars|one\s*hundred\s*dollars|federal\s*reserve|banknote|currency)\b", # physical cash / bills
    r"\bhttps?://\S+",                                # URLs
]
PII_RE = re.compile("|".join(PII_PATTERNS), re.I)

STOP = set("why is my a an the not what how does do can i to of in on for it this that are was".split())


import transformers.dynamic_module_utils as dmu
_orig_get_imports = dmu.get_imports

def _no_flash(filename):
    imps = _orig_get_imports(filename)
    if "flash_attn" in imps:
        imps.remove("flash_attn")
    return imps


def get_florence_models(model_id: str = DEFAULT_MID):
    global _FLORENCE_MODEL, _FLORENCE_PROC
    if _FLORENCE_MODEL is None:
        from transformers import AutoProcessor, AutoModelForCausalLM
        with patch.object(dmu, "get_imports", _no_flash):
            _FLORENCE_MODEL = AutoModelForCausalLM.from_pretrained(
                model_id, trust_remote_code=True, torch_dtype=dtype
            ).to(dev).eval()
        _FLORENCE_PROC = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    return _FLORENCE_MODEL, _FLORENCE_PROC


def run_florence(task: str, img: Image.Image, text: str = "", model_id: str = DEFAULT_MID):
    model, proc = get_florence_models(model_id)
    prompt = task + text
    inputs = proc(text=prompt, images=img, return_tensors="pt").to(dev, dtype)
    with torch.no_grad():
        out = model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=1024,
            num_beams=3,
            do_sample=False
        )
    raw = proc.batch_decode(out, skip_special_tokens=False)[0]
    return proc.post_process_generation(raw, task=task, image_size=img.size)[task]


def quad_to_box(q):
    xs, ys = q[0::2], q[1::2]
    return [float(min(xs)), float(min(ys)), float(max(xs)), float(max(ys))]


def area(b):
    return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])


DEFAULT_SENSITIVE_VISUAL_TERMS = "credit card, debit card, money, dollar bill, banknote, passport, driver license, id card, prescription"


def box_iou(b1: List[float], b2: List[float]) -> float:
    i = inter(b1, b2)
    u = area(b1) + area(b2) - i
    return i / u if u > 0 else 0.0


def deduplicate_boxes(items: List[Tuple[List[float], str, str]], iou_thresh: float = 0.6, subsume_thresh: float = 0.85) -> List[Tuple[List[float], str, str]]:
    """
    Deduplicates overlapping or subsumed bounding boxes.
    Prioritizes larger containers to avoid duplicate nesting of sub-elements.
    """
    sorted_items = sorted(items, key=lambda x: area(x[0]), reverse=True)
    kept = []
    for item in sorted_items:
        b = item[0]
        if any(box_iou(b, k[0]) >= iou_thresh for k in kept):
            continue
        if any((inter(b, k[0]) / max(area(b), 1.0)) >= subsume_thresh for k in kept):
            continue
        kept.append(item)
    return kept


def inter(a, b):
    return area([max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])])


def keywords(q: str):
    return {w for w in re.findall(r"[a-z]+", q.lower()) if w not in STOP and len(w) > 2}


def get_detector_boxes(img: Image.Image, checkpoint_path: str = DEFAULT_DETECTOR_CKPT, score_thresh: float = 0.25) -> List[List[float]]:
    if not os.path.exists(checkpoint_path):
        return []
    try:
        from torchvision.transforms.functional import to_tensor
        from src.models.detector_model import create_model
        det = create_model(backbone_type="mobilenet_v3", num_classes=2).to(dev)
        ck = torch.load(checkpoint_path, map_location=dev)
        state_dict = ck["model_state_dict"] if "model_state_dict" in ck else (ck["model"] if "model" in ck else ck)
        det.load_state_dict(state_dict)
        det.eval()
        with torch.no_grad():
            out = det([to_tensor(img).to(dev)])[0]
        return [b.tolist() for b, s in zip(out["boxes"], out["scores"]) if float(s) >= score_thresh]
    except Exception as e:
        print(f"Detector inference skipped: {e}", flush=True)
        return []


SENSITIVE_VISUAL_KEYWORDS = [
    "credit card", "debit card", "bank card",
    "hundred dollar bill", "dollar bill", "banknote", "cash", "money", "currency", "check",
    "passport", "driver license", "driver's license", "id card", "identity card", "identification card",
    "social security", "prescription"
]


def run_question_aware_redaction(
    image: Image.Image,
    question: str,
    detector_ckpt: str = DEFAULT_DETECTOR_CKPT,
    florence_model_id: str = DEFAULT_MID,
    keep_overlap: float = KEEP_OVERLAP,
    padding: int = 4,
    detector_score_thresh: float = 0.25,
    enable_visual_grounding: bool = True,
    sensitive_visual_terms: str = ""
) -> Tuple[Image.Image, Image.Image, Dict[str, Any]]:
    """
    Executes question-aware redaction pipeline:
    1. OCR lines & boxes via Florence-2 <OCR_WITH_REGION>.
    2. Sensitive boxes via PII regex + Caption-Verified Florence-2 visual grounding + Faster R-CNN detector.
    3. Needed regions via grounding <CAPTION_TO_PHRASE_GROUNDING> + query keyword overlap.
    4. Redact sensitive boxes that are NOT inside needed regions.
    """
    img = image.convert("RGB")
    w, h = img.size

    # 1. Florence-2 OCR with Region
    ocr = run_florence("<OCR_WITH_REGION>", img, model_id=florence_model_id)
    lines = []
    if "quad_boxes" in ocr and "labels" in ocr:
        for q, l in zip(ocr["quad_boxes"], ocr["labels"]):
            clean_l = l.replace("</s>", "").replace("<s>", "").strip()
            lines.append((quad_to_box(q), clean_l))

    # 2. Sensitive regions
    raw_sensitive = []

    # 2a. OCR-based text PII
    for b, t in lines:
        if PII_RE.search(t):
            raw_sensitive.append((b, t, "pii_text"))

    # 2b. Visual sensitive objects (credit cards, currency, IDs, documents)
    # CRITICAL: Caption-verified to prevent hallucinated groundings on normal objects
    if enable_visual_grounding:
        try:
            caption = run_florence("<MORE_DETAILED_CAPTION>", img, model_id=florence_model_id)
            ocr_text = " ".join([t for _, t in lines])
            combined_desc = (caption + " " + ocr_text).lower()

            verified_terms = [
                kw for kw in SENSITIVE_VISUAL_KEYWORDS
                if re.search(r'\b' + re.escape(kw) + r'\b', combined_desc, re.I)
            ]

            if verified_terms:
                grounding_prompt = ", ".join(verified_terms)
                vg = run_florence("<CAPTION_TO_PHRASE_GROUNDING>", img, text=grounding_prompt, model_id=florence_model_id)
                if "bboxes" in vg and "labels" in vg:
                    for b, l in zip(vg["bboxes"], vg["labels"]):
                        if area(b) < 0.85 * (w * h):
                            raw_sensitive.append((b, f"Visual {l}", f"visual:{l}"))
        except Exception as e:
            print(f"Visual sensitive grounding skipped: {e}", flush=True)

    # 2c. Trained Faster R-CNN detector
    if detector_ckpt and os.path.exists(detector_ckpt):
        det_boxes = get_detector_boxes(img, checkpoint_path=detector_ckpt, score_thresh=detector_score_thresh)
        for b in det_boxes:
            raw_sensitive.append((b, "", "trained_detector"))

    # Deduplicate overlapping & subsumed sensitive regions
    sensitive = deduplicate_boxes(raw_sensitive)

    # 3. Grounding needed region for the question
    needed_boxes = []
    if question.strip():
        try:
            g = run_florence("<CAPTION_TO_PHRASE_GROUNDING>", img, text=question, model_id=florence_model_id)
            if "bboxes" in g:
                for gb in g["bboxes"]:
                    # Discard degenerate bounding boxes that cover almost the entire photograph (>85% area)
                    if area(gb) < 0.85 * (w * h):
                        needed_boxes.append(gb)
        except Exception as e:
            print(f"Grounding task skipped: {e}", flush=True)

        kws = keywords(question)
        for b, t in lines:
            line_tokens = set(re.findall(r"[a-z]+", t.lower()))
            # Only match if keyword is not part of a PII pattern
            if (kws & line_tokens) and not PII_RE.search(t):
                needed_boxes.append(b)

    # 4. Non-destructive redaction decision
    redacted_img = img.copy()
    overlay_img = img.copy()
    draw_redacted = ImageDraw.Draw(redacted_img)
    draw_overlay = ImageDraw.Draw(overlay_img)

    report = {
        "question": question,
        "redacted": [],
        "kept_needed": [],
        "all_ocr_lines": len(lines),
        "needed_boxes_count": len(needed_boxes)
    }

    # Draw needed boxes in green on overlay
    for nb in needed_boxes:
        draw_overlay.rectangle([nb[0], nb[1], nb[2], nb[3]], outline=(40, 167, 69), width=2)

    for b, t, kind in sensitive:
        frac = max((inter(b, n) / max(area(b), 1.0) for n in needed_boxes), default=0.0)
        item = {
            "box": [round(v, 1) for v in b],
            "text": t,
            "kind": kind,
            "overlap_with_needed": round(frac, 2)
        }

        if frac >= keep_overlap:
            report["kept_needed"].append(item)
            draw_overlay.rectangle([b[0], b[1], b[2], b[3]], outline=(255, 193, 7), width=3)
        else:
            report["redacted"].append(item)
            draw_overlay.rectangle([b[0], b[1], b[2], b[3]], outline=(220, 53, 69), width=3)
            # Black out on final image
            bx1 = max(0, int(b[0]) - padding)
            by1 = max(0, int(b[1]) - padding)
            bx2 = min(w, int(b[2]) + padding)
            by2 = min(h, int(b[3]) + padding)
            draw_redacted.rectangle([bx1, by1, bx2, by2], fill="black")

    return redacted_img, overlay_img, report
