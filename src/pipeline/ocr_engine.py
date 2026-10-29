import re
from typing import List, Dict, Any, Tuple
from PIL import Image
import numpy as np

# Regular expressions for high-risk sensitive text patterns
REGEX_PATTERNS = {
    "Email": re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,7}\b'),
    "Phone Number": re.compile(r'(\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b'),
    "Credit Card / Account": re.compile(r'\b(?:\d{4}[-\s]?){3}\d{4}\b|\b\d{13,19}\b'),
    "SSN / National ID": re.compile(r'\b\d{3}-\d{2}-\d{4}\b'),
    "Date of Birth": re.compile(r'\b(?:DOB|Date of Birth|D\.O\.B)[:\s]*\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b', re.IGNORECASE)
}

_OCR_READER = None

def get_ocr_reader():
    global _OCR_READER
    if _OCR_READER is None:
        import easyocr
        _OCR_READER = easyocr.Reader(['en'], gpu=False, verbose=False)
    return _OCR_READER

def extract_ocr_regions(image: Image.Image) -> List[Dict[str, Any]]:
    """
    Extracts text regions with bounding boxes using EasyOCR.
    Returns: List of dicts with:
      - 'box': [x1, y1, x2, y2]
      - 'text': recognized string
      - 'confidence': float
      - 'pattern_match': regex name or None
    """
    reader = get_ocr_reader()
    img_np = np.array(image.convert("RGB"))
    results = reader.readtext(img_np)
    
    ocr_regions = []
    for bbox, text, conf in results:
        # bbox is [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
        xs = [pt[0] for pt in bbox]
        ys = [pt[1] for pt in bbox]
        x1, x2 = float(min(xs)), float(max(xs))
        y1, y2 = float(min(ys)), float(max(ys))
        
        # Check against sensitive regex patterns
        matched_pattern = None
        for name, pattern in REGEX_PATTERNS.items():
            if pattern.search(text):
                matched_pattern = name
                break
                
        ocr_regions.append({
            "box": [x1, y1, x2, y2],
            "text": text.strip(),
            "confidence": float(conf),
            "pattern_match": matched_pattern
        })
        
    return ocr_regions
