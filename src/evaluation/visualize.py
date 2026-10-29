from PIL import Image, ImageDraw, ImageFont
from typing import List, Dict, Any, Optional

def draw_boxes_and_polygons(
    image: Image.Image,
    regions: List[Dict[str, Any]],
    box_color: str = "red",
    fill_opacity: int = 60,
    outline_width: int = 3,
    show_labels: bool = True
) -> Image.Image:
    """Draws bounding boxes, polygons, and labels onto a copy of the image."""
    img_out = image.copy()
    overlay = Image.new("RGBA", img_out.size, (255, 255, 255, 0))
    draw_overlay = ImageDraw.Draw(overlay)
    draw_img = ImageDraw.Draw(img_out)
    
    for r in regions:
        # Check polygon
        if "polygon" in r and r["polygon"]:
            pts = [tuple(p) for p in r["polygon"]]
            if len(pts) >= 3:
                draw_overlay.polygon(pts, fill=(255, 0, 0, fill_opacity))
                draw_img.polygon(pts, outline=box_color, width=outline_width)
        
        # Check bbox [x1, y1, x2, y2]
        if "bbox" in r and r["bbox"]:
            b = r["bbox"]
            draw_img.rectangle(b, outline=box_color, width=outline_width)
            
            if show_labels and "label" in r:
                lbl = str(r["label"])
                score_str = f" {r['score']:.2f}" if "score" in r else ""
                text = f"{lbl}{score_str}"
                # Background for text
                draw_img.rectangle([b[0], max(0, b[1] - 18), b[0] + len(text) * 8 + 6, max(18, b[1])], fill=box_color)
                draw_img.text((b[0] + 3, max(0, b[1] - 16)), text, fill="white")
                
    # Composite overlay
    img_out = Image.alpha_composite(img_out.convert("RGBA"), overlay).convert("RGB")
    return img_out

def create_side_by_side_comparison(
    image: Image.Image,
    gt_regions: List[Dict[str, Any]],
    pred_regions: List[Dict[str, Any]],
    title_gt: str = "Ground Truth",
    title_pred: str = "Baseline Prediction"
) -> Image.Image:
    """Creates a side-by-side visualization of Ground Truth vs Predictions."""
    gt_vis = draw_boxes_and_polygons(image, gt_regions, box_color="green", fill_opacity=50)
    pred_vis = draw_boxes_and_polygons(image, pred_regions, box_color="red", fill_opacity=50)
    
    w, h = image.size
    banner_h = 30
    canvas = Image.new("RGB", (w * 2 + 10, h + banner_h), (240, 240, 240))
    draw = ImageDraw.Draw(canvas)
    
    # Titles
    draw.text((10, 8), title_gt, fill=(0, 100, 0))
    draw.text((w + 20, 8), title_pred, fill=(180, 0, 0))
    
    # Paste images
    canvas.paste(gt_vis, (0, banner_h))
    canvas.paste(pred_vis, (w + 10, banner_h))
    
    return canvas
