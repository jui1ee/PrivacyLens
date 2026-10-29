import torch
import torchvision
from torch.utils.data import Dataset
from typing import List, Tuple, Dict, Any, Optional
from PIL import Image

from src.data.vizwiz_dataset import VizWizSample
from src.data.dataset_cache import DatasetCacheManager

class VizWizDetectionDataset(Dataset):
    """
    PyTorch Dataset for VizWiz-Priv sensitive region detection.
    mode: 'class_agnostic' (single sensitive class 1) or 'multi_class' (classes 1..23).
    Background is always label 0.
    """
    def __init__(
        self,
        samples: List[VizWizSample],
        cache_manager: DatasetCacheManager,
        mode: str = "class_agnostic",
        transforms = None
    ):
        self.samples = samples
        self.cache_mgr = cache_manager
        self.mode = mode
        self.transforms = transforms

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        sample = self.samples[idx]
        img = self.cache_mgr.get_image(sample.image_filename)
        w, h = img.size
        
        boxes = []
        labels = []
        
        for r in sample.regions:
            b = r.bbox  # [x_min, y_min, x_max, y_max]
            x1 = max(0.0, min(float(w), float(b[0])))
            y1 = max(0.0, min(float(h), float(b[1])))
            x2 = max(0.0, min(float(w), float(b[2])))
            y2 = max(0.0, min(float(h), float(b[3])))
            
            # Require valid box area (at least 6.0px to avoid sub-pixel degradation after scaling)
            if (x2 - x1) >= 6.0 and (y2 - y1) >= 6.0:
                boxes.append([x1, y1, x2, y2])
                if self.mode == "class_agnostic":
                    labels.append(1)  # Class 1 = Sensitive
                else:
                    labels.append(r.class_idx + 1)  # 1..23
                    
        if len(boxes) == 0:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
        else:
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            labels = torch.as_tensor(labels, dtype=torch.int64)
            
        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([idx]),
            "orig_size": torch.tensor([h, w])
        }
        
        img_tensor = torchvision.transforms.functional.to_tensor(img)
        
        return img_tensor, target

def collate_fn(batch):
    return tuple(zip(*batch))
