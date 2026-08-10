"""
Phase 3.2 — Dataset سفارشی torchvision روی COCO JSON تولیدشده در Phase 2.7.
"""
import json
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset

CLASS_NAMES = ["glioma", "meningioma", "pituitary"]
# توجه مهم: در torchvision detection API، class_id=0 همیشه رزرو "background" است.
# بنابراین category_id های ما (0,1,2 در COCO json) باید +1 شوند -> (1,2,3) هنگام آموزش.


class CocoDetectionDataset(Dataset):
    def __init__(self, images_dir: str, coco_json_path: str, transforms=None):
        self.images_dir = Path(images_dir)
        self.transforms = transforms

        with open(coco_json_path) as f:
            coco = json.load(f)

        self.images = {img["id"]: img for img in coco["images"]}
        self.image_ids = sorted(self.images.keys())

        self.annotations_by_image = {}
        for ann in coco["annotations"]:
            self.annotations_by_image.setdefault(ann["image_id"], []).append(ann)

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        img_id = self.image_ids[idx]
        img_info = self.images[img_id]
        img_path = self.images_dir / img_info["file_name"]
        image = Image.open(img_path).convert("RGB")

        anns = self.annotations_by_image.get(img_id, [])
        boxes, labels, areas, iscrowd = [], [], [], []
        for ann in anns:
            x, y, w, h = ann["bbox"]
            if w <= 0 or h <= 0:
                continue  # حفاظت در برابر annotation دژنره (نباید طبق Phase 2.6 وجود داشته باشد، اما محکم‌کاری می‌کنیم)
            boxes.append([x, y, x + w, y + h])  # torchvision: [x_min, y_min, x_max, y_max]
            labels.append(ann["category_id"] + 1)  # +1 چون 0 در torchvision یعنی background
            areas.append(ann["area"])
            iscrowd.append(ann.get("iscrowd", 0))

        target = {
            "boxes": torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4),
            "labels": torch.as_tensor(labels, dtype=torch.int64),
            "image_id": torch.tensor([img_id]),
            "area": torch.as_tensor(areas, dtype=torch.float32),
            "iscrowd": torch.as_tensor(iscrowd, dtype=torch.int64),
        }

        if self.transforms:
            image, target = self.transforms(image, target)
        else:
            import torchvision.transforms.functional as F
            image = F.to_tensor(image)

        return image, target


def collate_fn(batch):
    return tuple(zip(*batch))


NUM_CLASSES_WITH_BACKGROUND = len(CLASS_NAMES) + 1  # 3 کلاس تومور + 1 background = 4
