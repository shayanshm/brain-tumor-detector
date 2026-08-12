"""
Augmentation آگاه از bbox برای Faster R-CNN — اصلاح باگ Phase 3 (transforms=None بود).
شامل: Horizontal Flip (p=0.5) + Brightness/Contrast jitter — دقیقاً همان دو augmentation
که قرار بود از ابتدا اعمال شود، و در Phase 4 روی YOLO11 هم عیناً پیاده می‌شود.

نکته‌ی حیاتی: هنگام flip افقی تصویر، مختصات bbox هم باید flip شوند وگرنه annotation
اشتباه می‌شود (این دقیقاً همان باگی است که این ماژول با تست دقیق از آن جلوگیری می‌کند).
"""
import random

import torch
import torchvision.transforms.functional as F


class DetectionAugmentation:
    def __init__(self, hflip_prob: float = 0.5, brightness: float = 0.2, contrast: float = 0.2, train: bool = True):
        self.hflip_prob = hflip_prob if train else 0.0
        self.brightness = brightness if train else 0.0
        self.contrast = contrast if train else 0.0
        self.train = train

    def __call__(self, image, target):
        # --- Horizontal Flip (هم تصویر هم bbox ها) ---
        if random.random() < self.hflip_prob:
            width = image.width
            image = F.hflip(image)
            boxes = target["boxes"].clone()
            if boxes.numel() > 0:
                x1 = boxes[:, 0].clone()
                x2 = boxes[:, 2].clone()
                boxes[:, 0] = width - x2
                boxes[:, 2] = width - x1
            target["boxes"] = boxes

        # --- Brightness / Contrast jitter (فقط تصویر، bbox تغییر نمی‌کند) ---
        if self.brightness > 0 and random.random() < 0.5:
            factor = 1.0 + random.uniform(-self.brightness, self.brightness)
            image = F.adjust_brightness(image, max(factor, 0.1))
        if self.contrast > 0 and random.random() < 0.5:
            factor = 1.0 + random.uniform(-self.contrast, self.contrast)
            image = F.adjust_contrast(image, max(factor, 0.1))

        image = F.to_tensor(image)
        return image, target


def get_transform(train: bool):
    return DetectionAugmentation(hflip_prob=0.5, brightness=0.2, contrast=0.2, train=train)
