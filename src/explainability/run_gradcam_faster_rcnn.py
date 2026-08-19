"""
Phase 7.2 — تولید heatmap برای Faster R-CNN روی تصاویر منتخب (good/bad/failure).

نحوه‌ی اجرا در Kaggle:
    !python /kaggle/working/brain-tumor-detector/src/explainability/run_gradcam_faster_rcnn.py \\
        --data-root /kaggle/working/brain-tumor-detector/data/processed \\
        --checkpoint /kaggle/working/brain-tumor-detector/outputs_v3/checkpoints/faster_rcnn_epoch50.pt \\
        --out-dir /kaggle/working/brain-tumor-detector/outputs_v3/logs/gradcam
"""
import sys
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms.functional as F
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from models.faster_rcnn import build_faster_rcnn  # noqa: E402
from models.coco_dataset import NUM_CLASSES_WITH_BACKGROUND  # noqa: E402
from explainability.eigencam import EigenCAM, overlay_heatmap  # noqa: E402

CLASS_NAMES = ["background", "glioma", "meningioma", "pituitary"]

# نمونه‌های منتخب (از تحلیل Phase 5 predictions -- پوشش good/bad/failure)
SELECTED_IMAGES = {
    "good_1": "593_jpg.rf.c2654f680d2f411dc9d641205a53efeb.jpg",
    "good_2": "2115_jpg.rf.6c0e37ceb10f6b65e87b6abd65199ec4.jpg",
    "bad_1": "1538_jpg.rf.ca397a75212020ed5e7bccf80c961bdc.jpg",
    "bad_2": "2278_jpg.rf.36dd4fff23e253924259387cda33a1c8.jpg",
    "failure_1": "1058_jpg.rf.a8d95af23ed6b011705441ef351c1f73.jpg",
    "failure_2_crossmodel": "733_jpg.rf.c4435dc9fc79ad36292d2f43f6df321c.jpg",  # FR ناموفق، YOLO موفق
}


def main(args):
    data_root = Path(args.data_root)
    images_dir = data_root / "yolo_detection" / "test" / "images"
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_faster_rcnn(num_classes=NUM_CLASSES_WITH_BACKGROUND, pretrained=False)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.to(device)
    model.eval()

    # لایه‌ی هدف: آخرین بلوک ResNet50 backbone (قبل از FPN) -- غنی‌ترین لایه‌ی معنایی
    target_layer = model.backbone.body.layer4
    cam_engine = EigenCAM(model, target_layer)

    for tag, filename in SELECTED_IMAGES.items():
        img_path = images_dir / filename
        if not img_path.exists():
            print(f"⚠️ {filename} پیدا نشد، رد می‌شود")
            continue

        img = Image.open(img_path).convert("RGB")
        tensor = F.to_tensor(img).unsqueeze(0).to(device)
        tensor.requires_grad_(False)

        cam = cam_engine(tensor)

        # همچنین پیش‌بینی واقعی مدل را هم می‌گیریم تا کنار heatmap نشان دهیم
        with torch.no_grad():
            output = model(tensor)[0]
        boxes = output["boxes"].cpu().numpy()
        scores = output["scores"].cpu().numpy()
        labels = output["labels"].cpu().numpy()

        img_np = np.array(img)
        overlay = overlay_heatmap(img_np, cam, alpha=0.45)

        # رسم بهترین پیش‌بینی (اگر وجود داشت) روی overlay برای مرجع بصری
        if len(scores) > 0 and scores[0] >= 0.3:
            import cv2
            x1, y1, x2, y2 = boxes[0].astype(int)
            cv2.rectangle(overlay, (x1, y1), (x2, y2), (255, 255, 255), 2)
            label_txt = f"{CLASS_NAMES[labels[0]]} {scores[0]:.2f}"
            cv2.putText(overlay, label_txt, (x1, max(y1 - 5, 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        out_path = out_dir / f"faster_rcnn_{tag}.png"
        Image.fromarray(overlay).save(out_path)
        best_label = CLASS_NAMES[labels[0]] if len(labels) > 0 else "هیچ"
        best_score = scores[0] if len(scores) > 0 else 0.0
        print(f"{tag}: {filename} -> {out_path} (بهترین پیش‌بینی: {best_label}, score={best_score:.3f})")

    cam_engine.remove_hook()
    print("\nهمه‌ی نمونه‌ها تمام شد.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()
    main(args)
