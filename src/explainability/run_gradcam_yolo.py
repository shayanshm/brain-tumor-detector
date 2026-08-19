"""
Phase 7.2 — تولید heatmap برای YOLO11 روی همان تصاویر منتخب (برای مقایسه‌ی مستقیم با Faster R-CNN).

نحوه‌ی اجرا در Kaggle:
    !python /kaggle/working/brain-tumor-detector/src/explainability/run_gradcam_yolo.py \\
        --data-root /kaggle/working/brain-tumor-detector/data/processed \\
        --checkpoint /path/to/run1/weights/best.pt \\
        --out-dir /kaggle/working/brain-tumor-detector/outputs/yolo11/run1/gradcam
"""
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from explainability.eigencam import EigenCAM, overlay_heatmap  # noqa: E402

CLASS_NAMES = ["glioma", "meningioma", "pituitary"]

# دقیقاً همان تصاویر run_gradcam_faster_rcnn.py -- برای مقایسه‌ی مستقیم side-by-side
SELECTED_IMAGES = {
    "good_1": "593_jpg.rf.c2654f680d2f411dc9d641205a53efeb.jpg",
    "good_2": "2115_jpg.rf.6c0e37ceb10f6b65e87b6abd65199ec4.jpg",
    "bad_1": "1538_jpg.rf.ca397a75212020ed5e7bccf80c961bdc.jpg",
    "bad_2": "2278_jpg.rf.36dd4fff23e253924259387cda33a1c8.jpg",
    "failure_1": "1058_jpg.rf.a8d95af23ed6b011705441ef351c1f73.jpg",
    "failure_2_crossmodel": "733_jpg.rf.c4435dc9fc79ad36292d2f43f6df321c.jpg",  # FR ناموفق، YOLO موفق
}


def main(args):
    from ultralytics import YOLO

    data_root = Path(args.data_root)
    images_dir = data_root / "yolo_detection" / "test" / "images"
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    yolo = YOLO(args.checkpoint)
    underlying_model = yolo.model.to(device)
    underlying_model.eval()

    # لایه‌ی هدف: آخرین لایه‌ی backbone قبل از سر تشخیص (معادل معنایی layer4 در Faster R-CNN)
    # در YOLO11، model.model[9] معمولا آخرین بلوک backbone/SPPF قبل از neck است.
    target_layer = underlying_model.model[9]
    cam_engine = EigenCAM(underlying_model, target_layer)

    for tag, filename in SELECTED_IMAGES.items():
        img_path = images_dir / filename
        if not img_path.exists():
            print(f"⚠️ {filename} پیدا نشد، رد می‌شود")
            continue

        img = Image.open(img_path).convert("RGB").resize((640, 640))
        img_np = np.array(img)
        tensor = torch.from_numpy(img_np).float().permute(2, 0, 1).unsqueeze(0) / 255.0
        tensor = tensor.to(device)

        cam = cam_engine(tensor)
        overlay = overlay_heatmap(img_np, cam, alpha=0.45)

        # پیش‌بینی واقعی برای نمایش کنار heatmap
        result = yolo.predict(source=img_np, imgsz=640, conf=0.3, verbose=False)[0]
        if len(result.boxes) > 0:
            import cv2
            box = result.boxes[0]
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            cls_id = int(box.cls[0])
            score = float(box.conf[0])
            cv2.rectangle(overlay, (x1, y1), (x2, y2), (255, 255, 255), 2)
            cv2.putText(overlay, f"{CLASS_NAMES[cls_id]} {score:.2f}", (x1, max(y1 - 5, 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            best_label, best_score = CLASS_NAMES[cls_id], score
        else:
            best_label, best_score = "هیچ", 0.0

        out_path = out_dir / f"yolo_{tag}.png"
        Image.fromarray(overlay).save(out_path)
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
