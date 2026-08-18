"""
Phase 6.2 — اجرای YOLO11 روی تصویر اصلی + همه‌ی ۸ نوع corruption.

نحوه‌ی اجرا در Kaggle:
    !python /kaggle/working/brain-tumor-detector/src/robustness/run_robustness_yolo.py \\
        --data-root /kaggle/working/brain-tumor-detector/data/processed \\
        --checkpoint /path/to/run1/weights/best.pt \\
        --out-dir /kaggle/working/brain-tumor-detector/outputs/yolo11/run1/robustness
"""
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from robustness.corruptions import CORRUPTIONS  # noqa: E402


def run_on_condition(model, images_dir: Path, filename_to_id: dict, corrupt_fn=None,
                      conf_threshold: float = 0.05):
    predictions = []
    image_files = sorted(images_dir.glob("*.jpg")) + sorted(images_dir.glob("*.png"))

    for img_path in image_files:
        image_id = filename_to_id.get(img_path.name)
        if image_id is None:
            continue
        img = Image.open(img_path).convert("RGB")
        if corrupt_fn is not None:
            img = corrupt_fn(img)

        result = model.predict(source=np.array(img), imgsz=640, conf=conf_threshold, verbose=False)[0]
        boxes = result.boxes
        for i in range(len(boxes)):
            x1, y1, x2, y2 = boxes.xyxy[i].tolist()
            score = float(boxes.conf[i])
            cls_id = int(boxes.cls[i])
            predictions.append({
                "image_id": image_id,
                "category_id": cls_id,
                "bbox": [x1, y1, x2 - x1, y2 - y1],
                "score": score,
            })
    return predictions


def main(args):
    from ultralytics import YOLO

    data_root = Path(args.data_root)
    with open(data_root / "coco_format" / "test.json") as f:
        coco_gt = json.load(f)
    filename_to_id = {img["file_name"]: img["id"] for img in coco_gt["images"]}

    images_dir = data_root / "yolo_detection" / "test" / "images"
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(args.checkpoint)

    conditions = {"original": None, **CORRUPTIONS}
    for name, fn in conditions.items():
        print(f"== در حال اجرا: {name} ==")
        preds = run_on_condition(model, images_dir, filename_to_id, corrupt_fn=fn)
        out_path = out_dir / f"yolo_{name}.json"
        with open(out_path, "w") as f:
            json.dump(preds, f)
        print(f"  {len(preds)} پیش‌بینی -> {out_path}")

    print("\nهمه‌ی ۹ حالت تمام شد.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()
    main(args)
