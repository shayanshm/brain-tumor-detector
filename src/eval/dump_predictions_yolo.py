"""
Phase 5.1 (بخش ۲) — ذخیره‌ی خام پیش‌بینی‌های YOLO11 روی test set به همان فرمت COCO detection results
که برای Faster R-CNN استفاده شد (dump_predictions_faster_rcnn.py) -- تا compare_models.py هر دو را
با دقیقاً یک روش مصرف کند.

نحوه‌ی اجرا در Kaggle:
    !python /kaggle/working/brain-tumor-detector/src/eval/dump_predictions_yolo.py \\
        --data-root /kaggle/working/brain-tumor-detector/data/processed \\
        --checkpoint /path/to/run1/weights/best.pt \\
        --out /kaggle/working/brain-tumor-detector/outputs/yolo11/run1/yolo_test_predictions.json
"""
import json
from pathlib import Path


def main(args):
    from ultralytics import YOLO

    data_root = Path(args.data_root)
    coco_json_path = data_root / "coco_format" / "test.json"
    with open(coco_json_path) as f:
        coco_gt = json.load(f)

    # نگاشت file_name -> image_id (دقیقاً همان id هایی که در COCO json مرجع Faster R-CNN هم استفاده شد)
    filename_to_id = {img["file_name"]: img["id"] for img in coco_gt["images"]}

    images_dir = data_root / "yolo_detection" / "test" / "images"
    model = YOLO(args.checkpoint)

    predictions = []
    results = model.predict(source=str(images_dir), imgsz=640, conf=0.05, verbose=False, stream=True)
    for r in results:
        file_name = Path(r.path).name
        image_id = filename_to_id.get(file_name)
        if image_id is None:
            continue  # نباید اتفاق بیفتد، ولی محکم‌کاری می‌کنیم
        boxes = r.boxes
        for i in range(len(boxes)):
            x1, y1, x2, y2 = boxes.xyxy[i].tolist()
            score = float(boxes.conf[i])
            cls_id = int(boxes.cls[i])  # کلاس‌های YOLO (0,1,2) مستقیماً معادل category_id COCO ما هستند
            predictions.append({
                "image_id": image_id,
                "category_id": cls_id,
                "bbox": [x1, y1, x2 - x1, y2 - y1],
                "score": score,
            })

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(predictions, f)

    print(f"تعداد پیش‌بینی: {len(predictions)}")
    print(f"ذخیره شد: {args.out}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    main(args)
