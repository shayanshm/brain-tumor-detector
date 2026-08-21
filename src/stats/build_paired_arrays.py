"""
Phase 8.1 — ساخت آرایه‌ی زوجی per-image (IoU و correctness) برای هر دو مدل.
چون هر تصویر test دقیقاً ۱ annotation دارد (تایید Phase 2)، این آرایه‌ها خیلی مستقیم‌اند:
برای هر تصویر i: fr_iou[i], fr_correct[i], yolo_iou[i], yolo_correct[i]

اجرا (بدون نیاز به GPU):
    python src/stats/build_paired_arrays.py \\
        --gt data/processed/coco_format/test.json \\
        --pred-fr outputs_v3/logs/faster_rcnn_test_predictions.json \\
        --pred-yolo outputs/yolo11/run1/yolo_test_predictions.json \\
        --out reports/tables/phase8_paired_arrays.csv
"""
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eval.compare_models import box_iou, xywh_to_xyxy  # noqa: E402


def best_match_for_image(gts, preds, conf_threshold=0.5):
    """
    gts: [(category_id, bbox), ...] (برای این دیتاست همیشه طول ۱)
    preds: لیست پیش‌بینی‌های همین تصویر (همه‌ی کلاس‌ها)
    برمی‌گرداند: (best_iou, correct) برای بهترین match با کلاس درست
    """
    preds = [p for p in preds if p["score"] >= conf_threshold]
    preds = sorted(preds, key=lambda p: -p["score"])

    best_iou_correct_class = 0.0
    for cls_id, gt_box in gts:
        for p in preds:
            if p["category_id"] != cls_id:
                continue
            iou = box_iou(xywh_to_xyxy(p["bbox"]), xywh_to_xyxy(gt_box))
            if iou > best_iou_correct_class:
                best_iou_correct_class = iou

    correct = best_iou_correct_class >= 0.5
    return best_iou_correct_class, correct


def main(args):
    with open(args.gt) as f:
        gt_data = json.load(f)
    with open(args.pred_fr) as f:
        fr_preds = json.load(f)
    with open(args.pred_yolo) as f:
        yolo_preds = json.load(f)

    id_to_filename = {img["id"]: img["file_name"] for img in gt_data["images"]}
    categories = {c["id"]: c["name"] for c in gt_data["categories"]}

    gt_by_image = defaultdict(list)
    for ann in gt_data["annotations"]:
        gt_by_image[ann["image_id"]].append((ann["category_id"], ann["bbox"]))

    fr_by_image = defaultdict(list)
    for p in fr_preds:
        fr_by_image[p["image_id"]].append(p)
    yolo_by_image = defaultdict(list)
    for p in yolo_preds:
        yolo_by_image[p["image_id"]].append(p)

    rows = []
    for image_id, gts in sorted(gt_by_image.items()):
        if len(gts) != 1:
            print(f"⚠️ تصویر {image_id} تعداد annotation غیرمنتظره دارد: {len(gts)} (فرض ۱ نقض شد)")
        gt_class_id = gts[0][0]

        fr_iou, fr_correct = best_match_for_image(gts, fr_by_image.get(image_id, []))
        yolo_iou, yolo_correct = best_match_for_image(gts, yolo_by_image.get(image_id, []))

        rows.append({
            "image_id": image_id,
            "file_name": id_to_filename[image_id],
            "gt_class": categories[gt_class_id],
            "fr_iou": round(fr_iou, 4),
            "fr_correct": int(fr_correct),
            "yolo_iou": round(yolo_iou, 4),
            "yolo_correct": int(yolo_correct),
        })

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    n = len(rows)
    fr_acc = sum(r["fr_correct"] for r in rows) / n
    yolo_acc = sum(r["yolo_correct"] for r in rows) / n
    print(f"تعداد تصویر: {n}")
    print(f"Faster R-CNN: accuracy(IoU>=0.5)={fr_acc:.4f}, میانگین IoU={sum(r['fr_iou'] for r in rows)/n:.4f}")
    print(f"YOLO11:       accuracy(IoU>=0.5)={yolo_acc:.4f}, میانگین IoU={sum(r['yolo_iou'] for r in rows)/n:.4f}")
    print(f"ذخیره شد: {args.out}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt", required=True)
    parser.add_argument("--pred-fr", required=True)
    parser.add_argument("--pred-yolo", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    main(args)
