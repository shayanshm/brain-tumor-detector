"""
Phase 5.1-5.3 — مقایسه‌ی apples-to-apples دو مدل با محاسبه‌ی دستی و یکسان
Precision/Recall/F1/IoU/Dice (چیزی که COCOeval/Ultralytics هرکدام جدا و ناهمسان گزارش می‌دهند).
mAP@0.5 / mAP@0.5:0.95 از مقادیر استاندارد از‌پیش‌محاسبه‌شده (COCOeval/Ultralytics) خوانده می‌شود
چون آن‌ها خودشان از قبل روش‌شناسی استاندارد و قابل‌اعتمادی دارند.

ورودی: دو فایل JSON پیش‌بینی (خروجی dump_predictions_faster_rcnn.py و dump_predictions_yolo.py)
+ ground truth test.json مشترک.

اجرا (بدون نیاز به GPU، قابل اجرا هر جا):
    python src/eval/compare_models.py \\
        --gt data/processed/coco_format/test.json \\
        --pred-a outputs_v3/logs/faster_rcnn_test_predictions.json --name-a "Faster R-CNN" \\
        --pred-b outputs/yolo11/run1/yolo_test_predictions.json --name-b "YOLO11" \\
        --out reports/tables/phase5_comparison.json
"""
import json
from collections import defaultdict


def box_iou(box_a, box_b):
    """box: [x1,y1,x2,y2] (absolute pixel). برمی‌گرداند IoU اسکالر."""
    xa1, ya1, xa2, ya2 = box_a
    xb1, yb1, xb2, yb2 = box_b
    ix1, iy1 = max(xa1, xb1), max(ya1, yb1)
    ix2, iy2 = min(xa2, xb2), min(ya2, yb2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, xa2 - xa1) * max(0.0, ya2 - ya1)
    area_b = max(0.0, xb2 - xb1) * max(0.0, yb2 - yb1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def xywh_to_xyxy(box):
    x, y, w, h = box
    return [x, y, x + w, y + h]


def match_and_score(gt_by_image_class, preds, iou_threshold=0.5):
    """
    تطبیق حریصانه (greedy) استاندارد: پیش‌بینی‌ها بر اساس confidence نزولی مرتب می‌شوند،
    هرکدام به بهترین GT هم‌کلاس و تطبیق‌نشده با IoU>=threshold وصل می‌شود.
    برمی‌گرداند: TP, FP, FN, لیست IoU های matched (برای میانگین IoU/Dice)
    """
    preds_sorted = sorted(preds, key=lambda p: -p["score"])
    matched_gt = defaultdict(set)  # (image_id, category_id) -> set of matched gt indices
    tp, fp = 0, 0
    matched_ious = []

    for p in preds_sorted:
        key = (p["image_id"], p["category_id"])
        gts = gt_by_image_class.get(key, [])
        best_iou, best_idx = 0.0, -1
        for idx, gt_box in enumerate(gts):
            if idx in matched_gt[key]:
                continue
            iou = box_iou(xywh_to_xyxy(p["bbox"]), xywh_to_xyxy(gt_box))
            if iou > best_iou:
                best_iou, best_idx = iou, idx

        if best_iou >= iou_threshold:
            tp += 1
            matched_gt[key].add(best_idx)
            matched_ious.append(best_iou)
        else:
            fp += 1

    n_total_gt = sum(len(v) for v in gt_by_image_class.values())
    fn = n_total_gt - tp

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    mean_iou = sum(matched_ious) / len(matched_ious) if matched_ious else 0.0
    mean_dice = (2 * mean_iou / (1 + mean_iou)) if matched_ious else 0.0
    # میانگین Dice per-detection (دقیق‌تر از فرمول روی میانگین IoU):
    dices = [2 * iou / (1 + iou) for iou in matched_ious]
    mean_dice_per_detection = sum(dices) / len(dices) if dices else 0.0

    return {
        "tp": tp, "fp": fp, "fn": fn, "n_gt": n_total_gt,
        "precision": round(precision, 4), "recall": round(recall, 4), "f1": round(f1, 4),
        "mean_iou": round(mean_iou, 4),
        "mean_dice": round(mean_dice_per_detection, 4),
    }


def build_gt_index(coco_gt):
    gt_by_image_class = defaultdict(list)
    for ann in coco_gt["annotations"]:
        key = (ann["image_id"], ann["category_id"])
        x, y, w, h = ann["bbox"]
        gt_by_image_class[key].append([x, y, w, h])
    return gt_by_image_class


def evaluate_predictions(gt_path: str, pred_path: str, iou_threshold: float = 0.5,
                          conf_threshold: float = 0.0) -> dict:
    with open(gt_path) as f:
        coco_gt = json.load(f)
    with open(pred_path) as f:
        preds = json.load(f)

    if conf_threshold > 0:
        preds = [p for p in preds if p["score"] >= conf_threshold]

    gt_index = build_gt_index(coco_gt)
    overall = match_and_score(gt_index, preds, iou_threshold)

    # per-class هم محاسبه می‌کنیم (مفید برای گزارش)
    categories = {c["id"]: c["name"] for c in coco_gt["categories"]}
    per_class = {}
    for cat_id, cat_name in categories.items():
        gt_c = {k: v for k, v in gt_index.items() if k[1] == cat_id}
        preds_c = [p for p in preds if p["category_id"] == cat_id]
        per_class[cat_name] = match_and_score(gt_c, preds_c, iou_threshold)

    return {"overall": overall, "per_class": per_class}


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--gt", required=True)
    parser.add_argument("--pred-a", required=True)
    parser.add_argument("--name-a", default="Model A")
    parser.add_argument("--pred-b", required=True)
    parser.add_argument("--name-b", default="Model B")
    parser.add_argument("--out", required=True)
    parser.add_argument("--conf-threshold", type=float, default=0.0,
                         help="فیلتر پیش‌بینی‌های کم‌اطمینان قبل از محاسبه (پیشنهاد: 0.5 برای نقطه‌ی عملیاتی واقع‌بینانه)")
    args = parser.parse_args()

    result_a = evaluate_predictions(args.gt, args.pred_a, conf_threshold=args.conf_threshold)
    result_b = evaluate_predictions(args.gt, args.pred_b, conf_threshold=args.conf_threshold)

    print(f"== {args.name_a} (overall, IoU>=0.5) ==")
    print(result_a["overall"])
    print(f"\n== {args.name_b} (overall, IoU>=0.5) ==")
    print(result_b["overall"])

    output = {args.name_a: result_a, args.name_b: result_b}
    with open(args.out, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nذخیره شد: {args.out}")
