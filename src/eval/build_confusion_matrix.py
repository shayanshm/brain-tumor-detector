"""
ساخت ماتریس درهم‌ریختگی (شامل background برای FP/FN) از پیش‌بینی‌های خام + ground truth،
با همان منطق matching که در compare_models.py تست و تایید شد. قابل استفاده برای هر دو مدل
(رفع نقص: Faster R-CNN تا الان confusion matrix تصویری مثل YOLO نداشت).
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from compare_models import box_iou, xywh_to_xyxy  # noqa: E402


def build_confusion_matrix(gt_path: str, pred_path: str, class_names: list,
                            iou_threshold: float = 0.5, conf_threshold: float = 0.5):
    with open(gt_path) as f:
        coco_gt = json.load(f)
    with open(pred_path) as f:
        preds = json.load(f)
    preds = [p for p in preds if p["score"] >= conf_threshold]

    gt_by_image = defaultdict(list)  # image_id -> [(category_id, bbox), ...]
    for ann in coco_gt["annotations"]:
        gt_by_image[ann["image_id"]].append((ann["category_id"], ann["bbox"]))

    preds_by_image = defaultdict(list)
    for p in preds:
        preds_by_image[p["image_id"]].append(p)

    n = len(class_names)
    # ماتریس (n+1) x (n+1) -- سطر/ستون آخر = background
    matrix = [[0] * (n + 1) for _ in range(n + 1)]
    bg = n

    for image_id in set(list(gt_by_image.keys()) + list(preds_by_image.keys())):
        gts = list(gt_by_image.get(image_id, []))
        image_preds = sorted(preds_by_image.get(image_id, []), key=lambda p: -p["score"])
        matched_gt = set()

        for p in image_preds:
            best_iou, best_idx, best_cls = 0.0, -1, None
            for idx, (cls_id, gt_box) in enumerate(gts):
                if idx in matched_gt:
                    continue
                iou = box_iou(xywh_to_xyxy(p["bbox"]), xywh_to_xyxy(gt_box))
                if iou > best_iou:
                    best_iou, best_idx, best_cls = iou, idx, cls_id

            if best_iou >= iou_threshold:
                matched_gt.add(best_idx)
                matrix[p["category_id"]][best_cls] += 1  # [predicted][true]
            else:
                matrix[p["category_id"]][bg] += 1  # False Positive (چیزی پیش‌بینی شد که نبود)

        for idx, (cls_id, _) in enumerate(gts):
            if idx not in matched_gt:
                matrix[bg][cls_id] += 1  # False Negative (چیزی بود که تشخیص داده نشد)

    return matrix


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt", required=True)
    parser.add_argument("--pred", required=True)
    parser.add_argument("--out-json", required=True)
    args = parser.parse_args()

    class_names = ["glioma", "meningioma", "pituitary"]
    m = build_confusion_matrix(args.gt, args.pred, class_names)
    with open(args.out_json, "w") as f:
        json.dump({"matrix": m, "labels": class_names + ["background"]}, f, indent=2)
    print("ماتریس:", m)
