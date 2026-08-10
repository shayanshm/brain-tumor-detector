"""
Phase 3.5 (بخش دوم) — ارزیابی mAP@0.5, mAP@0.5:0.95, Precision, Recall با pycocotools.
مستقل از فریمورک است؛ هم برای Faster R-CNN (Phase 3) و هم YOLO11 (Phase 4) قابل استفاده است،
تا مقایسه‌ی Phase 5 دقیقاً apple-to-apple باشد.
"""
import io
import json
from contextlib import redirect_stdout

import torch
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


@torch.no_grad()
def run_inference(model, dataloader, device, score_threshold: float = 0.05):
    """
    خروجی: لیستی از پیش‌بینی‌ها در فرمت استاندارد COCO detection results:
    [{"image_id":..., "category_id":..., "bbox":[x,y,w,h], "score":...}, ...]
    توجه: category_id در اینجا -1 می‌شود تا با category_id اصلی COCO json (0,1,2) هم‌تراز شود
    (چون در آموزش +1 اضافه کرده بودیم برای رزرو background).
    """
    model.eval()
    model.to(device)
    results = []

    for images, targets in dataloader:
        images = [img.to(device) for img in images]
        outputs = model(images)

        for target, output in zip(targets, outputs):
            image_id = int(target["image_id"].item())
            boxes = output["boxes"].cpu().numpy()
            scores = output["scores"].cpu().numpy()
            labels = output["labels"].cpu().numpy()

            for box, score, label in zip(boxes, scores, labels):
                if score < score_threshold:
                    continue
                x1, y1, x2, y2 = box
                results.append({
                    "image_id": image_id,
                    "category_id": int(label) - 1,  # برگرداندن به فضای category_id اصلی COCO (0,1,2)
                    "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                    "score": float(score),
                })

    return results


def evaluate_coco(gt_json_path: str, predictions: list) -> dict:
    coco_gt = COCO(gt_json_path)

    if not predictions:
        print("⚠️ هیچ پیش‌بینی‌ای برای ارزیابی وجود ندارد (لیست خالی).")
        return {}

    # pycocotools برای loadRes نیاز به یک فایل json موقت یا لیست دیکشنری دارد
    with redirect_stdout(io.StringIO()):  # سرکوب پرینت‌های پرحجم داخلی pycocotools
        coco_dt = coco_gt.loadRes(predictions)
        coco_eval = COCOeval(coco_gt, coco_dt, iouType="bbox")
        coco_eval.evaluate()
        coco_eval.accumulate()

    coco_eval.summarize()  # این یکی را عمداً چاپ می‌کنیم چون خلاصه‌ی استاندارد و مفید COCO است

    metrics = {
        "mAP@0.5:0.95": coco_eval.stats[0],
        "mAP@0.5": coco_eval.stats[1],
        "mAP@0.75": coco_eval.stats[2],
        "AR@100": coco_eval.stats[8],
    }
    return metrics


if __name__ == "__main__":
    import argparse
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from models.coco_dataset import CocoDetectionDataset, collate_fn, NUM_CLASSES_WITH_BACKGROUND
    from models.faster_rcnn import build_faster_rcnn
    from torch.utils.data import DataLoader

    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--split", default="test", choices=["valid", "test"])
    parser.add_argument("--checkpoint", required=True)
    args = parser.parse_args()

    data_root = Path(args.data_root)
    ds = CocoDetectionDataset(
        images_dir=str(data_root / "yolo_detection" / args.split / "images"),
        coco_json_path=str(data_root / "coco_format" / f"{args.split}.json"),
    )
    loader = DataLoader(ds, batch_size=4, shuffle=False, collate_fn=collate_fn)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_faster_rcnn(num_classes=NUM_CLASSES_WITH_BACKGROUND, pretrained=False)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))

    predictions = run_inference(model, loader, device)
    metrics = evaluate_coco(str(data_root / "coco_format" / f"{args.split}.json"), predictions)
    print("\n== خلاصه ==")
    print(json.dumps(metrics, indent=2))
