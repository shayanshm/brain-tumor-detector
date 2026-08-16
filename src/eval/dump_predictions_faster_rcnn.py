"""
Phase 5.1 (بخش ۱) — ذخیره‌ی خام پیش‌بینی‌های Faster R-CNN روی test set به فرمت COCO detection results.
این خروجی (JSON سبک) بعداً بدون نیاز به GPU در compare_models.py مصرف می‌شود.

نحوه‌ی اجرا در Kaggle:
    !python /kaggle/working/brain-tumor-detector/src/eval/dump_predictions_faster_rcnn.py \\
        --data-root /kaggle/working/brain-tumor-detector/data/processed \\
        --checkpoint /kaggle/working/brain-tumor-detector/outputs_v3/checkpoints/faster_rcnn_epoch50.pt \\
        --out /kaggle/working/brain-tumor-detector/outputs_v3/logs/faster_rcnn_test_predictions.json
"""
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from models.coco_dataset import CocoDetectionDataset, collate_fn, NUM_CLASSES_WITH_BACKGROUND  # noqa: E402
from models.faster_rcnn import build_faster_rcnn  # noqa: E402
from eval.eval_coco import run_inference  # noqa: E402


def main(args):
    data_root = Path(args.data_root)
    ds = CocoDetectionDataset(
        images_dir=str(data_root / "yolo_detection" / "test" / "images"),
        coco_json_path=str(data_root / "coco_format" / "test.json"),
    )
    loader = DataLoader(ds, batch_size=4, shuffle=False, collate_fn=collate_fn)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_faster_rcnn(num_classes=NUM_CLASSES_WITH_BACKGROUND, pretrained=False)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))

    predictions = run_inference(model, loader, device, score_threshold=0.05)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(predictions, f)

    print(f"تعداد پیش‌بینی (قبل از فیلتر نهایی): {len(predictions)}")
    print(f"ذخیره شد: {args.out}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    main(args)
