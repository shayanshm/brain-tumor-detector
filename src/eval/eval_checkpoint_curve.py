"""
Phase 3.5 (تکمیلی) — بازسازی منحنی mAP/Precision/Recall روی validation set
با ارزیابی تمام چک‌پوینت‌های ذخیره‌شده (هر 5 epoch)، بدون نیاز به آموزش مجدد.

نحوه‌ی اجرا در Kaggle:
    !python /kaggle/working/brain-tumor-detector/src/eval/eval_checkpoint_curve.py \\
        --data-root /kaggle/working/brain-tumor-detector/data/processed \\
        --ckpt-dir /kaggle/working/brain-tumor-detector/outputs/checkpoints \\
        --epochs 5,10,15,20,25,30,35,40,45,50 \\
        --out /kaggle/working/brain-tumor-detector/outputs/logs/faster_rcnn_val_curve.csv
"""
import argparse
import csv
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from models.coco_dataset import CocoDetectionDataset, collate_fn, NUM_CLASSES_WITH_BACKGROUND  # noqa: E402
from models.faster_rcnn import build_faster_rcnn  # noqa: E402
from eval.eval_coco import run_inference, evaluate_coco  # noqa: E402


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_root = Path(args.data_root)
    ckpt_dir = Path(args.ckpt_dir)

    val_ds = CocoDetectionDataset(
        images_dir=str(data_root / "yolo_detection" / "valid" / "images"),
        coco_json_path=str(data_root / "coco_format" / "valid.json"),
    )
    val_loader = DataLoader(val_ds, batch_size=8, shuffle=False, collate_fn=collate_fn)
    gt_json = str(data_root / "coco_format" / "valid.json")

    epochs_to_eval = [int(e) for e in args.epochs.split(",")]
    rows = []

    for epoch in epochs_to_eval:
        ckpt_path = ckpt_dir / f"faster_rcnn_epoch{epoch}.pt"
        if not ckpt_path.exists():
            print(f"⚠️ چک‌پوینت epoch {epoch} پیدا نشد ({ckpt_path})، رد می‌شود.")
            continue

        print(f"\n== ارزیابی epoch {epoch} ==")
        model = build_faster_rcnn(num_classes=NUM_CLASSES_WITH_BACKGROUND, pretrained=False)
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        model.to(device)

        predictions = run_inference(model, val_loader, device)
        metrics = evaluate_coco(gt_json, predictions)

        row = {"epoch": epoch, **metrics}
        rows.append(row)
        print(f"  epoch {epoch}: {metrics}")

        del model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        if rows:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    print(f"\nذخیره شد: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--ckpt-dir", required=True)
    parser.add_argument("--epochs", default="5,10,15,20,25,30,35,40,45,50")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    main(args)
