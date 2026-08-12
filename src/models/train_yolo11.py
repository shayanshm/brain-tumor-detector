"""
Phase 4.1-4.3 — آموزش YOLO11 با شرایط یکسان با Faster R-CNN (Phase 3.3، پس از اصلاح باگ augmentation).

طبق سند اصلی: "Do NOT modify: dataset / augmentation / optimizer. Maintain identical training conditions."
augmentation دقیقاً منطبق با Faster R-CNN تنظیم شده: Horizontal Flip (p=0.5) + Brightness jitter (~0.2).
محدودیت فنی مستندشده: Ultralytics معادل مستقیمی برای "Contrast" ندارد (فقط hsv_v/s/h)؛ بنابراین
اثر brightness+contrast با hsv_v تقریب زده می‌شود -- این تفاوت جزئی و مستند است، نه یک انحراف پنهان.
همه‌ی augmentation های هندسی/رنگی دیگر (mosaic, mixup, rotation, shear, hue, saturation) خاموش‌اند
چون در Faster R-CNN هم وجود نداشتند.

نحوه‌ی اجرا در Kaggle (حتماً با Save & Run All / Commit، طبق CONTINUITY.md):
    !python /kaggle/working/brain-tumor-detector/src/models/train_yolo11.py \\
        --data /kaggle/working/brain-tumor-detector/data/processed/yolo_detection/data.yaml \\
        --project /kaggle/working/brain-tumor-detector/outputs/yolo11
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.seed import set_seed  # noqa: E402

# --- شرایط منجمد Phase 3.3، با augmentation منطبق‌شده با Faster R-CNN (نسخه‌ی اصلاح‌شده) ---
FROZEN_TRAIN_ARGS = dict(
    epochs=50,
    imgsz=640,
    batch=8,
    optimizer="SGD",
    lr0=0.005,
    momentum=0.9,
    weight_decay=0.0005,
    # --- augmentation: دقیقاً همان چیزی که در Faster R-CNN (نسخه اصلاح‌شده) اعمال می‌شود ---
    fliplr=0.5,       # Horizontal Flip p=0.5 -- مطابق DetectionAugmentation
    hsv_v=0.2,         # تقریب Brightness jitter ±20% (نزدیک‌ترین معادل بومی Ultralytics)
    flipud=0.0, hsv_h=0.0, hsv_s=0.0,
    mosaic=0.0, mixup=0.0, copy_paste=0.0,
    degrees=0.0, translate=0.0, scale=0.0, shear=0.0, perspective=0.0,
    seed=42,
)


def main(args):
    set_seed(42)
    from ultralytics import YOLO

    model = YOLO(args.model)

    resume_ckpt = Path(args.project) / args.name / "weights" / "last.pt"
    if resume_ckpt.exists():
        print(f"⏯️  چک‌پوینت قبلی پیدا شد ({resume_ckpt}) -- Ultralytics خودش resume می‌کند")
        model = YOLO(str(resume_ckpt))
        results = model.train(resume=True)
    else:
        print("اجرای تازه -- شروع از epoch 0")
        results = model.train(
            data=args.data,
            project=args.project,
            name=args.name,
            **FROZEN_TRAIN_ARGS,
        )

    print("آموزش تمام شد.")
    print(f"نتایج/چک‌پوینت‌ها: {args.project}/{args.name}/")
    print("results.csv شامل precision/recall/mAP50/mAP50-95 برای هر epoch است (بدون نیاز به بازسازی).")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="مسیر data.yaml")
    parser.add_argument("--model", default="yolo11n.pt", help="نسخه‌ی پایه YOLO11 (n/s/m/l/x)")
    parser.add_argument("--project", required=True)
    parser.add_argument("--name", default="run1")
    args = parser.parse_args()
    main(args)
