"""
Phase 2.1 / 2.2 — بازرسی ساختار خام دیتاست + شمارش واقعی تصاویر/annotation ها.
این اسکریپت هیچ فرض قطعی‌ای درباره‌ی زیرپوشه‌ها نمی‌کند؛ ابتدا کاوش می‌کند و گزارش می‌دهد.

نحوه‌ی اجرا در Kaggle Notebook:
    DATASET_ROOT = "/kaggle/input/datasets/pkdarabi/medical-image-dataset-brain-tumor-detection/BrainTumor/BrainTumorYolov11"
    !python /kaggle/working/brain-tumor-detector/src/data/inspect_dataset.py --root "$DATASET_ROOT"

یا مستقیم در یک سلول با فراخوانی تابع inspect(root).
"""
import argparse
import sys
from pathlib import Path

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def inspect(root: str) -> dict:
    root_path = Path(root)
    if not root_path.exists():
        print(f"❌ مسیر داده‌شده وجود ندارد: {root_path}")
        sys.exit(1)

    print(f"== بازرسی ساختار: {root_path} ==\n")

    # سطح اول: هر چه داخل root هست
    top_level = sorted(p.name for p in root_path.iterdir())
    print("محتویات سطح اول:", top_level)

    yaml_path = root_path / "data.yaml"
    if yaml_path.exists():
        print("\n-- محتوای data.yaml --")
        print(yaml_path.read_text())

    report = {}
    for split in ["train", "valid", "test"]:
        split_dir = root_path / split
        if not split_dir.exists():
            print(f"\n⚠️ پوشه‌ی '{split}' پیدا نشد.")
            continue

        sub_items = sorted(p.name for p in split_dir.iterdir())
        print(f"\n-- {split}/  (زیرپوشه‌ها/فایل‌ها: {sub_items}) --")

        # حالت رایج: split/images و split/labels
        img_dir = split_dir / "images"
        lbl_dir = split_dir / "labels"

        if img_dir.exists() and lbl_dir.exists():
            images = sorted(p for p in img_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS)
            labels = sorted(p for p in lbl_dir.iterdir() if p.suffix == ".txt")
            image_stems = {p.stem for p in images}
            label_stems = {p.stem for p in labels}
        else:
            # حالت جایگزین: تصاویر و لیبل‌ها مستقیم داخل split/ مخلوط‌اند
            images = sorted(p for p in split_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS)
            labels = sorted(p for p in split_dir.iterdir() if p.suffix == ".txt")
            image_stems = {p.stem for p in images}
            label_stems = {p.stem for p in labels}

        missing_labels = image_stems - label_stems
        missing_images = label_stems - image_stems
        empty_labels = [p for p in labels if p.stat().st_size == 0]

        print(f"  تعداد تصاویر: {len(images)}")
        print(f"  تعداد فایل‌های annotation (.txt): {len(labels)}")
        print(f"  تصاویر بدون annotation متناظر: {len(missing_labels)}")
        print(f"  annotation بدون تصویر متناظر: {len(missing_images)}")
        print(f"  annotation های خالی (بدون هیچ bbox — تصویر منفی/no-tumor): {len(empty_labels)}")
        if images:
            print(f"  نمونه نام فایل: {images[0].name}")

        report[split] = {
            "n_images": len(images),
            "n_labels": len(labels),
            "missing_labels": len(missing_labels),
            "missing_images": len(missing_images),
            "empty_labels": len(empty_labels),
        }

    print("\n== خلاصه ==")
    for split, r in report.items():
        print(f"  {split}: {r}")

    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, help="مسیر پوشه‌ی BrainTumorYolov11 (شامل data.yaml)")
    args = parser.parse_args()
    inspect(args.root)
