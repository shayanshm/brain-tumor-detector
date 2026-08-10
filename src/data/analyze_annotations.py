"""
Phase 2.4 (توزیع کلاس‌ها) + Phase 2.6 (کیفیت annotation).

نحوه‌ی اجرا در Kaggle:
    !python /kaggle/working/brain-tumor-detector/src/data/analyze_annotations.py --root "$DATASET_ROOT"
"""
import argparse
from collections import Counter, defaultdict
from pathlib import Path

CLASS_NAMES = ["glioma", "meningioma", "pituitary"]


def parse_label_file(path: Path):
    """
    هر خط را parse می‌کند و مشکلات احتمالی را برمی‌گرداند.
    فرمت مورد انتظار هر خط: class_id x_center y_center width height (همه نرمالایز 0..1)
    """
    boxes = []
    issues = []
    if not path.exists():
        return boxes, ["فایل annotation وجود ندارد"]

    text = path.read_text().strip()
    if not text:
        return boxes, []  # فایل خالی = تصویر بدون تومور (منفی)؛ خطا نیست

    for line_no, line in enumerate(text.splitlines(), start=1):
        parts = line.strip().split()
        if len(parts) != 5:
            issues.append(f"خط {line_no}: تعداد ستون‌ها {len(parts)} است (انتظار: 5)")
            continue
        try:
            cls_id = int(parts[0])
            x, y, w, h = (float(v) for v in parts[1:])
        except ValueError:
            issues.append(f"خط {line_no}: مقدار غیرعددی")
            continue

        if not (0 <= cls_id < len(CLASS_NAMES)):
            issues.append(f"خط {line_no}: class_id={cls_id} خارج از بازه‌ی مجاز [0,{len(CLASS_NAMES)-1}]")
        if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0):
            issues.append(f"خط {line_no}: مرکز بیرون از بازه [0,1] -> x={x}, y={y}")
        if not (0.0 < w <= 1.0 and 0.0 < h <= 1.0):
            issues.append(f"خط {line_no}: عرض/ارتفاع نامعتبر -> w={w}, h={h}")
        # چک سرریز از مرز تصویر (bbox باید کاملا داخل [0,1] بماند)
        if x - w / 2 < -1e-6 or x + w / 2 > 1 + 1e-6 or y - h / 2 < -1e-6 or y + h / 2 > 1 + 1e-6:
            issues.append(f"خط {line_no}: bbox از مرز تصویر بیرون می‌زند")

        boxes.append((cls_id, x, y, w, h))

    return boxes, issues


def analyze(root: str) -> dict:
    root_path = Path(root)
    report = {}

    for split in ["train", "valid", "test"]:
        lbl_dir = root_path / split / "labels"
        if not lbl_dir.exists():
            continue

        class_counts = Counter()
        boxes_per_image = []
        multi_class_images = 0
        total_issues = defaultdict(list)

        label_files = sorted(lbl_dir.glob("*.txt"))
        for lf in label_files:
            boxes, issues = parse_label_file(lf)
            if issues:
                total_issues[lf.name] = issues
            boxes_per_image.append(len(boxes))
            classes_in_image = {b[0] for b in boxes}
            if len(classes_in_image) > 1:
                multi_class_images += 1
            for cls_id, *_ in boxes:
                if 0 <= cls_id < len(CLASS_NAMES):
                    class_counts[CLASS_NAMES[cls_id]] += 1

        n_files = len(label_files)
        avg_boxes = sum(boxes_per_image) / n_files if n_files else 0

        print(f"\n== {split} ==")
        print(f"  تعداد فایل annotation: {n_files}")
        print(f"  توزیع کلاس‌ها (تعداد bbox): {dict(class_counts)}")
        print(f"  میانگین bbox به‌ازای هر تصویر: {avg_boxes:.3f}")
        print(f"  تصاویر چندکلاسه (بیش از یک نوع تومور در یک تصویر): {multi_class_images}")
        print(f"  فایل‌های دارای مشکل: {len(total_issues)} از {n_files}")
        if total_issues:
            sample = list(total_issues.items())[:3]
            for fname, issues in sample:
                print(f"    - {fname}: {issues}")

        report[split] = {
            "n_files": n_files,
            "class_counts": dict(class_counts),
            "avg_boxes_per_image": avg_boxes,
            "multi_class_images": multi_class_images,
            "n_files_with_issues": len(total_issues),
        }

    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    args = parser.parse_args()
    analyze(args.root)
