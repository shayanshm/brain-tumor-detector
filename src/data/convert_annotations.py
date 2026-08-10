"""
Phase 2.7 — تبدیل annotation ها به دو فرمت موازی:
  1) COCO JSON  -> برای آموزش/ارزیابی Faster R-CNN (torchvision) با pycocotools
  2) YOLO detection خالص (5 ستون bbox، نه polygon) -> برای بازآموزی YOLO11 در حالت Object Detection

ورودی خام: split/images/*.jpg + split/labels/*.txt (فرمت polygon، طبق کشف Phase 2.4/2.6)

نحوه‌ی اجرا در Kaggle:
    !python /kaggle/working/brain-tumor-detector/src/data/convert_annotations.py \\
        --root "$DATASET_ROOT" \\
        --out /kaggle/working/brain-tumor-detector/data/processed
"""
import argparse
import json
import shutil
from pathlib import Path

from PIL import Image

CLASS_NAMES = ["glioma", "meningioma", "pituitary"]


def parse_annotation_line(parts):
    """برمی‌گرداند (cls_id, (xc,yc,w,h)_normalized, polygon_points_or_None) یا None اگر نامعتبر باشد."""
    try:
        cls_id = int(parts[0])
        coords = [float(v) for v in parts[1:]]
    except ValueError:
        return None

    if not (0 <= cls_id < len(CLASS_NAMES)):
        return None

    if len(coords) == 4:
        xc, yc, w, h = coords
        return cls_id, (xc, yc, w, h), None
    elif len(coords) >= 6 and len(coords) % 2 == 0:
        pts = list(zip(coords[0::2], coords[1::2]))
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        x0, x1 = min(xs), max(xs)
        y0, y1 = min(ys), max(ys)
        xc, yc = (x0 + x1) / 2, (y0 + y1) / 2
        w, h = x1 - x0, y1 - y0
        return cls_id, (xc, yc, w, h), pts
    return None


def link_or_copy(src: Path, dst: Path) -> None:
    if dst.exists():
        return
    try:
        dst.symlink_to(src.resolve())
    except OSError:
        shutil.copy(src, dst)


def convert_split(root: Path, split: str, out_dir: Path) -> dict:
    img_dir = root / split / "images"
    lbl_dir = root / split / "labels"

    yolo_img_out = out_dir / "yolo_detection" / split / "images"
    yolo_lbl_out = out_dir / "yolo_detection" / split / "labels"
    yolo_img_out.mkdir(parents=True, exist_ok=True)
    yolo_lbl_out.mkdir(parents=True, exist_ok=True)

    images_list, annotations = [], []
    ann_id = 1
    n_skipped_lines = 0

    img_files = sorted(p for p in img_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})

    for img_id, img_path in enumerate(img_files, start=1):
        with Image.open(img_path) as im:
            w_img, h_img = im.size

        images_list.append({
            "id": img_id, "file_name": img_path.name, "width": w_img, "height": h_img,
        })
        link_or_copy(img_path, yolo_img_out / img_path.name)

        lbl_path = lbl_dir / f"{img_path.stem}.txt"
        yolo_lines = []
        if lbl_path.exists() and lbl_path.read_text().strip():
            for line in lbl_path.read_text().strip().splitlines():
                parts = line.split()
                if len(parts) < 5:
                    n_skipped_lines += 1
                    continue
                parsed = parse_annotation_line(parts)
                if parsed is None:
                    n_skipped_lines += 1
                    continue
                cls_id, (xc, yc, w, h), poly = parsed

                x_min = (xc - w / 2) * w_img
                y_min = (yc - h / 2) * h_img
                box_w = w * w_img
                box_h = h * h_img

                ann = {
                    "id": ann_id, "image_id": img_id, "category_id": cls_id,
                    "bbox": [round(x_min, 2), round(y_min, 2), round(box_w, 2), round(box_h, 2)],
                    "area": round(box_w * box_h, 2), "iscrowd": 0,
                }
                if poly:
                    seg_flat = [coord for pt in poly for coord in (pt[0] * w_img, pt[1] * h_img)]
                    ann["segmentation"] = [[round(c, 2) for c in seg_flat]]
                annotations.append(ann)
                ann_id += 1

                yolo_lines.append(f"{cls_id} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}")

        (yolo_lbl_out / f"{img_path.stem}.txt").write_text("\n".join(yolo_lines))

    coco = {
        "images": images_list,
        "annotations": annotations,
        "categories": [{"id": i, "name": n} for i, n in enumerate(CLASS_NAMES)],
    }
    coco_out = out_dir / "coco_format" / f"{split}.json"
    coco_out.parent.mkdir(parents=True, exist_ok=True)
    coco_out.write_text(json.dumps(coco))

    return {
        "split": split, "n_images": len(images_list), "n_annotations": len(annotations),
        "n_skipped_lines": n_skipped_lines, "coco_json": str(coco_out),
    }


def write_yolo_data_yaml(out_dir: Path) -> None:
    yaml_path = out_dir / "yolo_detection" / "data.yaml"
    content = (
        f"train: {(out_dir / 'yolo_detection' / 'train' / 'images').resolve()}\n"
        f"val: {(out_dir / 'yolo_detection' / 'valid' / 'images').resolve()}\n"
        f"test: {(out_dir / 'yolo_detection' / 'test' / 'images').resolve()}\n"
        f"nc: {len(CLASS_NAMES)}\n"
        f"names: {CLASS_NAMES}\n"
    )
    yaml_path.write_text(content)
    print(f"data.yaml نوشته شد: {yaml_path}")


def main(root: str, out: str) -> None:
    root_path = Path(root)
    out_path = Path(out)
    summary = []
    for split in ["train", "valid", "test"]:
        if not (root_path / split).exists():
            continue
        r = convert_split(root_path, split, out_path)
        summary.append(r)
        print(f"[{split}] تصاویر: {r['n_images']} | annotation: {r['n_annotations']} | "
              f"خط‌های رد شده: {r['n_skipped_lines']} | COCO json: {r['coco_json']}")

    write_yolo_data_yaml(out_path)

    total_imgs = sum(s["n_images"] for s in summary)
    total_anns = sum(s["n_annotations"] for s in summary)
    print(f"\n== خلاصه کلی == تصاویر: {total_imgs} | annotation ها: {total_anns}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    main(args.root, args.out)
