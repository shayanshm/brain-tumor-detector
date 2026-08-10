"""
Phase 2.5 — ویژوالایز نمونه‌های تصادفی (Image -> BBox -> Class Label).

نحوه‌ی اجرا در Kaggle Notebook:
    import sys
    sys.path.insert(0, "/kaggle/working/brain-tumor-detector/src")
    from data.visualize_samples import plot_samples
    plot_samples(DATASET_ROOT, split="train", n=9, save_path="/kaggle/working/sample_grid.png")
"""
import random
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.patches import Polygon as MplPolygon
from PIL import Image

CLASS_NAMES = ["glioma", "meningioma", "pituitary"]
CLASS_COLORS = ["#e74c3c", "#3498db", "#2ecc71"]


def load_yolo_annotations(label_path: Path):
    """
    برمی‌گرداند: لیستی از (cls_id, kind, data)
      kind='bbox'    -> data = (xc, yc, w, h)
      kind='polygon' -> data = [(x1,y1), (x2,y2), ...]
    """
    items = []
    if not label_path.exists() or not label_path.read_text().strip():
        return items
    for line in label_path.read_text().strip().splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        cls_id = int(parts[0])
        coords = [float(v) for v in parts[1:]]
        if len(coords) == 4:
            items.append((cls_id, "bbox", tuple(coords)))
        elif len(coords) >= 6 and len(coords) % 2 == 0:
            pts = list(zip(coords[0::2], coords[1::2]))
            items.append((cls_id, "polygon", pts))
    return items


def plot_samples(root: str, split: str = "train", n: int = 9, seed: int = 42, save_path: str | None = None):
    random.seed(seed)
    root_path = Path(root)
    img_dir = root_path / split / "images"
    lbl_dir = root_path / split / "labels"

    all_images = sorted(img_dir.glob("*"))
    if not all_images:
        raise FileNotFoundError(f"هیچ تصویری در {img_dir} پیدا نشد")

    sample = random.sample(all_images, min(n, len(all_images)))
    cols = 3
    rows = (len(sample) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    axes = axes.flatten() if len(sample) > 1 else [axes]

    for ax, img_path in zip(axes, sample):
        img = Image.open(img_path).convert("RGB")
        w_img, h_img = img.size
        ax.imshow(img)

        label_path = lbl_dir / f"{img_path.stem}.txt"
        annotations = load_yolo_annotations(label_path)
        for cls_id, kind, data in annotations:
            color = CLASS_COLORS[cls_id % len(CLASS_COLORS)]
            label = CLASS_NAMES[cls_id] if 0 <= cls_id < len(CLASS_NAMES) else f"cls{cls_id}"

            if kind == "bbox":
                xc, yc, w, h = data
                x0 = (xc - w / 2) * w_img
                y0 = (yc - h / 2) * h_img
                rect = Rectangle((x0, y0), w * w_img, h * h_img, linewidth=2,
                                   edgecolor=color, facecolor="none")
                ax.add_patch(rect)
                label_x, label_y = x0, y0
            else:  # polygon
                px = [p[0] * w_img for p in data]
                py = [p[1] * h_img for p in data]
                poly = MplPolygon(list(zip(px, py)), closed=True, linewidth=2,
                                    edgecolor=color, facecolor=color, alpha=0.15)
                ax.add_patch(poly)
                # bbox محاطی به‌صورت نقطه‌چین برای مرجع
                x0, x1 = min(px), max(px)
                y0, y1 = min(py), max(py)
                rect = Rectangle((x0, y0), x1 - x0, y1 - y0, linewidth=1,
                                   edgecolor=color, facecolor="none", linestyle="--", alpha=0.6)
                ax.add_patch(rect)
                label_x, label_y = x0, y0

            ax.text(label_x, max(label_y - 5, 0), label, color="white", fontsize=9,
                     bbox=dict(facecolor=color, alpha=0.8, pad=1))

        ax.set_title(img_path.name, fontsize=8)
        ax.axis("off")

    # خالی کردن محورهای اضافی
    for ax in axes[len(sample):]:
        ax.axis("off")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=120, bbox_inches="tight")
        print(f"ذخیره شد: {save_path}")
    else:
        plt.show()
    plt.close(fig)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--n", type=int, default=9)
    parser.add_argument("--save", default=None)
    args = parser.parse_args()
    plot_samples(args.root, args.split, args.n, save_path=args.save)
