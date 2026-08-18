"""
Phase 6.2 — اجرای Faster R-CNN روی تصویر اصلی + همه‌ی ۸ نوع corruption، و ذخیره‌ی پیش‌بینی‌ها
به فرمت یکسان با Phase 5 (قابل استفاده مستقیم در compare_models.py / degradation_analysis.py).

نحوه‌ی اجرا در Kaggle:
    !python /kaggle/working/brain-tumor-detector/src/robustness/run_robustness_faster_rcnn.py \\
        --data-root /kaggle/working/brain-tumor-detector/data/processed \\
        --checkpoint /kaggle/working/brain-tumor-detector/outputs_v3/checkpoints/faster_rcnn_epoch50.pt \\
        --out-dir /kaggle/working/brain-tumor-detector/outputs_v3/logs/robustness
"""
import json
import sys
from pathlib import Path

import torch
import torchvision.transforms.functional as F
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from models.faster_rcnn import build_faster_rcnn  # noqa: E402
from models.coco_dataset import NUM_CLASSES_WITH_BACKGROUND  # noqa: E402
from robustness.corruptions import CORRUPTIONS  # noqa: E402


@torch.no_grad()
def run_on_condition(model, device, images_dir: Path, filename_to_id: dict,
                      corrupt_fn=None, score_threshold: float = 0.05):
    model.eval()
    predictions = []
    image_files = sorted(images_dir.glob("*.jpg")) + sorted(images_dir.glob("*.png"))

    for img_path in image_files:
        image_id = filename_to_id.get(img_path.name)
        if image_id is None:
            continue
        img = Image.open(img_path).convert("RGB")
        if corrupt_fn is not None:
            img = corrupt_fn(img)

        tensor = F.to_tensor(img).to(device)
        output = model([tensor])[0]

        boxes = output["boxes"].cpu().numpy()
        scores = output["scores"].cpu().numpy()
        labels = output["labels"].cpu().numpy()
        for box, score, label in zip(boxes, scores, labels):
            if score < score_threshold:
                continue
            x1, y1, x2, y2 = box
            predictions.append({
                "image_id": image_id,
                "category_id": int(label) - 1,
                "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                "score": float(score),
            })
    return predictions


def main(args):
    data_root = Path(args.data_root)
    with open(data_root / "coco_format" / "test.json") as f:
        coco_gt = json.load(f)
    filename_to_id = {img["file_name"]: img["id"] for img in coco_gt["images"]}

    images_dir = data_root / "yolo_detection" / "test" / "images"
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_faster_rcnn(num_classes=NUM_CLASSES_WITH_BACKGROUND, pretrained=False)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.to(device)

    conditions = {"original": None, **CORRUPTIONS}
    for name, fn in conditions.items():
        print(f"== در حال اجرا: {name} ==")
        preds = run_on_condition(model, device, images_dir, filename_to_id, corrupt_fn=fn)
        out_path = out_dir / f"faster_rcnn_{name}.json"
        with open(out_path, "w") as f:
            json.dump(preds, f)
        print(f"  {len(preds)} پیش‌بینی -> {out_path}")

    print("\nهمه‌ی ۹ حالت تمام شد.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()
    main(args)
