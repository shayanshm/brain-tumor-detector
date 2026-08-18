"""
Phase 6.3 — تحلیل افت عملکرد: مقایسه‌ی F1/Precision/Recall هر corruption با حالت original،
با استفاده از همان موتور تست‌شده‌ی compare_models.py (Phase 5).

اجرا (بدون نیاز به GPU، بعد از دریافت خروجی‌های Kaggle):
    python src/robustness/degradation_analysis.py \\
        --gt data/processed/coco_format/test.json \\
        --pred-dir outputs_v3/logs/robustness --prefix faster_rcnn \\
        --out reports/tables/phase6_faster_rcnn_degradation.json
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eval.compare_models import evaluate_predictions  # noqa: E402

CONDITIONS = ["original", "darker", "brighter", "gaussian_noise", "salt_pepper_noise",
              "gaussian_blur", "motion_blur", "jpeg_20", "jpeg_50"]


def main(args):
    pred_dir = Path(args.pred_dir)
    results = {}

    baseline_f1 = None
    for cond in CONDITIONS:
        pred_path = pred_dir / f"{args.prefix}_{cond}.json"
        if not pred_path.exists():
            print(f"⚠️ فایل {pred_path} پیدا نشد، رد می‌شود")
            continue
        r = evaluate_predictions(args.gt, str(pred_path), conf_threshold=args.conf_threshold)
        overall = r["overall"]
        results[cond] = overall
        if cond == "original":
            baseline_f1 = overall["f1"]
        print(f"{cond:20s}: P={overall['precision']:.3f} R={overall['recall']:.3f} "
              f"F1={overall['f1']:.3f} IoU={overall['mean_iou']:.3f}")

    if baseline_f1:
        print(f"\n== درصد افت F1 نسبت به original (F1={baseline_f1:.3f}) ==")
        for cond, r in results.items():
            if cond == "original":
                continue
            drop_pct = (baseline_f1 - r["f1"]) / baseline_f1 * 100 if baseline_f1 > 0 else 0
            print(f"  {cond:20s}: F1={r['f1']:.3f}  (افت {drop_pct:+.1f}%)")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nذخیره شد: {args.out}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt", required=True)
    parser.add_argument("--pred-dir", required=True)
    parser.add_argument("--prefix", required=True, help="faster_rcnn یا yolo")
    parser.add_argument("--conf-threshold", type=float, default=0.5)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    main(args)
