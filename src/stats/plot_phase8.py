"""
Phase 8.5 (کمکی) — نمودار توزیع Bootstrap تفاوت میانگین IoU بین دو مدل.

اجرا (بدون نیاز به GPU):
    python src/stats/plot_phase8.py \\
        --paired-csv reports/tables/phase8_paired_arrays.csv \\
        --out reports/figures/phase8_bootstrap_distribution.png
"""
import argparse
import csv
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument("--paired-csv", required=True)
parser.add_argument("--out", required=True)
args = parser.parse_args()

with open(args.paired_csv) as f:
    rows = list(csv.DictReader(f))

fr_iou = np.array([float(r["fr_iou"]) for r in rows])
yolo_iou = np.array([float(r["yolo_iou"]) for r in rows])

rng = np.random.default_rng(42)
n = len(fr_iou)
n_boot = 10000
diffs = np.empty(n_boot)
for i in range(n_boot):
    idx = rng.integers(0, n, size=n)
    diffs[i] = yolo_iou[idx].mean() - fr_iou[idx].mean()

lo, hi = np.percentile(diffs, [2.5, 97.5])

fig, ax = plt.subplots(figsize=(9, 5.5))
ax.hist(diffs, bins=60, color="#3498db", alpha=0.75, edgecolor="white")
ax.axvline(0, color="black", linestyle="-", linewidth=1.5, label="صفر (بدون تفاوت)")
ax.axvline(diffs.mean(), color="#c0392b", linestyle="--", linewidth=2,
           label=f"میانگین مشاهده‌شده = {diffs.mean():.4f}")
ax.axvline(lo, color="#27ae60", linestyle=":", linewidth=2)
ax.axvline(hi, color="#27ae60", linestyle=":", linewidth=2, label=f"95% CI = [{lo:.3f}, {hi:.3f}]")
ax.set_xlabel("تفاوت میانگین IoU (YOLO11 - Faster R-CNN)")
ax.set_ylabel("فراوانی (از ۱۰,۰۰۰ نمونه‌ی Bootstrap)")
ax.set_title("Phase 8 -- توزیع Bootstrap تفاوت عملکرد (صفر داخل بازه‌ی اطمینان است)")
ax.legend()
plt.tight_layout()

Path(args.out).parent.mkdir(parents=True, exist_ok=True)
plt.savefig(args.out, dpi=130)
print(f"ذخیره شد در: {args.out}")
print("CI:", lo, hi)
