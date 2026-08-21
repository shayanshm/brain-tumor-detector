"""
Phase 8.2-8.5 — تحلیل آماری کامل روی آرایه‌ی زوجی per-image.

اجرا (بدون نیاز به GPU):
    python src/stats/statistical_tests.py \\
        --paired-csv reports/tables/phase8_paired_arrays.csv \\
        --out reports/tables/phase8_statistical_results.json
"""
import csv
import json
from pathlib import Path

import numpy as np
from scipy import stats


def bootstrap_ci(values, n_boot=10000, ci=95, seed=42):
    rng = np.random.default_rng(seed)
    values = np.asarray(values)
    n = len(values)
    boot_means = np.empty(n_boot)
    for i in range(n_boot):
        sample = rng.choice(values, size=n, replace=True)
        boot_means[i] = sample.mean()
    lower = np.percentile(boot_means, (100 - ci) / 2)
    upper = np.percentile(boot_means, 100 - (100 - ci) / 2)
    return float(values.mean()), float(lower), float(upper)


def bootstrap_ci_paired_diff(a, b, n_boot=10000, ci=95, seed=42):
    """CI بوت‌استرپ روی تفاوت میانگین a-b، با حفظ جفت‌شدگی (هر بار همان تصاویر برای هر دو resample می‌شوند)."""
    rng = np.random.default_rng(seed)
    a, b = np.asarray(a), np.asarray(b)
    n = len(a)
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        diffs[i] = a[idx].mean() - b[idx].mean()
    lower = np.percentile(diffs, (100 - ci) / 2)
    upper = np.percentile(diffs, 100 - (100 - ci) / 2)
    return float((a - b).mean()), float(lower), float(upper)


def paired_permutation_test(a, b, n_perm=10000, seed=42):
    """آزمون permutation زوجی: برای هر تصویر با احتمال ۵۰٪ مقادیر a و b را جابه‌جا می‌کنیم."""
    rng = np.random.default_rng(seed)
    a, b = np.asarray(a), np.asarray(b)
    observed_diff = a.mean() - b.mean()

    n = len(a)
    perm_diffs = np.empty(n_perm)
    for i in range(n_perm):
        swap = rng.random(n) < 0.5
        a_perm = np.where(swap, b, a)
        b_perm = np.where(swap, a, b)
        perm_diffs[i] = a_perm.mean() - b_perm.mean()

    p_value = np.mean(np.abs(perm_diffs) >= np.abs(observed_diff))
    return float(observed_diff), float(p_value)


def mcnemar_test(a_correct, b_correct):
    """a_correct, b_correct: آرایه‌ی 0/1 هم‌طول. جدول ۲x۲ discordant pairs را می‌سازد."""
    from statsmodels.stats.contingency_tables import mcnemar

    a_correct = np.asarray(a_correct)
    b_correct = np.asarray(b_correct)
    both_correct = int(np.sum((a_correct == 1) & (b_correct == 1)))
    only_a = int(np.sum((a_correct == 1) & (b_correct == 0)))
    only_b = int(np.sum((a_correct == 0) & (b_correct == 1)))
    both_wrong = int(np.sum((a_correct == 0) & (b_correct == 0)))

    table = [[both_correct, only_a], [only_b, both_wrong]]
    exact = (only_a + only_b) < 25  # قانون رایج: زیر ۲۵ discordant pair از نسخه‌ی exact استفاده شود
    result = mcnemar(table, exact=exact, correction=not exact)

    return {
        "table_both_correct": both_correct,
        "table_only_a_correct": only_a,
        "table_only_b_correct": only_b,
        "table_both_wrong": both_wrong,
        "statistic": float(result.statistic),
        "p_value": float(result.pvalue),
        "used_exact": exact,
    }


def cohens_d_paired(a, b):
    diff = np.asarray(a) - np.asarray(b)
    return float(diff.mean() / diff.std(ddof=1)) if diff.std(ddof=1) > 0 else 0.0


def main(args):
    with open(args.paired_csv) as f:
        rows = list(csv.DictReader(f))

    fr_iou = [float(r["fr_iou"]) for r in rows]
    yolo_iou = [float(r["yolo_iou"]) for r in rows]
    fr_correct = [int(r["fr_correct"]) for r in rows]
    yolo_correct = [int(r["yolo_correct"]) for r in rows]

    results = {}

    # --- 8.2: Bootstrap CI روی میانگین IoU هر مدل + تفاوت ---
    fr_mean, fr_lo, fr_hi = bootstrap_ci(fr_iou)
    yolo_mean, yolo_lo, yolo_hi = bootstrap_ci(yolo_iou)
    diff_mean, diff_lo, diff_hi = bootstrap_ci_paired_diff(yolo_iou, fr_iou)
    results["bootstrap_ci_mean_iou"] = {
        "faster_rcnn": {"mean": fr_mean, "ci95_lower": fr_lo, "ci95_upper": fr_hi},
        "yolo11": {"mean": yolo_mean, "ci95_lower": yolo_lo, "ci95_upper": yolo_hi},
        "diff_yolo_minus_fr": {"mean": diff_mean, "ci95_lower": diff_lo, "ci95_upper": diff_hi},
    }

    # --- 8.3a: Wilcoxon Signed-Rank Test (IoU زوجی) ---
    diffs = np.array(yolo_iou) - np.array(fr_iou)
    nonzero = diffs[diffs != 0]
    if len(nonzero) > 0:
        w_stat, w_p = stats.wilcoxon(nonzero)
    else:
        w_stat, w_p = float("nan"), 1.0
    results["wilcoxon_signed_rank_iou"] = {"statistic": float(w_stat), "p_value": float(w_p)}

    # --- 8.3b: Paired Permutation Test (روی IoU) ---
    perm_diff, perm_p = paired_permutation_test(yolo_iou, fr_iou)
    results["paired_permutation_test_iou"] = {
        "observed_diff_yolo_minus_fr": perm_diff, "p_value": perm_p, "n_permutations": 10000,
    }

    # --- 8.4: McNemar's Test (تصمیم‌های correct/incorrect زوجی) ---
    results["mcnemar_test_correct_detection"] = mcnemar_test(yolo_correct, fr_correct)

    # --- Effect size ---
    results["effect_size_cohens_d_iou"] = cohens_d_paired(yolo_iou, fr_iou)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)

    print(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"\nذخیره شد: {args.out}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--paired-csv", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    main(args)
