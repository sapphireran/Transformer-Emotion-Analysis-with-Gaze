#!/usr/bin/env python3
"""Label balance and class-conditional gaze means on the 400-sentence table.

This is the table `model_ZuCo_SST.py` actually trains on
(`combined_sst_et_standard.csv`). Gaze columns are z-scored, so a mean of
+0.4 for `GPT` on negative sentences is "0.4 standard deviations above the
corpus mean," not 0.4 milliseconds.

Usage (repo root):

    python examples/label_and_gaze_summary.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from examples.common import (
    GAZE5_ZUCO,
    LABEL_NAMES,
    ZUCO_SST_DIR,
    format_stats_table,
    floats,
    int_col,
    label_histogram,
    matrix,
    read_dicts,
)

EXTRA = ("omissionRate", "meanPupilSize", "SFD")


def _means_by_label(rows, columns):
    labels = int_col(rows, "sentiment_label")
    x = matrix(rows, columns)
    print(f"{'label':<10} " + " ".join(f"{c:>12}" for c in columns))
    print("-" * (10 + 13 * len(columns)))
    for cls in sorted(LABEL_NAMES):
        mask = labels == cls
        means = x[mask].mean(axis=0)
        print(f"{cls} {LABEL_NAMES[cls]:<8} " + " ".join(f"{m:12.4f}" for m in means))
    overall = x.mean(axis=0)
    print(f"{'all':<10} " + " ".join(f"{m:12.4f}" for m in overall))


def _between_over_within(rows, columns) -> None:
    """Rough one-way signal: between-class variance / within-class variance."""
    labels = int_col(rows, "sentiment_label")
    x = matrix(rows, columns)
    print()
    print("=== between-class / within-class variance (higher → more label signal) ===")
    print(f"{'column':<16} {'ratio':>8}")
    for j, name in enumerate(columns):
        col = x[:, j]
        grand = col.mean()
        between = 0.0
        within = 0.0
        for cls in sorted(LABEL_NAMES):
            part = col[labels == cls]
            if part.size == 0:
                continue
            between += part.size * float(part.mean() - grand) ** 2
            within += float(np.sum((part - part.mean()) ** 2))
        ratio = between / within if within > 0 else float("nan")
        print(f"{name:<16} {ratio:8.4f}")


def main() -> None:
    combined = read_dicts(ZUCO_SST_DIR / "combined_sst_et_standard.csv")
    labels = int_col(combined, "sentiment_label")
    print(f"ZuCo_SST_data/combined_sst_et_standard.csv  rows={len(combined)}")
    print()
    print("=== sentiment_label histogram ===")
    print(label_histogram(labels))

    print()
    print("=== z-scored gaze, overall ===")
    cols = tuple(EXTRA) + GAZE5_ZUCO
    print(format_stats_table(cols, [floats(combined, c) for c in cols]))

    print()
    print("=== class-conditional means (fusion 5-D) ===")
    _means_by_label(combined, GAZE5_ZUCO)

    print()
    print("=== class-conditional means (held-out analysis columns) ===")
    _means_by_label(combined, EXTRA)

    _between_over_within(combined, cols)

    print()
    print("Hold-out files from spilt.py (not used by model_ZuCo_SST.py):")
    for name in ("train.csv", "valid.csv", "test.csv"):
        part = read_dicts(ZUCO_SST_DIR / name)
        hist = label_histogram(int_col(part, "sentiment_label"))
        print(f"\n-- {name} n={len(part)} --")
        print(hist)

    print()
    print("If class-conditional gaze means are almost identical, a linear")
    print("gaze-only model will sit near the majority baseline — see")
    print("examples/gaze_only_baseline.py.")


if __name__ == "__main__":
    main()
