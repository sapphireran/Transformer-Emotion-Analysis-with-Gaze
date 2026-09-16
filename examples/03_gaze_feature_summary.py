#!/usr/bin/env python3
"""Summarize the five fusion gaze features on the measured ZuCo set.

Prints per-column mean/std/min/max, a Pearson correlation matrix, class
conditional means, and how many rows are exactly zero (placeholder-like).

    python3 examples/03_gaze_feature_summary.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np

from common import (
    LABEL_NAMES,
    SST_GAZE_COLS,
    ZUCO_EXTRA_GAZE,
    ZUCO_GAZE_COLS,
    as_int_column,
    correlation_matrix,
    format_counts,
    label_counts,
    matrix,
    read_csv,
    table,
    write_text,
)


def describe(names: list[str], x: np.ndarray) -> str:
    rows = []
    for j, name in enumerate(names):
        col = x[:, j]
        finite = col[np.isfinite(col)]
        rows.append(
            (
                name,
                f"{finite.mean():.4f}",
                f"{finite.std(ddof=0):.4f}",
                f"{finite.min():.4f}",
                f"{np.quantile(finite, 0.25):.4f}",
                f"{np.median(finite):.4f}",
                f"{np.quantile(finite, 0.75):.4f}",
                f"{finite.max():.4f}",
                str(int(np.sum(col == 0))),
            )
        )
    return table(
        ["feature", "mean", "std", "min", "q25", "median", "q75", "max", "n_zero"],
        rows,
    )


def corr_table(names: list[str], x: np.ndarray) -> str:
    corr = correlation_matrix(x)
    rows = []
    for i, name in enumerate(names):
        cells = [name] + [f"{corr[i, j]:.3f}" for j in range(len(names))]
        rows.append(cells)
    return table(["feature", *names], rows)


def class_means(names: list[str], x: np.ndarray, y: np.ndarray) -> str:
    rows = []
    for label in (0, 1, 2):
        mask = y == label
        means = x[mask].mean(axis=0)
        rows.append((f"{label} {LABEL_NAMES[label]} (n={int(mask.sum())})", *[f"{v:.4f}" for v in means]))
    return table(["class", *names], rows)


def section(title: str, relative: str, gaze_cols: list[str], extra: list[str] | None = None) -> str:
    header, rows = read_csv(relative)
    y = as_int_column(rows, header, "sentiment_label")
    x = matrix(rows, header, gaze_cols)
    blocks = [
        f"## {title}",
        f"path: {relative}",
        f"rows: {len(rows)}  labels: {format_counts(label_counts(y))}",
        "",
        "### fusion columns (order used by EyeTrackingModel)",
        describe(gaze_cols, x),
        "",
        "### Pearson correlation (fusion columns)",
        corr_table(gaze_cols, x),
        "",
        "### mean by sentiment class",
        class_means(gaze_cols, x, y),
    ]
    if extra:
        extra_x = matrix(rows, header, extra)
        blocks.extend(
            [
                "",
                "### extra ZuCo columns (not fed to the fusion head)",
                describe(extra, extra_x),
                "",
                "### mean by class (extra columns)",
                class_means(extra, extra_x, y),
            ]
        )
    return "\n".join(blocks) + "\n"


def main() -> int:
    parts = [
        "# Gaze feature summary",
        "",
        "Standard-scaled ZuCo numbers are z-scores, so means sit near 0.",
        "Full-SST numbers are transferred / mapped — treat them as a different",
        "feature space even when the column names look familiar.",
        "",
        section(
            "Measured gaze — ZuCo ∩ SST (standard)",
            "ZuCo_SST_data/combined_sst_et_standard.csv",
            ZUCO_GAZE_COLS,
            extra=ZUCO_EXTRA_GAZE,
        ),
        section(
            "Transferred gaze — full SST train split",
            "SST_data/train_full_sst.csv",
            SST_GAZE_COLS,
        ),
    ]
    text = "\n".join(parts)
    print(text)
    write_text("03_gaze_feature_summary.txt", text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
