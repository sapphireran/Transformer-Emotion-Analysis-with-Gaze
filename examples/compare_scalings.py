#!/usr/bin/env python3
"""Confirm the min-max and z-score ZuCo tables are the same 400 sentences.

Usage:
    python3 examples/compare_scalings.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from examples.lib.loaders import ZUCO_GAZE_COLUMNS, load_zuco_experiment


def main() -> int:
    std = load_zuco_experiment("standard")
    mm = load_zuco_experiment("minmax")

    problems: list[str] = []
    if not std["sentence_id"].equals(mm["sentence_id"]):
        problems.append("sentence_id columns differ")
    if not std["sentence"].equals(mm["sentence"]):
        problems.append("sentence text differs")
    if not std["sentiment_label"].equals(mm["sentiment_label"]):
        problems.append("labels differ")

    print(f"rows: standard={len(std)} minmax={len(mm)}")
    print("identity columns match" if not problems else "IDENTITY PROBLEMS:")
    for item in problems:
        print(f"  - {item}")

    print("\nPer-column check (z-score ~mean 0 std 1; min-max in [0, 1]; Spearman ≈ 1)")
    print(f"{'column':16} {'z_mean':>9} {'z_std':>9} {'mm_min':>9} {'mm_max':>9} {'spearman':>9}")
    for col in ZUCO_GAZE_COLUMNS:
        z = std[col].to_numpy(dtype=float)
        m = mm[col].to_numpy(dtype=float)
        rho = spearman(z, m)
        print(
            f"{col:16} {z.mean():9.4f} {z.std(ddof=0):9.4f} "
            f"{m.min():9.4f} {m.max():9.4f} {rho:9.6f}"
        )
        if m.min() < -1e-9 or m.max() > 1 + 1e-9:
            problems.append(f"{col} min-max outside [0, 1]")
        # Tiny float ties can break exact rank equality (omissionRate has one).
        if rho < 0.999:
            problems.append(f"{col} Spearman {rho:.6f} is below 0.999")

    print()
    if problems:
        print("failed:")
        for item in problems:
            print(f"  - {item}")
        return 1
    print(
        "These files are the same 400 reviews under two column-wise transforms.\n"
        "Do not use them as a train/test pair."
    )
    return 0


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    """Pearson correlation of ranks. Average ties, ignore a constant column."""
    ra = _rank_average(a)
    rb = _rank_average(b)
    ra = ra - ra.mean()
    rb = rb - rb.mean()
    denom = np.sqrt((ra**2).sum() * (rb**2).sum())
    if denom == 0:
        return 1.0 if np.allclose(ra, rb) else 0.0
    return float((ra * rb).sum() / denom)


def _rank_average(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, values.size + 1, dtype=float)
    sorted_vals = values[order]
    i = 0
    while i < values.size:
        j = i + 1
        while j < values.size and sorted_vals[j] == sorted_vals[i]:
            j += 1
        if j - i > 1:
            ranks[order[i:j]] = ranks[order[i:j]].mean()
        i = j
    return ranks


if __name__ == "__main__":
    raise SystemExit(main())
