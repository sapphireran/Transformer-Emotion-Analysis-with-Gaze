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

    print("\nPer-column check (z-score should be ~mean 0 std 1; min-max in [0, 1])")
    print(f"{'column':16} {'z_mean':>9} {'z_std':>9} {'mm_min':>9} {'mm_max':>9} {'rank_eq':>8}")
    rank_ok = True
    for col in ZUCO_GAZE_COLUMNS:
        z = std[col].to_numpy(dtype=float)
        m = mm[col].to_numpy(dtype=float)
        # Same order statistics? Spearman via rank of finite values.
        z_rank = pd_rank(z)
        m_rank = pd_rank(m)
        eq = bool(np.array_equal(z_rank, m_rank))
        rank_ok = rank_ok and eq
        print(
            f"{col:16} {z.mean():9.4f} {z.std(ddof=0):9.4f} "
            f"{m.min():9.4f} {m.max():9.4f} {'yes' if eq else 'NO':>8}"
        )
        if m.min() < -1e-9 or m.max() > 1 + 1e-9:
            problems.append(f"{col} min-max outside [0, 1]")

    if not rank_ok:
        problems.append("at least one column does not share rank order across scales")

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


def pd_rank(values: np.ndarray) -> np.ndarray:
    """Average ranks, matching pandas default, for a 1-d array."""
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(values.size, dtype=float)
    # Average ties.
    sorted_vals = values[order]
    i = 0
    while i < values.size:
        j = i + 1
        while j < values.size and sorted_vals[j] == sorted_vals[i]:
            j += 1
        if j - i > 1:
            avg = ranks[order[i:j]].mean()
            ranks[order[i:j]] = avg
        i = j
    return ranks


if __name__ == "__main__":
    raise SystemExit(main())
