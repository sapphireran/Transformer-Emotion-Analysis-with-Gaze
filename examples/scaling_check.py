#!/usr/bin/env python3
"""Rebuild min-max and z-score tables from average_data.csv and compare to git."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from csvutil import read_dicts
from paths import (
    AVERAGE_NUMERIC_COLS,
    ZUCO_AVERAGE,
    ZUCO_AVERAGE_MM,
    ZUCO_AVERAGE_STD,
)


def matrix(path, columns) -> tuple[np.ndarray, list[int]]:
    _, rows = read_dicts(path)
    ids = [int(row["id"]) for row in rows]
    data = np.array([[float(row[col]) for col in columns] for row in rows], dtype=np.float64)
    return data, ids


def minmax(data: np.ndarray) -> np.ndarray:
    lo = data.min(axis=0)
    hi = data.max(axis=0)
    span = np.where(hi > lo, hi - lo, 1.0)
    return (data - lo) / span


def zscore(data: np.ndarray) -> np.ndarray:
    # sklearn.preprocessing.StandardScaler uses population std (ddof=0).
    mu = data.mean(axis=0)
    sigma = data.std(axis=0, ddof=0)
    sigma = np.where(sigma == 0, 1.0, sigma)
    return (data - mu) / sigma


def max_abs_diff(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.max(np.abs(left - right)))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--atol", type=float, default=1e-8)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    raw, raw_ids = matrix(ZUCO_AVERAGE, AVERAGE_NUMERIC_COLS)
    std, std_ids = matrix(ZUCO_AVERAGE_STD, AVERAGE_NUMERIC_COLS)
    mm, mm_ids = matrix(ZUCO_AVERAGE_MM, AVERAGE_NUMERIC_COLS)

    problems = []
    if raw_ids != std_ids or raw_ids != mm_ids:
        problems.append("id columns are not aligned across the three average tables")

    rebuilt_mm = minmax(raw)
    rebuilt_std = zscore(raw)
    mm_err = max_abs_diff(rebuilt_mm, mm)
    std_err = max_abs_diff(rebuilt_std, std)

    if not args.quiet:
        print("Scaling check on ZuCo_et_csv_data/average_data.csv")
        print(f"  rows={raw.shape[0]} cols={raw.shape[1]}")
        print(f"  min-max max|delta| vs min_max_scaled_average_data.csv: {mm_err:.3e}")
        print(f"  z-score max|delta| vs standard_scaled_average_data.csv: {std_err:.3e}")
        print("  z-score column means (should be ~0):", np.round(std.mean(axis=0), 6))
        print("  z-score column stds  (should be ~1):", np.round(std.std(axis=0, ddof=0), 6))
        print("  min-max column mins (should be 0):", np.round(mm.min(axis=0), 6))
        print("  min-max column maxs (should be 1):", np.round(mm.max(axis=0), 6))

    if mm_err > args.atol:
        problems.append(f"min-max mismatch {mm_err} > {args.atol}")
    if std_err > args.atol:
        problems.append(f"z-score mismatch {std_err} > {args.atol}")

    if problems:
        print("SCALING CHECK FAILED:")
        for item in problems:
            print(" -", item)
        return 1

    print("scaling check OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
