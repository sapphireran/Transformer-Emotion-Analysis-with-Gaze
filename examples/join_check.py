#!/usr/bin/env python3
"""Check that ZuCo labels + scaled averages reconstruct the combined table."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from csvutil import read_dicts
from paths import (
    ZUCO_ALL_GAZE_COLS,
    ZUCO_AVERAGE_MM,
    ZUCO_AVERAGE_STD,
    ZUCO_COMBINED_MM,
    ZUCO_COMBINED_STD,
    ZUCO_LABELS,
)


def index_by_id(path, key: str) -> dict[str, dict[str, str]]:
    _, rows = read_dicts(path)
    return {row[key]: row for row in rows}


def compare_join(labels, averages, combined, gaze_cols, id_key: str) -> list[str]:
    problems = []
    if set(labels) != set(combined):
        problems.append(
            f"label ids and combined ids differ "
            f"(only-labels={len(set(labels) - set(combined))}, "
            f"only-combined={len(set(combined) - set(labels))})"
        )
    missing_avg = sorted(set(combined) - set(averages), key=lambda x: int(x))
    if missing_avg:
        problems.append(f"{len(missing_avg)} combined ids missing from averages")

    sentence_mismatches = 0
    label_mismatches = 0
    gaze_mismatches = 0
    compared = 0
    for sid, row in combined.items():
        if sid not in labels or sid not in averages:
            continue
        compared += 1
        if row["sentence"] != labels[sid]["sentence"]:
            sentence_mismatches += 1
        if row["sentiment_label"] != labels[sid]["sentiment_label"]:
            label_mismatches += 1
        for col in gaze_cols:
            if abs(float(row[col]) - float(averages[sid][col])) > 1e-9:
                gaze_mismatches += 1
                break
    if sentence_mismatches:
        problems.append(f"{sentence_mismatches} sentence text mismatches")
    if label_mismatches:
        problems.append(f"{label_mismatches} sentiment_label mismatches")
    if gaze_mismatches:
        problems.append(f"{gaze_mismatches} rows have a gaze column mismatch")
    if compared != 400:
        problems.append(f"only compared {compared} rows")
    return problems


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    labels = index_by_id(ZUCO_LABELS, "sentence_id")
    std_avg = index_by_id(ZUCO_AVERAGE_STD, "id")
    mm_avg = index_by_id(ZUCO_AVERAGE_MM, "id")
    std_combined = index_by_id(ZUCO_COMBINED_STD, "sentence_id")
    mm_combined = index_by_id(ZUCO_COMBINED_MM, "sentence_id")

    problems = []
    problems += compare_join(labels, std_avg, std_combined, ZUCO_ALL_GAZE_COLS, "id")
    problems += [
        f"min-max: {item}"
        for item in compare_join(labels, mm_avg, mm_combined, ZUCO_ALL_GAZE_COLS, "id")
    ]

    if not args.quiet:
        print("ZuCo join check")
        print(f"  labels: {len(labels)}")
        print(f"  standard averages: {len(std_avg)}")
        print(f"  min-max averages: {len(mm_avg)}")
        print(f"  combined standard: {len(std_combined)}")
        print(f"  combined min-max: {len(mm_combined)}")
        print("  join key: ssts_ZuCo.sentence_id = average.id = combined.sentence_id")
        print("  gaze columns copied from the scaled average tables")

    if problems:
        print("JOIN CHECK FAILED:")
        for item in problems:
            print(" -", item)
        return 1

    print("join check OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
