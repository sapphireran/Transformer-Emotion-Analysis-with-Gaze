#!/usr/bin/env python3
"""Inspect sentence-level ZuCo gaze CSVs (raw subjects, mean, scaled).

Prints row counts and finite-value summaries so you can see that the
checked-in tables are complete and that the z-scored average really is
centered near zero.

Usage (repo root):

    python examples/inspect_sentence_gaze.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from examples.common import (
    SENTENCE_GAZE_COLS,
    ZUCO_SENT_DIR,
    format_stats_table,
    floats,
    read_dicts,
)


def _load(name: str):
    path = ZUCO_SENT_DIR / name
    rows = read_dicts(path)
    return path, rows


def main() -> None:
    print("=== Per-subject sentence tables ===")
    for i in range(1, 13):
        path, rows = _load(f"{i}_SR.csv")
        nfix = floats(rows, "nFixations")
        print(
            f"{path.name:10} rows={len(rows):4d}  "
            f"nFixations mean={nfix.mean():6.3f}  "
            f"omissionRate mean={floats(rows, 'omissionRate').mean():6.3f}"
        )

    print()
    print("=== average_data.csv (raw mean across subjects) ===")
    path, rows = _load("average_data.csv")
    print(f"{path}  rows={len(rows)}")
    print(format_stats_table(SENTENCE_GAZE_COLS, [floats(rows, c) for c in SENTENCE_GAZE_COLS]))

    print()
    print("=== standard_scaled_average_data.csv (z-score of the mean) ===")
    path, rows = _load("standard_scaled_average_data.csv")
    print(f"{path}  rows={len(rows)}")
    print(format_stats_table(SENTENCE_GAZE_COLS, [floats(rows, c) for c in SENTENCE_GAZE_COLS]))
    print()
    print("Z-scored columns should have mean ~0 and std ~1 (population std).")
    print("SentLen is scaled too: a '1.0' sentence is long relative to this set,")
    print("not 'one word'.")

    print()
    print("=== min_max_scaled_average_data.csv ===")
    path, rows = _load("min_max_scaled_average_data.csv")
    print(f"{path}  rows={len(rows)}")
    print(format_stats_table(SENTENCE_GAZE_COLS, [floats(rows, c) for c in SENTENCE_GAZE_COLS]))
    print()
    print("Min-max columns should sit in [0, 1] aside from numerical noise.")


if __name__ == "__main__":
    main()
