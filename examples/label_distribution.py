#!/usr/bin/env python3
"""Three-class sentiment counts for every labeled table in this repo.

Run from the repository root:

    python3 examples/label_distribution.py
"""

from __future__ import annotations

import sys
from collections import Counter

from common import (
    abs_path,
    as_int,
    format_label_counts,
    read_rows,
)

LABELED = (
    ("ZuCo text only", "ZuCo_SST_data/ssts_ZuCo.csv"),
    ("ZuCo + standard gaze", "ZuCo_SST_data/combined_sst_et_standard.csv"),
    ("ZuCo + min-max gaze", "ZuCo_SST_data/combined_sst_et_min_max.csv"),
    ("ZuCo hold-out train", "ZuCo_SST_data/train.csv"),
    ("ZuCo hold-out valid", "ZuCo_SST_data/valid.csv"),
    ("ZuCo hold-out test", "ZuCo_SST_data/test.csv"),
    ("SST combined", "SST_data/combined_full_sst_et.csv"),
    ("SST train", "SST_data/train_full_sst.csv"),
    ("SST valid", "SST_data/valid_full_sst.csv"),
    ("SST test", "SST_data/test_full_sst.csv"),
)


def counts_for(rel: str) -> Counter:
    _cols, rows = read_rows(rel)
    return Counter(as_int(row["sentiment_label"]) for row in rows)


def stacked_bar(counts: Counter, width: int = 40) -> str:
    total = sum(counts.values()) or 1
    chars = {0: "N", 1: ".", 2: "P"}
    pieces = []
    used = 0
    for lab in (0, 1, 2):
        n = max(0, round(width * counts.get(lab, 0) / total))
        pieces.append(chars[lab] * n)
        used += n
    # keep width stable after rounding
    bar = "".join(pieces)[:width].ljust(width)
    return f"[{bar}]"


def main() -> int:
    print("Label distribution (0=neg, 1=neu, 2=pos)")
    print(f"repo root: {abs_path('')}")
    print()
    print(f"{'table':<28}{'n':>6}  {'breakdown':<52}bar")
    print("-" * 110)

    zuco_holdout = Counter()
    for title, rel in LABELED:
        counts = counts_for(rel)
        n = sum(counts.values())
        print(f"{title:<28}{n:>6}  {format_label_counts(counts):<52}{stacked_bar(counts)}")
        if title.startswith("ZuCo hold-out"):
            zuco_holdout += counts

    print()
    print("Hold-out vs 5-fold reminder")
    print("  The 320/40/40 ZuCo CSVs were split without stratify=.")
    print("  model_ZuCo_SST.py ignores them and uses StratifiedKFold on the 400-row file.")
    print(f"  Re-sum of hold-out parts: {format_label_counts(zuco_holdout)}")
    print()
    print("Bar key: N = negative, . = neutral, P = positive (scaled to 40 chars).")
    print("Full SST is ~2:1:2; ZuCo is close to balanced.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
