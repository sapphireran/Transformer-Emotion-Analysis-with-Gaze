#!/usr/bin/env python3
"""Print label priors for every official split.

The full-SST splitter did not stratify, so valid/test priors drifted.
The ZuCo 80/10/10 files are unused by the CV trainer; they are still
listed so a 40-row hold-out is not mistaken for a balanced test set.

Usage:
    python3 examples/split_balance.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from examples.lib.loaders import LABEL_NAMES, label_counts, load_csv


GROUPS = (
    (
        "ZuCo ∩ SST (measured gaze)",
        ("zuco_standard", "zuco_train", "zuco_valid", "zuco_test"),
    ),
    (
        "Full SST (projected gaze)",
        ("sst_combined", "sst_train", "sst_valid", "sst_test"),
    ),
)


def main() -> int:
    for title, keys in GROUPS:
        print(title)
        print("=" * len(title))
        print(f"{'split':18} {'n':>6} {'neg':>8} {'neu':>8} {'pos':>8} {'majority':>10}")
        for key in keys:
            df = load_csv(key)
            counts = label_counts(df["sentiment_label"])
            n = sum(counts.values())
            shares = {k: counts[k] / n for k in (0, 1, 2)}
            maj_k = max(shares, key=shares.get)
            print(
                f"{key:18} {n:6d} "
                f"{counts[0]:4d} {shares[0]:3.0%} "
                f"{counts[1]:4d} {shares[1]:3.0%} "
                f"{counts[2]:4d} {shares[2]:3.0%} "
                f"{LABEL_NAMES[maj_k]:>6} {shares[maj_k]:3.0%}"
            )
        print()

    print("Notes")
    print("-----")
    print("* Chance if you always guess the majority class is the last column.")
    print("* model_ZuCo_SST.py ignores zuco_train/valid/test and uses 5-fold CV.")
    print("* SST_data/spilt.py did not pass stratify=, so full-SST priors drift.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
