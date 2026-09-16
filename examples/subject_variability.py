#!/usr/bin/env python3
"""Spread of sentence-level gaze across the 12 ZuCo subjects.

The experiment tables collapse subjects first. This script shows how
much that collapse hides, using the five fusion columns in raw units.

Usage:
    python3 examples/subject_variability.py
    python3 examples/subject_variability.py --sentence-id 3
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from examples.lib.loaders import FUSION_GAZE_COLUMNS
from examples.lib.paths import subject_sentence_csv


def load_subjects() -> pd.DataFrame:
    frames = []
    for subject in range(1, 13):
        df = pd.read_csv(subject_sentence_csv(subject))
        df["subject"] = subject
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sentence-id",
        type=int,
        default=None,
        help="optional id to print 12 raw rows for",
    )
    args = parser.parse_args(argv)

    all_rows = load_subjects()
    print(f"loaded {len(all_rows)} rows (11 × 400 + 299 for subject 3)")

    print("\nBetween-subject std of the 400-sentence means, then mean of")
    print("within-sentence (across-subject) std — fusion columns, raw units.\n")
    print(f"{'column':16} {'mean':>10} {'std_of_subj_means':>18} {'mean_within_std':>16}")

    for col in FUSION_GAZE_COLUMNS:
        per_subject_mean = all_rows.groupby("subject")[col].mean()
        within = all_rows.groupby("id")[col].std(ddof=0)
        print(
            f"{col:16} {all_rows[col].mean():10.3f} "
            f"{per_subject_mean.std(ddof=0):18.3f} "
            f"{within.mean():16.3f}"
        )

    # Coefficient of variation per sentence, then median across sentences.
    print("\nMedian across sentences of (subject-std / subject-mean) for nFixations")
    g = all_rows.groupby("id")["nFixations"]
    cv = (g.std(ddof=0) / g.mean().replace(0, np.nan)).replace([np.inf, -np.inf], np.nan)
    print(f"  {float(cv.median()):.3f}  (values near 0.3+ mean the 12-subject mean is a blur)")

    if args.sentence_id is not None:
        sid = args.sentence_id
        rows = all_rows.loc[all_rows["id"].astype(int) == sid].sort_values("subject")
        if rows.empty:
            print(f"\nNo rows for sentence_id={sid}", file=sys.stderr)
            return 1
        print(f"\nSentence {sid} raw fusion columns by subject")
        cols = ["subject", *FUSION_GAZE_COLUMNS, "omissionRate"]
        print(rows[cols].to_string(index=False, float_format=lambda x: f"{x:8.2f}"))
        print(
            "\nThe experiment table stores only the column-wise mean (after a\n"
            "second scale). Leave-one-subject-out is a different experiment."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
