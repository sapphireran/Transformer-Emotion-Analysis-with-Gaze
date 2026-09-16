#!/usr/bin/env python3
"""Audit hold-out splits, class balance, and ZuCo subject coverage."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gaze_emotion.audit import (
    audit_splits,
    format_split_audit,
    label_entropy,
    majority_baseline,
    subject_coverage_notes,
    subject_sentence_counts,
)
from gaze_emotion.constants import FUSION_GAZE_FEATURES, FUSION_GAZE_FEATURES_FULL_SST
from gaze_emotion.datasets import load_and_summarize
from gaze_emotion.metrics import weighted_scores


def majority_predictions(n: int, majority_id: int) -> list[int]:
    return [majority_id] * n


def baseline_line(path: str) -> str:
    summary = load_and_summarize(path)
    counts = summary.label_counts
    majority_name = max(counts, key=counts.get)
    mapping = {"NEGATIVE": 0, "NEUTRAL": 1, "POSITIVE": 2}
    # Rebuild labels from the file so the dummy predictor is honest.
    from gaze_emotion.datasets import load_csv_rows

    labels = [int(row["sentiment_label"]) for row in load_csv_rows(path)]
    preds = majority_predictions(len(labels), mapping[majority_name])
    scores = weighted_scores(labels, preds)
    return (
        f"{path}: majority={majority_name} "
        f"acc={majority_baseline(counts):.3f} "
        f"entropy={label_entropy(counts):.3f} bits  "
        f"weighted_f1(always-{majority_name})={scores['f1']:.3f}"
    )


def main() -> int:
    print("Split and coverage audit")
    print("=" * 72)

    zuco = audit_splits(
        "ZuCo_SST_data/train.csv",
        "ZuCo_SST_data/valid.csv",
        "ZuCo_SST_data/test.csv",
    )
    full = audit_splits(
        "SST_data/train_full_sst.csv",
        "SST_data/valid_full_sst.csv",
        "SST_data/test_full_sst.csv",
    )
    print()
    print(format_split_audit("ZuCo-SST hold-out (spilt.py, seed 42)", zuco))
    print()
    print(format_split_audit("Full SST hold-out (spilt.py, seed 42)", full))

    print()
    print("Majority-class baselines")
    print("-" * 72)
    for path in (
        "ZuCo_SST_data/combined_sst_et_standard.csv",
        "ZuCo_SST_data/valid.csv",
        "SST_data/train_full_sst.csv",
        "SST_data/test_full_sst.csv",
    ):
        print(baseline_line(path))

    print()
    print("ZuCo Task-1 subject coverage")
    print("-" * 72)
    counts = subject_sentence_counts()
    for name, n in counts.items():
        print(f"  {name}: {n} sentences")
    for note in subject_coverage_notes(counts):
        print(f"  note: {note}")

    print()
    print("Fusion columns present?")
    print("-" * 72)
    zuco_cols = load_and_summarize("ZuCo_SST_data/combined_sst_et_standard.csv").columns
    sst_cols = load_and_summarize("SST_data/train_full_sst.csv").columns
    print(f"  ZuCo has {list(FUSION_GAZE_FEATURES)}: {set(FUSION_GAZE_FEATURES) <= set(zuco_cols)}")
    print(
        f"  Full SST has {list(FUSION_GAZE_FEATURES_FULL_SST)}: "
        f"{set(FUSION_GAZE_FEATURES_FULL_SST) <= set(sst_cols)}"
    )

    if zuco.has_leakage or full.has_leakage:
        print()
        print("ERROR: sentence_id leakage across splits.")
        return 1
    print()
    print("No sentence_id overlap in either hold-out triple.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
