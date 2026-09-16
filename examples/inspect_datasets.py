#!/usr/bin/env python3
"""Reprint dataset sizes, label counts, and gaze ranges from the CSVs.

This is the source of truth for the tables in README.md and docs/.
Run from the repository root:

    python examples/inspect_datasets.py

No third-party imports. Paths are relative to the current working
directory, not to this file, so that a stray `cd examples` fails
loudly instead of reading the wrong tree.
"""

from __future__ import annotations

import csv
import math
import os
import sys
from collections import Counter


# Files the two training tracks actually consume, plus the supporting
# tables the docs quote. A missing file is a warning, not a crash —
# this clone is allowed to grow without every intermediate CSV.
DATASETS = [
    {
        "path": "ZuCo_SST_data/combined_sst_et_standard.csv",
        "role": "Track A train file (5-fold CV reads this whole table)",
        "label": "sentiment_label",
        "text": "sentence",
        "gaze": ["nFixations", "FFD", "GPT", "TRT", "GD"],
    },
    {
        "path": "ZuCo_SST_data/train.csv",
        "role": "Unused 80% split (not read by model_ZuCo_SST.py)",
        "label": "sentiment_label",
        "text": "sentence",
        "gaze": ["nFixations", "FFD", "GPT", "TRT", "GD"],
    },
    {
        "path": "ZuCo_SST_data/valid.csv",
        "role": "Unused 10% split — note the unbalanced 40 rows",
        "label": "sentiment_label",
        "text": "sentence",
        "gaze": ["nFixations", "FFD", "GPT", "TRT", "GD"],
    },
    {
        "path": "ZuCo_SST_data/test.csv",
        "role": "Unused 10% split",
        "label": "sentiment_label",
        "text": "sentence",
        "gaze": ["nFixations", "FFD", "GPT", "TRT", "GD"],
    },
    {
        "path": "ZuCo_et_csv_data/average_data.csv",
        "role": "12-reader mean, raw milliseconds / counts",
        "label": None,
        "text": None,
        "gaze": [
            "omissionRate",
            "nFixations",
            "meanPupilSize",
            "GD",
            "TRT",
            "FFD",
            "SFD",
            "GPT",
        ],
    },
    {
        "path": "SST_data/combined_full_sst_et.csv",
        "role": "Track B source table before the 80/10/10 split",
        "label": "sentiment_label",
        "text": "sentence",
        "gaze": ["nFix", "FFD", "GPT", "TRT", "GD"],
    },
    {
        "path": "SST_data/train_full_sst.csv",
        "role": "Track B train",
        "label": "sentiment_label",
        "text": "sentence",
        "gaze": ["nFix", "FFD", "GPT", "TRT", "GD"],
    },
    {
        "path": "SST_data/valid_full_sst.csv",
        "role": "Track B valid (checkpoint on accuracy)",
        "label": "sentiment_label",
        "text": "sentence",
        "gaze": ["nFix", "FFD", "GPT", "TRT", "GD"],
    },
    {
        "path": "SST_data/test_full_sst.csv",
        "role": "Track B test (see last-batch bug in docs/05)",
        "label": "sentiment_label",
        "text": "sentence",
        "gaze": ["nFix", "FFD", "GPT", "TRT", "GD"],
    },
    {
        "path": "ZuCo_et_csv_data/word/word_averages_v2.csv",
        "role": "Word-level 12-reader mean (example walkthroughs)",
        "label": None,
        "text": None,
        "gaze": ["nFixations", "FFD", "GPT", "TRT", "GD"],
    },
    {
        "path": "gaze_prediction/data/prediction_test.csv",
        "role": "Predicted word-level gaze (different units)",
        "label": None,
        "text": None,
        "gaze": ["nFix", "FFD", "GPT", "TRT", "GD"],
    },
    {
        "path": "gaze_prediction/data/provo.csv",
        "role": "PROVO reference (has fixProp, no GD)",
        "label": None,
        "text": None,
        "gaze": ["nFix", "FFD", "GPT", "TRT", "fixProp"],
    },
]

LABEL_NAMES = {"0": "NEGATIVE", "1": "NEUTRAL", "2": "POSITIVE"}


def _f(value: str) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _mean_std(values: list[float]) -> tuple[float, float]:
    if not values:
        return float("nan"), float("nan")
    mean = sum(values) / len(values)
    var = sum((v - mean) ** 2 for v in values) / len(values)
    return mean, math.sqrt(var)


def summarize(spec: dict) -> None:
    path = spec["path"]
    print("=" * 72)
    print(path)
    print(spec["role"])
    if not os.path.isfile(path):
        print("  MISSING")
        return

    with open(path, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        columns = reader.fieldnames or []

    print(f"  rows={len(rows)}  cols={len(columns)}")

    label_col = spec["label"]
    if label_col and rows and label_col in rows[0]:
        counts = Counter(row[label_col] for row in rows)
        parts = []
        for key in sorted(counts):
            name = LABEL_NAMES.get(key, key)
            parts.append(f"{key}:{name}={counts[key]}")
        print("  labels  " + "  ".join(parts))

    text_col = spec["text"]
    if text_col and rows and text_col in rows[0]:
        lengths = [len(row[text_col].split()) for row in rows]
        print(
            "  words   "
            f"mean={sum(lengths)/len(lengths):.2f}  "
            f"min={min(lengths)}  max={max(lengths)}"
        )

    for col in spec["gaze"]:
        if not rows or col not in rows[0]:
            print(f"  {col:16s}  (column absent)")
            continue
        values = [v for v in (_f(row[col]) for row in rows) if v is not None]
        mean, std = _mean_std(values)
        print(
            f"  {col:16s}  "
            f"mean={mean:9.4f}  std={std:9.4f}  "
            f"min={min(values):9.4f}  max={max(values):9.4f}"
        )


def main() -> int:
    if not os.path.isdir("ZuCo_SST_data"):
        print(
            "Run this from the repository root "
            "(expected ./ZuCo_SST_data).",
            file=sys.stderr,
        )
        return 2
    for spec in DATASETS:
        summarize(spec)
    print("=" * 72)
    print("done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
