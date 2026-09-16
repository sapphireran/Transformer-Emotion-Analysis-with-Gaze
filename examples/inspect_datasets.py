#!/usr/bin/env python3
"""Print row counts, columns, and sentiment histograms for the committed CSVs."""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

_EXAMPLES = Path(__file__).resolve().parent
if str(_EXAMPLES) not in sys.path:
    sys.path.insert(0, str(_EXAMPLES))

import pandas as pd

from paths import (
    FULL_SST_COMBINED,
    FULL_SST_RAW,
    FULL_SST_TEST,
    FULL_SST_TRAIN,
    FULL_SST_VALID,
    FULL_SST_WORD_TEMPLATE,
    PRED_TEST,
    PRED_TEST_V2,
    PROVO,
    REPO_ROOT,
    SENTIMENT_NAME,
    WORD_AVERAGES_V2,
    ZUCO_ET_AVERAGE,
    ZUCO_ET_DIR,
    ZUCO_ET_MINMAX,
    ZUCO_ET_STANDARD,
    ZUCO_MINMAX,
    ZUCO_STANDARD,
    ZUCO_TEST,
    ZUCO_TEXT,
    ZUCO_TRAIN,
    ZUCO_VALID,
    ZUCO_WORD_DIR,
    subject_sentence_csv,
    subject_word_csv,
)


def _rel(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _label_hist(series: pd.Series) -> str:
    counts = Counter(int(v) for v in series.dropna())
    parts = []
    for key in sorted(counts):
        name = SENTIMENT_NAME.get(key, str(key))
        parts.append(f"{key} {name}={counts[key]}")
    return ", ".join(parts) if parts else "(no integer labels)"


def describe(path: Path, label_col: str | None = None) -> None:
    if not path.is_file():
        print(f"MISSING  {_rel(path)}")
        return
    df = pd.read_csv(path)
    cols = ", ".join(df.columns.tolist())
    print(f"{_rel(path)}")
    print(f"  rows={len(df):,}  cols={len(df.columns)}  [{cols}]")
    if label_col and label_col in df.columns:
        print(f"  labels: {_label_hist(df[label_col])}")
    print()


def main() -> int:
    print("=== Full SST ===\n")
    describe(FULL_SST_RAW)
    describe(FULL_SST_COMBINED, "sentiment_label")
    describe(FULL_SST_TRAIN, "sentiment_label")
    describe(FULL_SST_VALID, "sentiment_label")
    describe(FULL_SST_TEST, "sentiment_label")
    describe(FULL_SST_WORD_TEMPLATE)

    print("=== ZuCo ∩ SST ===\n")
    describe(ZUCO_TEXT, "sentiment_label")
    describe(ZUCO_STANDARD, "sentiment_label")
    describe(ZUCO_MINMAX, "sentiment_label")
    describe(ZUCO_TRAIN, "sentiment_label")
    describe(ZUCO_VALID, "sentiment_label")
    describe(ZUCO_TEST, "sentiment_label")

    print("=== Per-subject sentence ET ===\n")
    for i in range(1, 13):
        describe(subject_sentence_csv(i))
    describe(ZUCO_ET_AVERAGE)
    describe(ZUCO_ET_MINMAX)
    describe(ZUCO_ET_STANDARD)

    print("=== Per-subject word ET (subject 1, 3, averages) ===\n")
    describe(subject_word_csv(1))
    describe(subject_word_csv(3))
    describe(WORD_AVERAGES_V2)
    extra_word = list(ZUCO_WORD_DIR.glob("*_SR.csv"))
    print(f"  word-level subject files on disk: {len(extra_word)}")
    print()

    print("=== Gaze-prediction / PROVO ===\n")
    describe(PRED_TEST)
    describe(PRED_TEST_V2)
    describe(PROVO)

    sentence_subjects = list(ZUCO_ET_DIR.glob("*_SR.csv"))
    print(
        f"Inventory complete. {len(sentence_subjects)} per-subject sentence "
        f"files under {_rel(ZUCO_ET_DIR)}."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
