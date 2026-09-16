#!/usr/bin/env python3
"""Summarize the word-level ET tables the gaze-prediction side uses."""

from __future__ import annotations

import sys
from pathlib import Path

_EXAMPLES = Path(__file__).resolve().parent
if str(_EXAMPLES) not in sys.path:
    sys.path.insert(0, str(_EXAMPLES))

import pandas as pd

from paths import (
    PRED_TEST,
    PRED_TEST_V2,
    PROVO,
    REPO_ROOT,
    WORD_AVERAGES_V2,
    subject_word_csv,
)


def _rel(path: Path) -> str:
    return str(path.relative_to(REPO_ROOT))


def _numeric_preview(df: pd.DataFrame, cols: list[str]) -> None:
    present = [c for c in cols if c in df.columns]
    if not present:
        print("  (requested numeric columns not in this file)")
        return
    stats = df[present].astype(float).describe().T[["mean", "std", "min", "max"]]
    print(stats.to_string(float_format=lambda v: f"{v:10.4f}"))


def preview(path: Path, sentence_col: str, feature_cols: list[str]) -> None:
    print(f"=== {_rel(path)} ===")
    if not path.is_file():
        print("  MISSING\n")
        return
    df = pd.read_csv(path)
    print(f"  rows={len(df):,}  cols={list(df.columns)}")
    if sentence_col in df.columns:
        n_sent = df[sentence_col].nunique()
        print(f"  unique {sentence_col}={n_sent:,}")
        sizes = df.groupby(sentence_col).size()
        print(
            f"  tokens/sentence: min={int(sizes.min())}  "
            f"median={float(sizes.median()):.1f}  max={int(sizes.max())}"
        )
    if "word" in df.columns:
        empty = int((df["word"].astype(str).str.len() == 0).sum())
        print(f"  unique words={df['word'].nunique():,}  empty-word rows={empty}")
    elif "Word" in df.columns:
        empty = int(df["Word"].isna().sum())
        print(f"  unique Word={df['Word'].nunique():,}  null Word={empty}")
    print("  feature ranges:")
    _numeric_preview(df, feature_cols)
    print()
    print("  first 5 rows:")
    preview_cols = [c for c in df.columns if c in (
        ["sentence_id", "word_id", "word", "Sent_ID", "Word_ID", "Word"] + feature_cols
    )]
    print(df[preview_cols].head(5).to_string(index=False))
    print()


def main() -> int:
    preview(
        subject_word_csv(1),
        "Sent_ID",
        ["nFixations", "FFD", "GPT", "TRT", "GD", "SFD", "meanPupilSize", "WordLen"],
    )
    preview(
        WORD_AVERAGES_V2,
        "Sent_ID",
        ["nFixations", "FFD", "GPT", "TRT", "GD", "SFD", "meanPupilSize", "WordLen"],
    )
    preview(
        PRED_TEST,
        "sentence_id",
        ["nFix", "FFD", "GPT", "TRT", "GD"],
    )
    # prediction_test_v2 is ~192k rows; only stats, not a huge print.
    print(f"=== {_rel(PRED_TEST_V2)} ===")
    pred = pd.read_csv(PRED_TEST_V2)
    print(f"  rows={len(pred):,}  unique sentence_id={pred['sentence_id'].nunique():,}")
    print(
        "  tokens/sentence: "
        f"min={int(pred.groupby('sentence_id').size().min())}  "
        f"median={float(pred.groupby('sentence_id').size().median()):.1f}  "
        f"max={int(pred.groupby('sentence_id').size().max())}"
    )
    print("  feature ranges:")
    _numeric_preview(pred, ["nFix", "FFD", "GPT", "TRT", "GD"])
    print()

    preview(
        PROVO,
        "sentence_id",
        ["nFix", "FFD", "GPT", "TRT", "fixProp"],
    )
    print(
        "PROVO uses fixProp (percent of readers who fixated the word) "
        "instead of GD. Do not concatenate it into the 5-d SST/ZuCo vector "
        "without renaming."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
