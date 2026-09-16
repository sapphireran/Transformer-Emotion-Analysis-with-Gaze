#!/usr/bin/env python3
"""Walk one ZuCo sentence per polarity class from text → words → ET vector.

This is the “what does a single training row actually contain?” example.
For each of negative / neutral / positive it prints:

1. the sentence and integer label
2. the 5-d sentence-level vector ``model_ZuCo_SST.py`` consumes
3. the extra sentence-level columns the model ignores
4. the matching word-level rows from ``word_averages_v2.csv``
"""

from __future__ import annotations

import sys
from pathlib import Path

_EXAMPLES = Path(__file__).resolve().parent
if str(_EXAMPLES) not in sys.path:
    sys.path.insert(0, str(_EXAMPLES))

import pandas as pd

from paths import (
    SENTIMENT_NAME,
    WORD_AVERAGES_V2,
    ZUCO_ET_COLS,
    ZUCO_STANDARD,
)


def _pick_rows(df: pd.DataFrame) -> pd.DataFrame:
    """One mid-length sentence per class so the word table is readable."""
    picked = []
    for label in (0, 1, 2):
        subset = df[df["sentiment_label"] == label].copy()
        subset["n_chars"] = subset["sentence"].astype(str).str.len()
        # Prefer sentences that are long enough to have several words but
        # short enough to print, around the median length of that class.
        target = float(subset["n_chars"].median())
        subset["dist"] = (subset["n_chars"] - target).abs()
        picked.append(subset.sort_values("dist").iloc[0])
    return pd.DataFrame(picked)


def _word_rows(words: pd.DataFrame, sentence_id: int) -> pd.DataFrame:
    tag = f"{int(sentence_id)}_NR"
    hit = words[words["Sent_ID"].astype(str) == tag]
    if hit.empty:
        # Some dumps store Sent_ID as a bare integer.
        hit = words[words["Sent_ID"].astype(str).str.startswith(f"{int(sentence_id)}_")]
    return hit


def main() -> int:
    joined = pd.read_csv(ZUCO_STANDARD)
    words = pd.read_csv(WORD_AVERAGES_V2)
    extra_cols = [
        c
        for c in ("omissionRate", "meanPupilSize", "SFD")
        if c in joined.columns
    ]

    print(
        "ZuCo ∩ SST sentence walkthrough  "
        f"(joined n={len(joined)}, word-average n={len(words)})"
    )
    print()

    for _, row in _pick_rows(joined).iterrows():
        sid = int(row["sentence_id"])
        label = int(row["sentiment_label"])
        print("=" * 72)
        print(f"sentence_id={sid}  label={label} ({SENTIMENT_NAME[label]})")
        print(row["sentence"])
        print()
        print("5-d vector consumed by EyeTrackingModel  (z-scored sentence ET):")
        for col in ZUCO_ET_COLS:
            print(f"  {col:<12} {float(row[col]):+.4f}")
        if extra_cols:
            print("columns present in the CSV but not passed to the classifier:")
            for col in extra_cols:
                print(f"  {col:<12} {float(row[col]):+.4f}")
        print()

        wdf = _word_rows(words, sid)
        if wdf.empty:
            print("  (no word-level rows found for this sentence_id)")
            print()
            continue
        show = wdf[
            [c for c in ("Word_ID", "Word", "nFixations", "FFD", "GPT", "TRT", "GD", "WordLen") if c in wdf.columns]
        ].copy()
        print(f"word-level average ({len(show)} tokens, subject-mean milliseconds):")
        print(show.to_string(index=False, float_format=lambda v: f"{v:8.2f}"))
        print()
        print(
            "sentence-level nFixations / FFD / … in the join are *not* the "
            "sum of this table: DataTransformer averages per fixated word, "
            "then get_average_sentence_level.py averages across subjects "
            "and z-scores. See docs/eye-tracking-features.md."
        )
        print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
