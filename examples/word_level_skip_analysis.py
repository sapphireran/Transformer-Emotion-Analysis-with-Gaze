#!/usr/bin/env python3
"""How often do ZuCo subjects skip a word?

A skip is an all-zero gaze row with a real token. The sentence-level
averager treats those zeros as missing; the word-level averager keeps
them. This script reports the skip rate both ways.

Usage:
    python3 examples/word_level_skip_analysis.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from examples.lib.loaders import load_csv
from examples.lib.paths import subject_word_csv

GAZE = ["nFixations", "meanPupilSize", "GD", "TRT", "FFD", "SFD", "GPT"]


def is_skip(df: pd.DataFrame) -> pd.Series:
    numeric = df[GAZE].apply(pd.to_numeric, errors="coerce").fillna(0)
    return (numeric == 0).all(axis=1)


def main() -> int:
    print(f"{'subject':>8} {'rows':>8} {'skips':>8} {'rate':>8} {'unique skipped types':>22}")
    rates = []
    for subject in range(1, 13):
        df = pd.read_csv(subject_word_csv(subject))
        skip = is_skip(df)
        words = df.loc[skip, "Word"].fillna("").astype(str)
        # Ignore empty / unknown placeholders when listing types.
        types = sorted({w.lower() for w in words if w and w.lower() != "unknown"})
        rate = float(skip.mean())
        rates.append(rate)
        print(
            f"{subject:8d} {len(df):8d} {int(skip.sum()):8d} {rate:8.3f} "
            f"{len(types):22d}"
        )

    print()
    print(f"mean skip rate across 12 subjects: {sum(rates) / len(rates):.3f}")
    print(f"min / max subject skip rate:       {min(rates):.3f} / {max(rates):.3f}")

    avg = load_csv("zuco_word_average")
    avg_skip = is_skip(avg)
    print(
        f"rows still all-zero after averaging: "
        f"{int(avg_skip.sum())} / {len(avg)} ({avg_skip.mean():.3f})"
    )
    print(
        "A much smaller averaged skip rate is expected: a word only stays\n"
        "all-zero if *every* subject skipped it."
    )

    # Short tokens dominate skips.
    print("\nMost-skipped surface forms (subject 1, top 15)")
    s1 = pd.read_csv(subject_word_csv(1))
    skipped = s1.loc[is_skip(s1), "Word"].fillna("unknown").astype(str).str.lower()
    print(skipped.value_counts().head(15).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
