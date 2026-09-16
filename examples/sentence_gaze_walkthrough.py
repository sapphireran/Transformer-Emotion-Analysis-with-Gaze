#!/usr/bin/env python3
"""Print one ZuCo sentence from text → words → subject rows → fusion vector.

Default is sentence 3, the row documented in docs/worked-example.md.

Usage:
    python3 examples/sentence_gaze_walkthrough.py
    python3 examples/sentence_gaze_walkthrough.py --sentence-id 4
    python3 examples/sentence_gaze_walkthrough.py --sentence-id 3 --subjects 1,2
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from examples.lib.loaders import (
    FUSION_GAZE_COLUMNS,
    LABEL_NAMES,
    load_csv,
    load_zuco_experiment,
    sentence_text,
    word_rows_for_sentence,
)
from examples.lib.paths import spec_by_key


WORD_PRINT = ["Word_ID", "Word", "nFixations", "GD", "TRT", "FFD", "SFD", "GPT", "WordLen"]


def _fmt_table(df: pd.DataFrame, columns: list[str]) -> str:
    present = [c for c in columns if c in df.columns]
    work = df[present].copy()
    for col in present:
        if col in {"Word", "Sent_ID"}:
            continue
        if pd.api.types.is_numeric_dtype(work[col]):
            if col in {"Word_ID", "WordLen"}:
                work[col] = work[col].astype(int)
            else:
                work[col] = work[col].map(lambda x: f"{float(x):.2f}")
    return work.to_string(index=False)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sentence-id", type=int, default=3)
    parser.add_argument(
        "--subjects",
        default="1,2",
        help="comma-separated 1-based subject ids to print word rows for",
    )
    args = parser.parse_args(argv)

    sid = args.sentence_id
    text, label = sentence_text(sid)
    print(f"sentence_id: {sid}")
    print(f"label:       {label} ({LABEL_NAMES[label]})")
    print(f"text:        {text}")
    print("gaze source: measured (ZuCo Task 1, 12-subject mean)")
    print()

    words = word_rows_for_sentence(sid, source="average")
    if words.empty:
        print(f"No word-level rows for sentence {sid}.", file=sys.stderr)
        return 1
    print(f"Word-level subject mean ({spec_by_key('zuco_word_average').relative})")
    print(_fmt_table(words, WORD_PRINT))
    print()

    subjects = [int(s) for s in args.subjects.split(",") if s.strip()]
    for subject in subjects:
        rows = word_rows_for_sentence(sid, source="subject", subject=subject)
        skips = int((rows["nFixations"].astype(float) == 0).sum())
        print(f"Subject {subject} ({skips} skipped word(s))")
        print(_fmt_table(rows, WORD_PRINT))
        print()

    avg = load_csv("zuco_sentence_average")
    raw = avg.loc[avg["id"].astype(int) == sid]
    if raw.empty:
        print("No sentence-level average row.", file=sys.stderr)
        return 1
    rec = raw.iloc[0]
    print("Sentence-level subject mean (raw units)")
    for col in (
        "SentLen",
        "omissionRate",
        "nFixations",
        "meanPupilSize",
        "GD",
        "TRT",
        "FFD",
        "SFD",
        "GPT",
    ):
        print(f"  {col:16} {float(rec[col]):.4f}")
    print()

    std = load_zuco_experiment("standard")
    row = std.loc[std["sentence_id"] == sid].iloc[0]
    print("Fusion vector read by model_ZuCo_SST.py (z-scores)")
    for col in FUSION_GAZE_COLUMNS:
        print(f"  {col:16} {float(row[col]):+.4f}")
    print("Dropped by the trainer:")
    for col in ("omissionRate", "meanPupilSize", "SFD"):
        print(f"  {col:16} {float(row[col]):+.4f}")
    print()
    print(
        "These five numbers are concatenated with a 768-d pooler vector.\n"
        "They describe reading effort, not polarity. See docs/worked-example.md."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
