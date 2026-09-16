#!/usr/bin/env python3
"""Print a few concrete rows so the CSVs are not just schemas.

Picks one negative / neutral / positive ZuCo sentence, one long and one
short SST sentence, and a short word-level window from ZuCo and PROVO.

    python3 examples/sample_rows.py
"""

from __future__ import annotations

import sys
from typing import Dict, List, Optional

from common import (
    LABEL_NAME,
    SST_GAZE_COLS,
    ZUCO_GAZE_COLS,
    as_int,
    read_rows,
    truncate,
)


def first_with_label(rows: List[Dict[str, str]], label: int) -> Optional[Dict[str, str]]:
    for row in rows:
        if as_int(row["sentiment_label"]) == label:
            return row
    return None


def print_zuco_examples() -> None:
    _c, rows = read_rows("ZuCo_SST_data/combined_sst_et_standard.csv")
    print("ZuCo combined (standard-scaled gaze) — one sentence per class")
    print()
    for lab in (0, 1, 2):
        row = first_with_label(rows, lab)
        if row is None:
            continue
        print(f"  [{lab} {LABEL_NAME[lab]}] id={row['sentence_id']}")
        print(f"    {truncate(row['sentence'], 96)}")
        feats = "  ".join(f"{c}={float(row[c]):+.3f}" for c in ZUCO_GAZE_COLS)
        print(f"    {feats}")
        print()


def print_sst_length_extremes() -> None:
    _c, rows = read_rows("SST_data/train_full_sst.csv")
    by_len = sorted(rows, key=lambda r: len(r["sentence"]))
    short, long = by_len[0], by_len[-1]
    print("SST train — shortest and longest sentence (predicted gaze)")
    print()
    for title, row in (("shortest", short), ("longest", long)):
        print(f"  [{title}] id={row['sentence_id']} label={row['sentiment_label']} "
              f"chars={len(row['sentence'])}")
        print(f"    {truncate(row['sentence'], 96)}")
        feats = "  ".join(f"{c}={float(row[c]):.3f}" for c in SST_GAZE_COLS)
        print(f"    {feats}")
        print()


def print_word_window(rel: str, sent_key: str, sent_value: str, word_col: str, cols: List[str]) -> None:
    _c, rows = read_rows(rel)
    window = [r for r in rows if r[sent_key] == sent_value]
    print(f"{rel} — words in {sent_key}={sent_value!r} (n={len(window)})")
    if not window:
        print("  (none)")
        print()
        return
    header = f"  {'wid':>4} {'word':<16}" + "".join(f"{c:>10}" for c in cols)
    print(header)
    for row in window[:12]:
        word = (row[word_col] or "unknown")[:16]
        nums = "".join(f"{float(row[c]):10.2f}" for c in cols)
        print(f"  {row.get('word_id', row.get('Word_ID')):>4} {word:<16}{nums}")
    if len(window) > 12:
        print(f"  … {len(window) - 12} more words")
    print()


def main() -> int:
    print_zuco_examples()
    print_sst_length_extremes()
    print_word_window(
        "ZuCo_et_csv_data/word/word_averages_v2.csv",
        sent_key="Sent_ID",
        sent_value="0_NR",
        word_col="Word",
        cols=["nFixations", "FFD", "GPT", "TRT", "GD", "WordLen"],
    )
    print_word_window(
        "gaze_prediction/data/provo.csv",
        sent_key="sentence_id",
        sent_value="0",
        word_col="word",
        cols=["nFix", "FFD", "GPT", "TRT", "fixProp"],
    )
    print("These rows are illustrations, not a train split.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
