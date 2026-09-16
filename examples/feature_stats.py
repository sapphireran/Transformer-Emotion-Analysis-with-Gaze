#!/usr/bin/env python3
"""Min / mean / max / zero-rate for gaze columns on the joined tables.

This is the personal sanity check before fine-tuning: standard-scaled ZuCo
should sit near mean 0, min-max ZuCo should sit in [0, 1], and full-SST
predicted gaze should not be a column of zeros (that would mean the
placeholder table was used by mistake).

    python3 examples/feature_stats.py
"""

from __future__ import annotations

import sys

from common import (
    SST_GAZE_COLS,
    ZUCO_GAZE_COLS,
    ZUCO_JOINED_GAZE_ALL,
    column_stats,
    fmt_stat_table,
    read_rows,
)

REPORTS = (
    (
        "ZuCo combined (standard) — five fusion features",
        "ZuCo_SST_data/combined_sst_et_standard.csv",
        ZUCO_GAZE_COLS,
    ),
    (
        "ZuCo combined (standard) — unused extras",
        "ZuCo_SST_data/combined_sst_et_standard.csv",
        ("omissionRate", "meanPupilSize", "SFD"),
    ),
    (
        "ZuCo combined (min-max) — five fusion features",
        "ZuCo_SST_data/combined_sst_et_min_max.csv",
        ZUCO_GAZE_COLS,
    ),
    (
        "SST train — predicted gaze (what model_full_SST.py reads)",
        "SST_data/train_full_sst.csv",
        SST_GAZE_COLS,
    ),
    (
        "SST test — predicted gaze",
        "SST_data/test_full_sst.csv",
        SST_GAZE_COLS,
    ),
    (
        "ZuCo word averages v2 — raw-ish means",
        "ZuCo_et_csv_data/word/word_averages_v2.csv",
        ("nFixations", "FFD", "GPT", "TRT", "GD", "SFD", "meanPupilSize", "WordLen"),
    ),
    (
        "PROVO word table",
        "gaze_prediction/data/provo.csv",
        ("nFix", "FFD", "GPT", "TRT", "fixProp"),
    ),
)


def pearson(xs: list, ys: list) -> float:
    n = len(xs)
    if n < 3:
        return float("nan")
    mx = sum(xs) / n
    my = sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    denx = sum((x - mx) ** 2 for x in xs) ** 0.5
    deny = sum((y - my) ** 2 for y in ys) ** 0.5
    if denx == 0 or deny == 0:
        return float("nan")
    return num / (denx * deny)


def correlation_block() -> str:
    """Cheap pairwise correlations among ZuCo fusion features + label."""
    _cols, rows = read_rows("ZuCo_SST_data/combined_sst_et_standard.csv")
    names = list(ZUCO_GAZE_COLS) + ["sentiment_label"]
    series = {name: [float(row[name]) for row in rows] for name in names}
    width = 12
    header = " " * 14 + "".join(f"{n[:10]:>{width}}" for n in names)
    lines = ["Pairwise Pearson on ZuCo combined (standard), including the label", header]
    for a in names:
        cells = []
        for b in names:
            cells.append(f"{pearson(series[a], series[b]):{width}.3f}")
        lines.append(f"{a[:14]:<14}" + "".join(cells))
    lines.append(
        "A large |r| between a gaze column and sentiment_label would be "
        "surprising but useful. Large |r| among TRT/GD/GPT is expected."
    )
    return "\n".join(lines)


def unused_warning() -> str:
    unused = [c for c in ZUCO_JOINED_GAZE_ALL if c not in ZUCO_GAZE_COLS]
    return (
        "Joined ZuCo tables also carry "
        + ", ".join(unused)
        + " — present in the CSV, ignored by both training scripts."
    )


def main() -> int:
    print("Gaze feature moments")
    print()
    for title, rel, cols in REPORTS:
        print(title)
        print(f"  file: {rel}")
        _header, rows = read_rows(rel)
        print(fmt_stat_table(column_stats(rows, cols)))
        print()

    print(unused_warning())
    print()
    print(correlation_block())
    print()
    print("Interpretation hints")
    print("  • Standard-scaled means near 0, std near 1: scaler was applied.")
    print("  • Min-max mins near 0 and maxs near 1: scaler was applied.")
    print("  • SST train zeros should be rare; a 100% zero column is the placeholder table.")
    print("  • Word-level nFixations = 0 is a skipped word, not a missing file.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
