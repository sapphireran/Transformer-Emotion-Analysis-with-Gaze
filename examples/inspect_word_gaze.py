#!/usr/bin/env python3
"""Inspect word-level ZuCo gaze (subject 1 vs averaged v2).

Reports skip rate (nFixations == 0), go-past vs first-pass gaps, and the
Pearson correlation between character length and fixation count — a cheap
sanity check that longer words attract more fixations.

Usage (repo root):

    python examples/inspect_word_gaze.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from examples.common import (
    WORD_GAZE_COLS,
    ZUCO_WORD_DIR,
    format_stats_table,
    floats,
    read_dicts,
)


def _pearson(x: np.ndarray, y: np.ndarray) -> float:
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 2:
        return float("nan")
    return float(np.corrcoef(x[mask], y[mask])[0, 1])


def main() -> None:
    subject1 = read_dicts(ZUCO_WORD_DIR / "1_SR.csv")
    averaged = read_dicts(ZUCO_WORD_DIR / "word_averages_v2.csv")

    print("=== word-level row counts ===")
    for i in range(1, 13):
        rows = read_dicts(ZUCO_WORD_DIR / f"{i}_SR.csv")
        nfix = floats(rows, "nFixations")
        skipped = float((nfix == 0).mean())
        print(f"{i:2d}_SR.csv  words={len(rows):5d}  skip_rate={skipped:6.1%}")

    print()
    print(f"word_averages_v2.csv  words={len(averaged)}")
    print(format_stats_table(WORD_GAZE_COLS, [floats(averaged, c) for c in WORD_GAZE_COLS]))

    nfix = floats(averaged, "nFixations")
    ffd = floats(averaged, "FFD")
    gd = floats(averaged, "GD")
    trt = floats(averaged, "TRT")
    gpt = floats(averaged, "GPT")
    length = floats(averaged, "WordLen")

    print()
    print("=== skip / duration relations on the averaged table ===")
    print(f"skip rate (nFixations == 0): {(nfix == 0).mean():.1%}")
    print(f"share with TRT > GD:         {(trt > gd).mean():.1%}  (re-reading after first pass)")
    print(f"share with GPT > TRT:        {(gpt > trt).mean():.1%}  (regressions that leave the word)")
    print(f"corr(WordLen, nFixations):   {_pearson(length, nfix):.3f}")
    print(f"corr(WordLen, FFD):          {_pearson(length, ffd):.3f}")
    print(f"corr(GD, TRT):               {_pearson(gd, trt):.3f}")

    # Per-sentence word counts from Sent_ID (e.g. 0_NR)
    sent_ids = [row["Sent_ID"] for row in averaged]
    unique = sorted(set(sent_ids), key=lambda s: int(s.split("_")[0]))
    counts = [sum(1 for s in sent_ids if s == sid) for sid in unique]
    print()
    print(f"unique Sent_ID values: {len(unique)}")
    print(
        f"words per sentence: min={min(counts)} median={float(np.median(counts)):.1f} "
        f"max={max(counts)}"
    )

    print()
    print("=== subject 1 sample (first 8 rows) ===")
    print(f"{'Word':<16} {'nFix':>6} {'FFD':>7} {'GD':>7} {'TRT':>7} {'GPT':>7}")
    for row in subject1[:8]:
        print(
            f"{row['Word']:<16} {float(row['nFixations']):6.2f} "
            f"{float(row['FFD']):7.1f} {float(row['GD']):7.1f} "
            f"{float(row['TRT']):7.1f} {float(row['GPT']):7.1f}"
        )
    print()
    print("A row of all zeros on subject 1 is a skipped token, not a 0 ms fixation.")


if __name__ == "__main__":
    main()
