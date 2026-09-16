#!/usr/bin/env python3
"""Word-level gaze coverage: skip rate, sentence lengths, name overlap.

Useful when you later try word-piece alignment. Right now the classifiers
are sentence-level only; this script is the personal map of the word CSVs.

    python3 examples/word_level_preview.py
"""

from __future__ import annotations

import sys
from collections import Counter, defaultdict
from typing import Dict, List

from common import read_rows


def zuco_word_report() -> None:
    _c, rows = read_rows("ZuCo_et_csv_data/word/word_averages_v2.csv")
    per_sent: Dict[str, List[dict]] = defaultdict(list)
    skipped = 0
    unknown = 0
    for row in rows:
        per_sent[row["Sent_ID"]].append(row)
        if float(row["nFixations"]) == 0.0:
            skipped += 1
        if (row["Word"] or "").lower() in {"", "unknown"}:
            unknown += 1

    lengths = [len(v) for v in per_sent.values()]
    print("ZuCo word averages v2")
    print(f"  words                 {len(rows)}")
    print(f"  sentences             {len(per_sent)}")
    print(f"  words / sentence      min={min(lengths)} mean={sum(lengths)/len(lengths):.2f} max={max(lengths)}")
    print(f"  nFixations == 0       {skipped} ({100.0 * skipped / len(rows):.1f}%)  [skipped / unfixated]")
    print(f"  Word empty/unknown    {unknown}")
    print()

    # Show the sentence with the highest skip rate.
    worst_id = max(
        per_sent,
        key=lambda sid: sum(1 for r in per_sent[sid] if float(r["nFixations"]) == 0.0)
        / len(per_sent[sid]),
    )
    worst = per_sent[worst_id]
    skip_n = sum(1 for r in worst if float(r["nFixations"]) == 0.0)
    print(f"  highest skip-rate sentence: {worst_id}  {skip_n}/{len(worst)} words unfixated")
    print("   ", " ".join((r["Word"] or "_") for r in worst[:24]), "…" if len(worst) > 24 else "")
    print()


def provo_report() -> None:
    _c, rows = read_rows("gaze_prediction/data/provo.csv")
    sents = Counter(r["sentence_id"] for r in rows)
    print("PROVO word table")
    print(f"  words                 {len(rows)}")
    print(f"  sentences             {len(sents)}")
    print(f"  words / sentence      min={min(sents.values())} max={max(sents.values())}")
    print()


def sst_placeholder_vs_pred() -> None:
    """Same shape? How many placeholder zeros got filled in v2?"""
    # Stream both files; they are 191k rows. Compare nFix zero-rate only.
    from common import abs_path
    import csv

    def zero_rate(rel: str, col: str) -> tuple:
        path = abs_path(rel)
        n = 0
        z = 0
        sents = set()
        with open(path, newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                n += 1
                sents.add(row["sentence_id"])
                if float(row[col]) == 0.0:
                    z += 1
        return n, z, len(sents)

    p_n, p_z, p_s = zero_rate("SST_data/sst_et_test.csv", "nFix")
    v_n, v_z, v_s = zero_rate("gaze_prediction/data/prediction_test_v2.csv", "nFix")
    print("SST word placeholder vs predicted v2")
    print(f"  placeholder   rows={p_n} sentences={p_s} nFix==0 {p_z} ({100.0 * p_z / p_n:.1f}%)")
    print(f"  predicted v2  rows={v_n} sentences={v_s} nFix==0 {v_z} ({100.0 * v_z / v_n:.1f}%)")
    if p_n != v_n:
        print("  ROW COUNT MISMATCH — predictor output is not aligned to the placeholder table")
    else:
        print("  row counts match; v2 is a filled copy of the placeholder shape")
    print()


def subject3_gap() -> None:
    _c, s1 = read_rows("ZuCo_et_csv_data/word/1_SR.csv")
    _c, s3 = read_rows("ZuCo_et_csv_data/word/3_SR.csv")
    print("Subject 3 word-table gap")
    print(f"  subject 1 words       {len(s1)}")
    print(f"  subject 3 words       {len(s3)}  (delta={len(s1) - len(s3)})")
    print("  Do not positional-mean subject 3 with the others; join on Sent_ID + Word_ID.")
    print()


def main() -> int:
    zuco_word_report()
    provo_report()
    sst_placeholder_vs_pred()
    subject3_gap()
    return 0


if __name__ == "__main__":
    sys.exit(main())
