#!/usr/bin/env python3
"""Reconstruct the ZuCo text ⨝ averaged-gaze join and report alignment.

The combined training table is a left-join of ssts_ZuCo.csv onto the
scaled sentence-level average on sentence_id == id. This script rebuilds
that join in memory and compares it to the checked-in combined files.

    python3 examples/sentence_gaze_join.py
"""

from __future__ import annotations

import sys
from typing import Dict, List, Tuple

from common import (
    ZUCO_GAZE_COLS,
    as_int,
    format_label_counts,
    label_counts,
    read_rows,
    truncate,
)


def index_by_id(rows: List[Dict[str, str]], column: str) -> Dict[int, Dict[str, str]]:
    out: Dict[int, Dict[str, str]] = {}
    for row in rows:
        key = as_int(row[column])
        if key in out:
            raise ValueError(f"duplicate {column}={key}")
        out[key] = row
    return out


def compare_numeric(a: str, b: str, atol: float = 1e-6) -> bool:
    return abs(float(a) - float(b)) <= atol


def diff_combined(
    combined_rel: str,
    gaze_rel: str,
    gaze_cols: Tuple[str, ...],
) -> List[str]:
    _c, text_rows = read_rows("ZuCo_SST_data/ssts_ZuCo.csv")
    _c, gaze_rows = read_rows(gaze_rel)
    _c, comb_rows = read_rows(combined_rel)

    text = index_by_id(text_rows, "sentence_id")
    gaze = index_by_id(gaze_rows, "id")
    comb = index_by_id(comb_rows, "sentence_id")

    issues: List[str] = []
    text_ids = set(text)
    gaze_ids = set(gaze)
    comb_ids = set(comb)

    if text_ids != gaze_ids:
        issues.append(
            f"text vs {gaze_rel}: only-text={sorted(text_ids - gaze_ids)[:8]} "
            f"only-gaze={sorted(gaze_ids - text_ids)[:8]}"
        )
    if text_ids != comb_ids:
        issues.append(
            f"text vs {combined_rel}: only-text={sorted(text_ids - comb_ids)[:8]} "
            f"only-combined={sorted(comb_ids - text_ids)[:8]}"
        )

    sentence_mismatch = 0
    label_mismatch = 0
    gaze_mismatch = 0
    for sid in sorted(text_ids & comb_ids):
        t = text[sid]
        c = comb[sid]
        if t["sentence"].strip() != c["sentence"].strip():
            sentence_mismatch += 1
        if as_int(t["sentiment_label"]) != as_int(c["sentiment_label"]):
            label_mismatch += 1
        if sid in gaze:
            g = gaze[sid]
            for col in gaze_cols:
                if not compare_numeric(c[col], g[col]):
                    gaze_mismatch += 1
                    break

    if sentence_mismatch:
        issues.append(f"{combined_rel}: {sentence_mismatch} sentences differ from ssts_ZuCo.csv")
    if label_mismatch:
        issues.append(f"{combined_rel}: {label_mismatch} labels differ from ssts_ZuCo.csv")
    if gaze_mismatch:
        issues.append(f"{combined_rel}: {gaze_mismatch} rows differ from {gaze_rel} on {gaze_cols}")
    return issues


def preview_join(n: int = 5) -> None:
    _c, text_rows = read_rows("ZuCo_SST_data/ssts_ZuCo.csv")
    _c, gaze_rows = read_rows("ZuCo_et_csv_data/standard_scaled_average_data.csv")
    text = index_by_id(text_rows, "sentence_id")
    gaze = index_by_id(gaze_rows, "id")
    print("Sample reconstructed rows (standard-scaled gaze)")
    print(f"{'id':>4} {'lab':>3} {'nFix':>8} {'FFD':>8} {'GPT':>8} {'TRT':>8} {'GD':>8}  sentence")
    for sid in range(n):
        t = text[sid]
        g = gaze[sid]
        print(
            f"{sid:>4} {as_int(t['sentiment_label']):>3} "
            f"{float(g['nFixations']):8.3f} {float(g['FFD']):8.3f} "
            f"{float(g['GPT']):8.3f} {float(g['TRT']):8.3f} {float(g['GD']):8.3f}  "
            f"{truncate(t['sentence'], 56)}"
        )


def main() -> int:
    print("ZuCo sentence_id join check")
    print()
    preview_join()
    print()

    issues: List[str] = []
    issues.extend(
        diff_combined(
            "ZuCo_SST_data/combined_sst_et_standard.csv",
            "ZuCo_et_csv_data/standard_scaled_average_data.csv",
            ZUCO_GAZE_COLS,
        )
    )
    issues.extend(
        diff_combined(
            "ZuCo_SST_data/combined_sst_et_min_max.csv",
            "ZuCo_et_csv_data/min_max_scaled_average_data.csv",
            ZUCO_GAZE_COLS,
        )
    )

    _c, comb = read_rows("ZuCo_SST_data/combined_sst_et_standard.csv")
    print("Combined standard label counts:", format_label_counts(label_counts(comb)))
    print()

    if issues:
        print(f"FOUND {len(issues)} alignment issue(s):")
        for item in issues:
            print(f"  - {item}")
        return 1

    print("OK  400 text rows, 400 standard gaze rows, 400 min-max gaze rows")
    print("    sentence strings, labels, and the five fusion features match.")
    print()
    print("model_ZuCo_SST.py can therefore treat each CSV row as one aligned example.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
