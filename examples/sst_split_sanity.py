#!/usr/bin/env python3
"""Sanity-check SST / ZuCo splits: sizes, disjoint IDs, label sets.

Also warns that Track A `sentence_id` and Track B `sentence_id` are
different namespaces — joining them blindly is wrong.

Usage (repo root):

    python examples/sst_split_sanity.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from examples.common import (
    SST_DIR,
    SST_STRING_TO_INT,
    ZUCO_SST_DIR,
    int_col,
    label_histogram,
    read_dicts,
    read_sst_string_labels,
)


def _id_set(rows, column="sentence_id") -> set[int]:
    return {int(float(row[column])) for row in rows}


def _disjoint_report(parts: dict[str, set[int]]) -> None:
    names = list(parts)
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            overlap = parts[a] & parts[b]
            status = "OK disjoint" if not overlap else f"OVERLAP {len(overlap)}"
            print(f"  {a} ∩ {b}: {status}")


def main() -> None:
    print("=== Track A (human gaze, ~400 sentences) ===")
    combined = read_dicts(ZUCO_SST_DIR / "combined_sst_et_standard.csv")
    labels_only = read_dicts(ZUCO_SST_DIR / "ssts_ZuCo.csv")
    print(f"combined_sst_et_standard.csv  {len(combined)}")
    print(f"ssts_ZuCo.csv                 {len(labels_only)}")
    print(label_histogram(int_col(combined, "sentiment_label")))

    zuco_parts = {
        "train": _id_set(read_dicts(ZUCO_SST_DIR / "train.csv")),
        "valid": _id_set(read_dicts(ZUCO_SST_DIR / "valid.csv")),
        "test": _id_set(read_dicts(ZUCO_SST_DIR / "test.csv")),
    }
    print()
    print(
        "80/10/10 files: "
        + ", ".join(f"{k}={len(v)}" for k, v in zuco_parts.items())
        + f"  sum={sum(len(v) for v in zuco_parts.values())}"
    )
    _disjoint_report(zuco_parts)
    union = set.union(*zuco_parts.values())
    combined_ids = _id_set(combined)
    print(f"  union vs combined ids: {union == combined_ids} (equal={union == combined_ids})")
    if union != combined_ids:
        print(f"  missing from splits: {sorted(combined_ids - union)[:10]}")
        print(f"  extra in splits:     {sorted(union - combined_ids)[:10]}")

    text_ids = _id_set(labels_only)
    print(f"  ssts_ZuCo ids == combined ids: {text_ids == combined_ids}")

    print()
    print("=== Track B (predicted gaze, full SST) ===")
    raw = read_sst_string_labels(SST_DIR / "stts_all_sentence_level.csv")
    print(f"stts_all_sentence_level.csv   {len(raw)}  (no header, string labels)")
    string_hist = {}
    for _, lab in raw:
        string_hist[lab] = string_hist.get(lab, 0) + 1
    for lab, n in sorted(string_hist.items()):
        mapped = SST_STRING_TO_INT.get(lab, "UNMAPPED")
        print(f"  {lab:<10} n={n:5d}  → sentiment_label {mapped}")

    unknown = [lab for lab in string_hist if lab not in SST_STRING_TO_INT]
    if unknown:
        print(f"  WARNING unknown labels: {unknown}")

    sst_parts = {
        "train_full_sst": _id_set(read_dicts(SST_DIR / "train_full_sst.csv")),
        "valid_full_sst": _id_set(read_dicts(SST_DIR / "valid_full_sst.csv")),
        "test_full_sst": _id_set(read_dicts(SST_DIR / "test_full_sst.csv")),
    }
    print()
    print(
        "full SST files: "
        + ", ".join(f"{k}={len(v)}" for k, v in sst_parts.items())
        + f"  sum={sum(len(v) for v in sst_parts.values())}"
    )
    _disjoint_report(sst_parts)

    combined_full = read_dicts(SST_DIR / "combined_full_sst_et.csv")
    print(f"combined_full_sst_et.csv      {len(combined_full)}")
    print(label_histogram(int_col(combined_full, "sentiment_label")))

    print()
    print("=== ID namespace check (Track A vs Track B) ===")
    a_ids = combined_ids
    b_ids = _id_set(combined_full)
    overlap = a_ids & b_ids
    print(f"Track A ids: {min(a_ids)}–{max(a_ids)}  n={len(a_ids)}")
    print(f"Track B ids: {min(b_ids)}–{max(b_ids)}  n={len(b_ids)}")
    print(f"numeric overlap: {len(overlap)} (coincidence is expected; not a join key)")
    print()
    print("Do not merge combined_sst_et_standard.csv with *_full_sst.csv on")
    print("sentence_id. Track B ids index the larger SST dump; Track A ids")
    print("index the ZuCo trial after skip rules.")

    # Column presence for the fusion head
    print()
    print("=== fusion columns present? ===")
    zuco_cols = set(combined[0])
    sst_cols = set(combined_full[0])
    print("Track A needs nFixations, FFD, GPT, TRT, GD:",
          {"nFixations", "FFD", "GPT", "TRT", "GD"} <= zuco_cols)
    print("Track B needs nFix, FFD, GPT, TRT, GD:      ",
          {"nFix", "FFD", "GPT", "TRT", "GD"} <= sst_cols)
    print("Track A extra:", sorted(zuco_cols - sst_cols))
    print("Track B extra:", sorted(sst_cols - zuco_cols))


if __name__ == "__main__":
    main()
