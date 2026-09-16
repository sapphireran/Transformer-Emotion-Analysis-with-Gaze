#!/usr/bin/env python3
"""Check that the committed splits are a partition, not a leaky shuffle.

Catches:
  - overlapping sentence_id across train / valid / test
  - rows that vanished between combined and the three split files
  - ZuCo leftover 320/40/40 vs the 400-row combined table
  - full SST 9482+1185+1186 vs 11853
  - duplicate sentence_id inside a single file

This does *not* claim the ZuCo leftover split is what model_ZuCo_SST.py
uses (it is not; that script re-folds). It only checks the CSVs.

    python3 examples/08_split_integrity.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import as_int_column, read_csv, table, write_text


def ids_of(relative: str) -> list[int]:
    header, rows = read_csv(relative)
    if "sentence_id" in header:
        return as_int_column(rows, header, "sentence_id").tolist()
    if "id" in header:
        return as_int_column(rows, header, "id").tolist()
    raise KeyError(relative)


def report_split(name: str, combined: str, parts: dict[str, str]) -> tuple[list[str], int]:
    errors = 0
    lines = [f"## {name}", f"combined: {combined}"]
    combined_ids = ids_of(combined)
    combined_set = set(combined_ids)
    rows = []
    all_parts: list[int] = []
    seen: dict[int, str] = {}
    overlap_notes = []
    for label, path in parts.items():
        ids = ids_of(path)
        dup = len(ids) - len(set(ids))
        missing = sum(1 for i in ids if i not in combined_set)
        rows.append((label, path, str(len(ids)), str(dup), str(missing)))
        for i in ids:
            if i in seen:
                overlap_notes.append(f"id {i} in {seen[i]} and {label}")
            else:
                seen[i] = label
        all_parts.extend(ids)

    lines.append(table(["split", "path", "n", "internal dups", "not in combined"], rows))
    extra = sorted(set(all_parts) - combined_set)
    vanished = sorted(combined_set - set(all_parts))
    lines.append(f"union of splits: {len(set(all_parts))} unique / {len(all_parts)} rows")
    lines.append(f"combined unique: {len(combined_set)} / {len(combined_ids)} rows")
    if overlap_notes:
        errors += 1
        lines.append(f"OVERLAP ({len(overlap_notes)}), first five: {overlap_notes[:5]}")
    else:
        lines.append("no cross-split sentence_id overlap")
    if extra:
        errors += 1
        lines.append(f"ids in splits but not combined: {len(extra)}")
    else:
        lines.append("no split ids outside combined")
    if vanished:
        errors += 1
        lines.append(f"ids in combined missing from splits: {len(vanished)} e.g. {vanished[:8]}")
    else:
        lines.append("every combined id appears in exactly one split (by set)")
    if len(combined_ids) != len(combined_set):
        errors += 1
        lines.append(f"combined itself has duplicate ids: {len(combined_ids) - len(combined_set)}")
    lines.append("")
    return lines, errors


def main() -> int:
    blocks: list[str] = ["# Split integrity", ""]
    total_err = 0

    zuco_lines, e1 = report_split(
        "ZuCo leftover 80/10/10 (NOT used by model_ZuCo_SST.py)",
        "ZuCo_SST_data/combined_sst_et_standard.csv",
        {
            "train": "ZuCo_SST_data/train.csv",
            "valid": "ZuCo_SST_data/valid.csv",
            "test": "ZuCo_SST_data/test.csv",
        },
    )
    sst_lines, e2 = report_split(
        "Full SST 80/10/10 (used by model_full_SST.py)",
        "SST_data/combined_full_sst_et.csv",
        {
            "train": "SST_data/train_full_sst.csv",
            "valid": "SST_data/valid_full_sst.csv",
            "test": "SST_data/test_full_sst.csv",
        },
    )
    total_err += e1 + e2

    # Label file vs combined
    label_ids = set(ids_of("ZuCo_SST_data/ssts_ZuCo.csv"))
    zuco_ids = set(ids_of("ZuCo_SST_data/combined_sst_et_standard.csv"))
    extra_blocks = [
        "## label table vs combined ZuCo",
        f"ssts_ZuCo.csv unique ids: {len(label_ids)}",
        f"combined_sst_et_standard.csv unique ids: {len(zuco_ids)}",
    ]
    if label_ids != zuco_ids:
        total_err += 1
        extra_blocks.append(
            f"MISMATCH only-in-labels={sorted(label_ids - zuco_ids)[:8]} "
            f"only-in-combined={sorted(zuco_ids - label_ids)[:8]}"
        )
    else:
        extra_blocks.append("same sentence_id set")

    std_ids = ids_of("ZuCo_SST_data/combined_sst_et_standard.csv")
    mm_ids = ids_of("ZuCo_SST_data/combined_sst_et_min_max.csv")
    extra_blocks.append("")
    extra_blocks.append("## standard vs min-max combined")
    if std_ids == mm_ids:
        extra_blocks.append("identical sentence_id order and values")
    else:
        extra_blocks.append(
            f"order-equal={std_ids == mm_ids} set-equal={set(std_ids) == set(mm_ids)}"
        )
        if std_ids != mm_ids:
            total_err += 1

    extra_blocks.append("")
    extra_blocks.append("## reminder")
    extra_blocks.append(
        "model_ZuCo_SST.py ignores train.csv/valid.csv/test.csv and runs "
        "StratifiedKFold on the 400-row combined table."
    )

    lines = blocks + zuco_lines + sst_lines + extra_blocks + [""]
    if total_err:
        lines.append(f"FAILURES: {total_err}")
        code = 1
    else:
        lines.append("ALL SPLIT CHECKS PASSED")
        code = 0
    text = "\n".join(lines)
    print(text)
    write_text("08_split_integrity.txt", text)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
