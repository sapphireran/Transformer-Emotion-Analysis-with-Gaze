#!/usr/bin/env python3
"""Confirm train/valid/test splits are a disjoint cover of the combined tables."""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from csvutil import label_counts, read_dicts, sentence_ids
from paths import (
    SST_COMBINED,
    SST_TEST,
    SST_TRAIN,
    SST_VALID,
    ZUCO_COMBINED_STD,
    ZUCO_TEST,
    ZUCO_TRAIN,
    ZUCO_VALID,
)


def _ids(path) -> list[str]:
    _, rows = read_dicts(path)
    return sentence_ids(rows)


def check_partition(name: str, combined, parts: dict[str, object]) -> list[str]:
    problems = []
    _, combined_rows = read_dicts(combined)
    combined_ids = sentence_ids(combined_rows)
    combined_set = set(combined_ids)
    if len(combined_ids) != len(combined_set):
        problems.append(f"{name} combined table has duplicate sentence_id values")

    seen: dict[str, str] = {}
    part_total = 0
    for part_name, path in parts.items():
        ids = _ids(path)
        part_total += len(ids)
        dupes = [item for item, count in Counter(ids).items() if count > 1]
        if dupes:
            problems.append(f"{name} {part_name} has duplicate ids: {dupes[:5]}")
        unknown = [item for item in ids if item not in combined_set]
        if unknown:
            problems.append(f"{name} {part_name} has {len(unknown)} ids not in combined")
        overlap = [item for item in ids if item in seen]
        if overlap:
            other = seen[overlap[0]]
            problems.append(
                f"{name} {part_name} overlaps {other} on {len(overlap)} ids "
                f"(e.g. {overlap[0]})"
            )
        for item in ids:
            seen.setdefault(item, part_name)

    if part_total != len(combined_ids):
        problems.append(
            f"{name} parts sum to {part_total} rows, combined has {len(combined_ids)}"
        )
    missing = combined_set - set(seen)
    if missing:
        problems.append(f"{name} combined has {len(missing)} ids in no split")
    return problems


def label_table() -> str:
    lines = ["Label counts"]
    for title, path in (
        ("SST combined", SST_COMBINED),
        ("SST train", SST_TRAIN),
        ("SST valid", SST_VALID),
        ("SST test", SST_TEST),
        ("ZuCo combined", ZUCO_COMBINED_STD),
        ("ZuCo train (unused by model_ZuCo_SST.py)", ZUCO_TRAIN),
        ("ZuCo valid (unused)", ZUCO_VALID),
        ("ZuCo test (unused)", ZUCO_TEST),
    ):
        _, rows = read_dicts(path)
        counts = label_counts(rows)
        lines.append(f"  {title}: {counts}  n={len(rows)}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    problems = []
    problems += check_partition(
        "SST",
        SST_COMBINED,
        {"train": SST_TRAIN, "valid": SST_VALID, "test": SST_TEST},
    )
    problems += check_partition(
        "ZuCo-holdout",
        ZUCO_COMBINED_STD,
        {"train": ZUCO_TRAIN, "valid": ZUCO_VALID, "test": ZUCO_TEST},
    )

    if not args.quiet:
        print(label_table())
        print()
        print("SST 9482 + 1185 + 1186 = 11853")
        print("ZuCo 320 + 40 + 40 = 400")
        print(
            "Note: model_ZuCo_SST.py ignores the ZuCo hold-out and uses "
            "StratifiedKFold on the combined file."
        )

    if problems:
        print("SPLIT INTEGRITY FAILED:")
        for item in problems:
            print(" -", item)
        return 1

    print("split integrity OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
