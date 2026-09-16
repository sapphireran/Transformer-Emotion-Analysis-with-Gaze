#!/usr/bin/env python3
"""Inventory every canonical CSV and check it against docs/datasets.md.

Usage:
    python3 examples/inspect_datasets.py
    python3 examples/inspect_datasets.py --strict
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

# Allow `python3 examples/inspect_datasets.py` from the repo root or elsewhere.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from examples.lib.loaders import LABEL_NAMES, load_csv
from examples.lib.paths import (
    DatasetSpec,
    iter_dataset_specs,
    spec_by_key,
    subject_sentence_csv,
    subject_word_csv,
)


def _row_count(spec: DatasetSpec) -> tuple[int, list[str]]:
    df = load_csv(spec.key)
    return len(df), list(df.columns)


def _label_summary(spec: DatasetSpec) -> str:
    df = load_csv(spec.key)
    if "sentiment_label" not in df.columns:
        return ""
    counts = Counter(int(v) for v in df["sentiment_label"])
    parts = [
        f"{LABEL_NAMES.get(k, k)}={counts.get(k, 0)}"
        for k in (0, 1, 2)
        if k in counts or True
    ]
    return "  labels: " + ", ".join(parts)


def inspect_spec(spec: DatasetSpec) -> list[str]:
    problems: list[str] = []
    if not spec.exists():
        return [f"MISSING {spec.relative}"]

    n, cols = _row_count(spec)
    status = "ok" if n == spec.expected_rows else "ROW MISMATCH"
    if n != spec.expected_rows:
        problems.append(
            f"{spec.key}: expected {spec.expected_rows} rows, found {n}"
        )
    print(f"[{status:12}] {spec.key:24} {n:7d} rows  {spec.relative}")
    if spec.expected_cols is not None and tuple(cols) != spec.expected_cols:
        problems.append(
            f"{spec.key}: columns {cols} != {list(spec.expected_cols)}"
        )
        print(f"               columns: {cols}")
    extra = _label_summary(spec)
    if extra:
        print(f"               {extra.strip()}")
    if spec.notes:
        print(f"               note: {spec.notes}")
    if spec.gaze_kind != "none":
        print(f"               gaze: {spec.gaze_kind} / {spec.level}")
    return problems


def inspect_subjects() -> list[str]:
    problems: list[str] = []
    print("\nSubject files (sentence-level, expect 400 rows each)")
    for subject in range(1, 13):
        path = subject_sentence_csv(subject)
        if not path.is_file():
            problems.append(f"missing {path}")
            print(f"[MISSING     ] subject {subject:2d} {path}")
            continue
        import pandas as pd

        n = len(pd.read_csv(path))
        status = "ok" if n == 400 else "ROW MISMATCH"
        if n != 400:
            problems.append(f"{path.name}: expected 400 rows, found {n}")
        print(f"[{status:12}] subject {subject:2d} {n:7d} rows  {path.relative_to(path.parents[2])}")

    print("\nSubject files (word-level, expect 7129 rows each)")
    for subject in range(1, 13):
        path = subject_word_csv(subject)
        if not path.is_file():
            problems.append(f"missing {path}")
            continue
        import pandas as pd

        n = len(pd.read_csv(path))
        status = "ok" if n == 7129 else "ROW MISMATCH"
        if n != 7129:
            problems.append(f"{path.name}: expected 7129 rows, found {n}")
        print(f"[{status:12}] subject {subject:2d} {n:7d} rows  {path.name}")
    return problems


def inspect_joins() -> list[str]:
    problems: list[str] = []
    print("\nJoins")
    text = load_csv("zuco_text")
    std = load_csv("zuco_standard")
    avg = load_csv("zuco_sentence_average")
    if list(text["sentence_id"].astype(int)) != list(std["sentence_id"].astype(int)):
        problems.append("zuco_text sentence_id != zuco_standard sentence_id")
    else:
        print("[ok          ] zuco_text ⋈ zuco_standard on sentence_id (400)")
    if list(avg["id"].astype(int)) != list(std["sentence_id"].astype(int)):
        problems.append("average_data id != zuco_standard sentence_id")
    else:
        print("[ok          ] average_data ⋈ zuco_standard on id/sentence_id (400)")

    train = load_csv("sst_train")
    valid = load_csv("sst_valid")
    test = load_csv("sst_test")
    combined = load_csv("sst_combined")
    parts = (
        set(train["sentence_id"].astype(int))
        | set(valid["sentence_id"].astype(int))
        | set(test["sentence_id"].astype(int))
    )
    all_ids = set(combined["sentence_id"].astype(int))
    if parts != all_ids:
        problems.append(
            f"full-SST splits are not a partition: union={len(parts)} combined={len(all_ids)}"
        )
    else:
        print("[ok          ] full-SST train/valid/test partition combined_full_sst_et.csv")
    overlap_tv = set(train["sentence_id"]) & set(valid["sentence_id"])
    overlap_tt = set(train["sentence_id"]) & set(test["sentence_id"])
    overlap_vt = set(valid["sentence_id"]) & set(test["sentence_id"])
    if overlap_tv or overlap_tt or overlap_vt:
        problems.append("full-SST splits overlap")
    else:
        print("[ok          ] full-SST splits have no shared sentence_id")

    z_train = set(load_csv("zuco_train")["sentence_id"].astype(int))
    z_valid = set(load_csv("zuco_valid")["sentence_id"].astype(int))
    z_test = set(load_csv("zuco_test")["sentence_id"].astype(int))
    z_all = set(std["sentence_id"].astype(int))
    if z_train | z_valid | z_test != z_all:
        problems.append("ZuCo 80/10/10 files are not a partition of the 400")
    else:
        print("[ok          ] ZuCo train/valid/test partition the 400-row table")
    return problems


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--strict",
        action="store_true",
        help="exit 1 if any row count, column, or join check fails",
    )
    args = parser.parse_args(argv)

    problems: list[str] = []
    print("Canonical tables\n")
    for spec in iter_dataset_specs():
        problems.extend(inspect_spec(spec))
    problems.extend(inspect_subjects())
    problems.extend(inspect_joins())

    print()
    if problems:
        print(f"{len(problems)} problem(s):")
        for item in problems:
            print(f"  - {item}")
        return 1 if args.strict else 0

    print("All inventory checks passed.")
    # Touch spec_by_key so a typo in docs is easy to catch in tests.
    spec_by_key("zuco_standard")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
