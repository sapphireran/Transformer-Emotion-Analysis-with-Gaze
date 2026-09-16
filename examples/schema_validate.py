#!/usr/bin/env python3
"""Validate headers, row counts, labels, and finite gaze cells.

Exit codes:
    0  all documented tables look consistent
    1  one or more soft mismatches (row count or unexpected extra issues)
    2  missing file or broken required column

    python3 examples/schema_validate.py
"""

from __future__ import annotations

import os
import sys
from typing import List

from common import (
    DATASETS,
    SUBJECT_SENTENCE_FILES,
    abs_path,
    as_int,
    is_finite_number,
    read_rows,
)

NUMERIC_KINDS = {
    "zuco-joined": (
        "sentiment_label",
        "omissionRate",
        "nFixations",
        "meanPupilSize",
        "GD",
        "TRT",
        "FFD",
        "SFD",
        "GPT",
    ),
    "sst-joined": ("sentiment_label", "nFix", "FFD", "GPT", "TRT", "GD"),
    "zuco-sentence-gaze": (
        "SentLen",
        "omissionRate",
        "nFixations",
        "meanPupilSize",
        "GD",
        "TRT",
        "FFD",
        "SFD",
        "GPT",
    ),
    "zuco-text": ("sentiment_label",),
}


def check_dataset(spec: dict) -> List[str]:
    errors: List[str] = []
    rel = str(spec["path"])
    path = abs_path(rel)
    key = spec["key"]
    if not os.path.isfile(path):
        return [f"{key}: missing file {rel}"]

    cols, rows = read_rows(rel)
    required = tuple(spec["required"])
    missing_cols = [c for c in required if c not in cols]
    if missing_cols:
        errors.append(f"{key}: missing columns {missing_cols}")
        return errors

    expected = int(spec["expected_rows"])
    if len(rows) != expected:
        errors.append(f"{key}: {len(rows)} rows, documented {expected}")

    kind = spec["kind"]
    numeric = NUMERIC_KINDS.get(kind, ())
    bad_numeric = 0
    bad_label = 0
    empty_sentence = 0
    for row in rows:
        if "sentence" in required and not (row.get("sentence") or "").strip():
            empty_sentence += 1
        if "sentiment_label" in required:
            try:
                lab = as_int(row["sentiment_label"])
            except ValueError:
                bad_label += 1
                continue
            if lab not in (0, 1, 2):
                bad_label += 1
        for col in numeric:
            if col == "sentiment_label":
                continue
            if col not in row:
                continue
            if not is_finite_number(row[col]):
                bad_numeric += 1
                break
    if empty_sentence:
        errors.append(f"{key}: {empty_sentence} empty sentences")
    if bad_label:
        errors.append(f"{key}: {bad_label} labels outside {{0,1,2}}")
    if bad_numeric:
        errors.append(f"{key}: {bad_numeric} rows with non-finite gaze")
    return errors


def check_subject_ids() -> List[str]:
    """Subject 1,2,4–12 should have ids 0..399. Subject 3 is allowed to be short."""
    errors: List[str] = []
    for rel in SUBJECT_SENTENCE_FILES:
        if not os.path.isfile(abs_path(rel)):
            errors.append(f"missing {rel}")
            continue
        _cols, rows = read_rows(rel)
        ids = [as_int(r["id"]) for r in rows]
        if "3_SR.csv" in rel:
            if len(ids) >= 400:
                errors.append(f"{rel}: expected the short subject-3 table")
            continue
        if ids != list(range(400)):
            errors.append(f"{rel}: id column is not 0..399 (n={len(ids)})")
    return errors


def main() -> int:
    print("Schema validation")
    problems: List[str] = []
    for spec in DATASETS:
        # Skip the 191k-row files' full numeric scan? We still scan them —
        # takes a couple of seconds and is the point of the script.
        problems.extend(check_dataset(spec))
    problems.extend(check_subject_ids())

    if not problems:
        print(f"OK  {len(DATASETS)} documented tables + 12 subject files")
        print("    headers, row counts, labels ∈ {0,1,2}, gaze finite")
        return 0

    print(f"FOUND {len(problems)} issue(s):")
    for item in problems:
        print(f"  - {item}")
    hard = any(
        ("missing file" in p) or ("missing columns" in p) or ("labels outside" in p)
        for p in problems
    )
    return 2 if hard else 1


if __name__ == "__main__":
    sys.exit(main())
