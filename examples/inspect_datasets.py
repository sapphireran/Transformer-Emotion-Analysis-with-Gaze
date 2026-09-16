#!/usr/bin/env python3
"""Print schemas, row counts, and label mixes for every documented table."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from csvutil import label_counts, read_dicts, read_rows
from paths import (
    PRED_SMALL,
    PRED_V2,
    PROVO,
    REPO_ROOT,
    SST_COMBINED,
    SST_HEADERLESS,
    SST_TEST,
    SST_TRAIN,
    SST_VALID,
    SST_WORD_ZEROS,
    ZUCO_AVERAGE,
    ZUCO_AVERAGE_MM,
    ZUCO_AVERAGE_STD,
    ZUCO_COMBINED_MM,
    ZUCO_COMBINED_STD,
    ZUCO_ET_DIR,
    ZUCO_LABELS,
    ZUCO_SUBJECT_3,
    ZUCO_TEST,
    ZUCO_TRAIN,
    ZUCO_VALID,
    ZUCO_WORD_AVG,
)

# Counts frozen in docs/datasets.md. The unit tests fail if a CSV is rewritten
# without updating the docs.
EXPECTED_ROWS = {
    SST_COMBINED: 11853,
    SST_TRAIN: 9482,
    SST_VALID: 1185,
    SST_TEST: 1186,
    SST_WORD_ZEROS: 191971,
    ZUCO_COMBINED_STD: 400,
    ZUCO_COMBINED_MM: 400,
    ZUCO_LABELS: 400,
    ZUCO_TRAIN: 320,
    ZUCO_VALID: 40,
    ZUCO_TEST: 40,
    ZUCO_AVERAGE: 400,
    ZUCO_AVERAGE_STD: 400,
    ZUCO_AVERAGE_MM: 400,
    ZUCO_WORD_AVG: 7129,
    PRED_SMALL: 1751,
    PRED_V2: 191971,
    PROVO: 2659,
}

EXPECTED_SST_LABELS = {"0": 4649, "1": 2241, "2": 4963}
EXPECTED_ZUCO_LABELS = {"0": 123, "1": 137, "2": 140}


def _rel(path: Path) -> str:
    return str(path.relative_to(REPO_ROOT))


def describe(path: Path, *, label_column: str | None = "sentiment_label") -> dict:
    columns, rows = read_dicts(path)
    info = {
        "path": _rel(path),
        "rows": len(rows),
        "columns": columns,
    }
    if label_column and label_column in columns:
        info["labels"] = label_counts(rows, label_column)
    return info


def describe_headerless(path: Path) -> dict:
    header, rows = read_rows(path)
    # First physical row is data, not a header. Include it in the count.
    return {
        "path": _rel(path),
        "rows": len(rows) + (1 if header else 0),
        "columns": ["sentence", "polarity_string"],
        "headerless": True,
        "first_polarity": header[1] if header and len(header) > 1 else None,
    }


def subject_row_counts() -> dict[str, int]:
    counts = {}
    for index in range(1, 13):
        path = ZUCO_ET_DIR / f"{index}_SR.csv"
        _, rows = read_dicts(path)
        counts[path.name] = len(rows)
    return counts


def inspect() -> list[dict]:
    reports = [
        describe(SST_COMBINED),
        describe(SST_TRAIN),
        describe(SST_VALID),
        describe(SST_TEST),
        describe(SST_WORD_ZEROS, label_column=None),
        describe_headerless(SST_HEADERLESS),
        describe(ZUCO_COMBINED_STD),
        describe(ZUCO_COMBINED_MM),
        describe(ZUCO_LABELS),
        describe(ZUCO_TRAIN),
        describe(ZUCO_VALID),
        describe(ZUCO_TEST),
        describe(ZUCO_AVERAGE, label_column=None),
        describe(ZUCO_AVERAGE_STD, label_column=None),
        describe(ZUCO_AVERAGE_MM, label_column=None),
        describe(ZUCO_WORD_AVG, label_column=None),
        describe(PRED_SMALL, label_column=None),
        describe(PRED_V2, label_column=None),
        describe(PROVO, label_column=None),
    ]
    return reports


def format_report(info: dict) -> str:
    cols = ", ".join(info["columns"][:10])
    extra = ""
    if len(info["columns"]) > 10:
        extra = f", … (+{len(info['columns']) - 10} more)"
    lines = [f"{info['path']}: {info['rows']} rows | {cols}{extra}"]
    if "labels" in info:
        pretty = ", ".join(f"{k}={v}" for k, v in info["labels"].items())
        lines.append(f"  labels: {pretty}")
    if info.get("headerless"):
        lines.append(f"  headerless file; first polarity token={info['first_polarity']}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quiet", action="store_true", help="only print a one-line OK")
    args = parser.parse_args(argv)

    reports = inspect()
    subjects = subject_row_counts()
    _, subject3 = read_dicts(ZUCO_SUBJECT_3)

    if not args.quiet:
        print("Dataset inventory")
        print("=================")
        for info in reports:
            print(format_report(info))
        print()
        print("Per-subject sentence tables")
        for name, count in subjects.items():
            print(f"  {name}: {count}")
        print(f"  subject 3 missing tail ids: {400 - len(subject3)}")

    problems = []
    for path, expected in EXPECTED_ROWS.items():
        _, rows = read_dicts(path)
        if len(rows) != expected:
            problems.append(f"{_rel(path)}: expected {expected} rows, got {len(rows)}")

    combined_labels = label_counts(read_dicts(SST_COMBINED)[1])
    if combined_labels != EXPECTED_SST_LABELS:
        problems.append(f"SST combined labels {combined_labels} != {EXPECTED_SST_LABELS}")

    zuco_labels = label_counts(read_dicts(ZUCO_COMBINED_STD)[1])
    if zuco_labels != EXPECTED_ZUCO_LABELS:
        problems.append(f"ZuCo labels {zuco_labels} != {EXPECTED_ZUCO_LABELS}")

    if subjects["3_SR.csv"] != 299:
        problems.append(f"subject 3 should have 299 rows, got {subjects['3_SR.csv']}")
    if any(count != 400 for name, count in subjects.items() if name != "3_SR.csv"):
        problems.append(f"unexpected subject counts: {subjects}")

    if problems:
        print("INVENTORY MISMATCH:")
        for item in problems:
            print(" -", item)
        return 1

    print("inventory OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
