#!/usr/bin/env python3
"""Fail loudly if the committed splits or ZuCo subject files look wrong."""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

_EXAMPLES = Path(__file__).resolve().parent
if str(_EXAMPLES) not in sys.path:
    sys.path.insert(0, str(_EXAMPLES))

import pandas as pd

from paths import (
    FULL_SST_COMBINED,
    FULL_SST_TEST,
    FULL_SST_TRAIN,
    FULL_SST_VALID,
    REPO_ROOT,
    ZUCO_MINMAX,
    ZUCO_STANDARD,
    ZUCO_TEST,
    ZUCO_TEXT,
    ZUCO_TRAIN,
    ZUCO_VALID,
    subject_sentence_csv,
    subject_word_csv,
)


@dataclass
class Report:
    errors: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def check(self, cond: bool, message: str) -> None:
        if cond:
            self.notes.append("ok  " + message)
        else:
            self.errors.append("ERR " + message)

    def print(self) -> None:
        for line in self.notes:
            print(line)
        if self.errors:
            print()
            for line in self.errors:
                print(line)


def _rel(path) -> str:
    return str(path.relative_to(REPO_ROOT))


def assert_unique(report: Report, df: pd.DataFrame, col: str, name: str) -> None:
    n = len(df)
    nuniq = df[col].nunique(dropna=False)
    report.check(n == nuniq, f"{name}: {col} unique ({nuniq}/{n})")


def assert_partition(
    report: Report,
    combined: pd.DataFrame,
    parts: list[pd.DataFrame],
    key: str,
    name: str,
) -> None:
    part_ids = [set(p[key].tolist()) for p in parts]
    for i, left in enumerate(part_ids):
        for j, right in enumerate(part_ids):
            if i < j:
                overlap = left & right
                report.check(
                    not overlap,
                    f"{name}: split {i} ∩ split {j} is empty (overlap={len(overlap)})",
                )
    union = set.union(*part_ids) if part_ids else set()
    combined_ids = set(combined[key].tolist())
    report.check(
        union == combined_ids,
        f"{name}: union of splits equals combined ({len(union)} vs {len(combined_ids)})",
    )
    report.check(
        sum(len(p) for p in parts) == len(combined),
        f"{name}: row counts add up "
        f"({'+'.join(str(len(p)) for p in parts)}={sum(len(p) for p in parts)} "
        f"vs combined {len(combined)})",
    )


def assert_no_nan(report: Report, df: pd.DataFrame, name: str) -> None:
    n = int(df.isna().sum().sum())
    report.check(n == 0, f"{name}: no NaNs (found {n})")


def assert_labels(report: Report, df: pd.DataFrame, name: str) -> None:
    labels = set(int(v) for v in df["sentiment_label"].dropna())
    report.check(
        labels <= {0, 1, 2} and labels,
        f"{name}: sentiment_label ⊆ {{0,1,2}} (got {sorted(labels)})",
    )


def check_full_sst(report: Report) -> None:
    combined = pd.read_csv(FULL_SST_COMBINED)
    train = pd.read_csv(FULL_SST_TRAIN)
    valid = pd.read_csv(FULL_SST_VALID)
    test = pd.read_csv(FULL_SST_TEST)
    for df, name in (
        (combined, _rel(FULL_SST_COMBINED)),
        (train, _rel(FULL_SST_TRAIN)),
        (valid, _rel(FULL_SST_VALID)),
        (test, _rel(FULL_SST_TEST)),
    ):
        assert_unique(report, df, "sentence_id", name)
        assert_no_nan(report, df, name)
        assert_labels(report, df, name)
    assert_partition(
        report,
        combined,
        [train, valid, test],
        "sentence_id",
        "full SST 80/10/10",
    )
    expected_cols = {
        "sentence_id",
        "sentence",
        "sentiment_label",
        "nFix",
        "GD",
        "TRT",
        "FFD",
        "GPT",
    }
    report.check(
        set(combined.columns) == expected_cols,
        f"full SST combined columns == {sorted(expected_cols)}",
    )


def check_zuco_joins(report: Report) -> None:
    text = pd.read_csv(ZUCO_TEXT)
    standard = pd.read_csv(ZUCO_STANDARD)
    minmax = pd.read_csv(ZUCO_MINMAX)
    train = pd.read_csv(ZUCO_TRAIN)
    valid = pd.read_csv(ZUCO_VALID)
    test = pd.read_csv(ZUCO_TEST)

    for df, name in (
        (text, _rel(ZUCO_TEXT)),
        (standard, _rel(ZUCO_STANDARD)),
        (minmax, _rel(ZUCO_MINMAX)),
        (train, _rel(ZUCO_TRAIN)),
        (valid, _rel(ZUCO_VALID)),
        (test, _rel(ZUCO_TEST)),
    ):
        assert_unique(report, df, "sentence_id", name)
        assert_no_nan(report, df, name)
        assert_labels(report, df, name)

    report.check(len(text) == 400, f"{_rel(ZUCO_TEXT)} has 400 rows (got {len(text)})")
    report.check(
        len(standard) == 400,
        f"{_rel(ZUCO_STANDARD)} has 400 rows (got {len(standard)})",
    )
    report.check(
        list(text["sentence_id"]) == list(standard["sentence_id"]),
        "ssts_ZuCo sentence_id order matches combined_sst_et_standard",
    )
    report.check(
        list(text["sentence"]) == list(standard["sentence"]),
        "ssts_ZuCo sentences match combined_sst_et_standard",
    )
    report.check(
        list(standard["sentence_id"]) == list(minmax["sentence_id"]),
        "standard and min-max joins share sentence_id order",
    )
    assert_partition(
        report,
        standard,
        [train, valid, test],
        "sentence_id",
        "ZuCo 80/10/10",
    )


def check_subjects(report: Report) -> None:
    for i in range(1, 13):
        sent = pd.read_csv(subject_sentence_csv(i))
        word = pd.read_csv(subject_word_csv(i))
        expected_sent = 299 if i == 3 else 400
        expected_word = 5293 if i == 3 else 7129
        report.check(
            len(sent) == expected_sent,
            f"{i}_SR.csv sentence rows == {expected_sent} (got {len(sent)})",
        )
        report.check(
            len(word) == expected_word,
            f"word/{i}_SR.csv rows == {expected_word} (got {len(word)})",
        )
        report.check(
            "nFixations" in sent.columns and "GPT" in sent.columns,
            f"{i}_SR.csv has nFixations and GPT",
        )


def main() -> int:
    report = Report()
    check_full_sst(report)
    check_zuco_joins(report)
    check_subjects(report)
    report.print()
    if report.errors:
        print(f"\n{len(report.errors)} integrity check(s) failed.")
        return 1
    print(f"\nAll {len(report.notes)} integrity checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
