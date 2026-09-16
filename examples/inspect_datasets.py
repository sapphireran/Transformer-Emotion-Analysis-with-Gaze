#!/usr/bin/env python3
"""Print schemas, label histograms, and null counts for committed tables.

Exits with status 1 if a required file is missing, a fusion column is absent,
or a split is missing a sentiment class. Safe to run as a cheap regression
check after regenerating CSVs.

Usage (from repo root):

    python3 examples/inspect_datasets.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

from paths import (
    LABEL_NAMES,
    PRED_WORD_V2,
    PROVO,
    SST_COMBINED,
    SST_FUSION_FEATURES,
    SST_RAW_TEXT,
    SST_TEST,
    SST_TRAIN,
    SST_VALID,
    ZUCO_AVERAGE_RAW,
    ZUCO_COMBINED_MINMAX,
    ZUCO_COMBINED_STANDARD,
    ZUCO_FUSION_FEATURES,
    ZUCO_SUBJECT_DIR,
    ZUCO_TEST,
    ZUCO_TEXT,
    ZUCO_TRAIN,
    ZUCO_VALID,
    ZUCO_WORD_AVERAGES,
)

REQUIRED_SENTIMENT = {0, 1, 2}


def _load_csv(path: Path, **kwargs) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"missing table: {path}")
    return pd.read_csv(path, **kwargs)


def _label_hist(series: pd.Series) -> dict[str, int]:
    counts = series.value_counts(dropna=False).to_dict()
    pretty = {}
    for key, value in sorted(counts.items(), key=lambda kv: str(kv[0])):
        name = LABEL_NAMES.get(key, str(key))
        pretty[f"{key}:{name}"] = int(value)
    return pretty


def _null_report(df: pd.DataFrame) -> dict[str, int]:
    nulls = df.isna().sum()
    return {col: int(n) for col, n in nulls.items() if n}


def describe_table(name: str, df: pd.DataFrame, extra: str | None = None) -> None:
    print(f"\n=== {name} ===")
    print(f"rows={len(df)} cols={len(df.columns)}")
    print("columns:", ", ".join(map(str, df.columns.tolist())))
    if extra:
        print(extra)
    nulls = _null_report(df)
    if nulls:
        print("nulls:", nulls)
    else:
        print("nulls: none")


def check_fusion(name: str, df: pd.DataFrame, features: list[str], errors: list[str]) -> None:
    missing = [col for col in features if col not in df.columns]
    if missing:
        errors.append(f"{name} missing fusion columns: {missing}")
        return
    numeric = df[features].apply(pd.to_numeric, errors="coerce")
    if numeric.isna().any().any():
        bad = numeric.isna().sum().to_dict()
        errors.append(f"{name} non-numeric/NaN in fusion columns: {bad}")


def check_labels(name: str, series: pd.Series, errors: list[str], require_all: bool) -> None:
    values = set(pd.to_numeric(series, errors="coerce").dropna().astype(int))
    if require_all and not REQUIRED_SENTIMENT.issubset(values):
        errors.append(f"{name} missing sentiment classes: {REQUIRED_SENTIMENT - values}")
    unexpected = values - REQUIRED_SENTIMENT
    if unexpected:
        errors.append(f"{name} unexpected label values: {unexpected}")


def main() -> int:
    errors: list[str] = []

    zuco = _load_csv(ZUCO_COMBINED_STANDARD)
    describe_table("ZuCo combined (standard)", zuco)
    print("labels:", _label_hist(zuco["sentiment_label"]))
    print("sentence_id range:", int(zuco["sentence_id"].min()), "→", int(zuco["sentence_id"].max()))
    print("mean sentence chars:", round(zuco["sentence"].astype(str).str.len().mean(), 1))
    check_fusion("ZuCo combined", zuco, ZUCO_FUSION_FEATURES, errors)
    check_labels("ZuCo combined", zuco["sentiment_label"], errors, require_all=True)

    zuco_mm = _load_csv(ZUCO_COMBINED_MINMAX)
    describe_table("ZuCo combined (min-max)", zuco_mm)
    if len(zuco_mm) != len(zuco):
        errors.append("min-max and standard ZuCo tables have different row counts")

    text = _load_csv(ZUCO_TEXT)
    describe_table("ZuCo text-only", text)
    if set(text["sentence_id"]) != set(zuco["sentence_id"]):
        errors.append("ZuCo text sentence_id set does not match combined table")

    for split_name, path in (("train", ZUCO_TRAIN), ("valid", ZUCO_VALID), ("test", ZUCO_TEST)):
        split = _load_csv(path)
        describe_table(f"ZuCo {split_name} split", split)
        print("labels:", _label_hist(split["sentiment_label"]))
        check_labels(f"ZuCo {split_name}", split["sentiment_label"], errors, require_all=True)
        check_fusion(f"ZuCo {split_name}", split, ZUCO_FUSION_FEATURES, errors)

    split_ids = pd.concat(
        [
            _load_csv(ZUCO_TRAIN)["sentence_id"],
            _load_csv(ZUCO_VALID)["sentence_id"],
            _load_csv(ZUCO_TEST)["sentence_id"],
        ]
    )
    if split_ids.duplicated().any():
        errors.append("ZuCo train/valid/test sentence_id overlap")
    if set(split_ids) != set(zuco["sentence_id"]):
        errors.append("ZuCo splits do not cover the combined sentence_id set")

    average = _load_csv(ZUCO_AVERAGE_RAW)
    describe_table("ZuCo subject-average (raw ms)", average)
    subject_files = sorted(ZUCO_SUBJECT_DIR.glob("[0-9]*_SR.csv"))
    print(f"per-subject files: {len(subject_files)}")
    if len(subject_files) != 12:
        errors.append(f"expected 12 subject CSVs, found {len(subject_files)}")
    for path in subject_files:
        sub = _load_csv(path)
        if len(sub) != len(average):
            errors.append(f"{path.name} has {len(sub)} rows; average has {len(average)}")

    words = _load_csv(ZUCO_WORD_AVERAGES)
    describe_table("ZuCo word averages v2", words)
    print("unique Sent_ID:", words["Sent_ID"].nunique())
    print("empty Word filled as unknown:", int((words["Word"] == "unknown").sum()))

    sst_train = _load_csv(SST_TRAIN)
    sst_valid = _load_csv(SST_VALID)
    sst_test = _load_csv(SST_TEST)
    sst_combined = _load_csv(SST_COMBINED)
    describe_table("full SST combined", sst_combined)
    print("labels:", _label_hist(sst_combined["sentiment_label"]))
    for name, frame in (("train", sst_train), ("valid", sst_valid), ("test", sst_test)):
        describe_table(f"full SST {name}", frame)
        print("labels:", _label_hist(frame["sentiment_label"]))
        check_fusion(f"full SST {name}", frame, SST_FUSION_FEATURES, errors)
        check_labels(f"full SST {name}", frame["sentiment_label"], errors, require_all=True)

    split_n = len(sst_train) + len(sst_valid) + len(sst_test)
    if split_n != len(sst_combined):
        errors.append(
            f"full SST splits sum to {split_n} rows; combined has {len(sst_combined)}"
        )

    raw = _load_csv(SST_RAW_TEXT, header=None, names=["sentence", "label_name"])
    describe_table("full SST raw text (no header)", raw)
    print("string labels:", raw["label_name"].value_counts().to_dict())
    if len(raw) != len(sst_combined):
        errors.append("raw SST text row count does not match combined_full_sst_et.csv")

    pred = _load_csv(PRED_WORD_V2)
    describe_table("predicted word-level gaze v2", pred)
    print("sentences:", pred["sentence_id"].nunique(), "words/sentence mean:",
          round(pred.groupby("sentence_id").size().mean(), 2))
    for col in SST_FUSION_FEATURES:
        if col not in pred.columns:
            errors.append(f"prediction_test_v2 missing {col}")

    provo = _load_csv(PROVO)
    describe_table("PROVO word table", provo)
    if "GD" in provo.columns:
        errors.append("PROVO unexpectedly has GD; inventory says fixProp")
    if "fixProp" not in provo.columns:
        errors.append("PROVO missing fixProp")

    print("\n=== summary ===")
    if errors:
        print(f"{len(errors)} problem(s):")
        for item in errors:
            print(" -", item)
        return 1
    print("all inventory checks passed")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except FileNotFoundError as exc:
        print(exc, file=sys.stderr)
        sys.exit(1)
