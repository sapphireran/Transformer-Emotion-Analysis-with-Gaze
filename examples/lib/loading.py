"""Load the checked-in CSVs with consistent dtypes and names."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .paths import resolve_root

LABEL_NAMES = {0: "negative", 1: "neutral", 2: "positive"}

FULL_SST_GAZE = ("nFix", "FFD", "GPT", "TRT", "GD")
ZUCO_GAZE = ("nFixations", "FFD", "GPT", "TRT", "GD")
# 1-indexed filenames. Subject 3 is DataTransformer subject 2 (task 1),
# which drops original sentences 150-249 and 399, then reindexes 0..298.
SUBJECT_SENTENCE_ROWS = {subject: 400 for subject in range(1, 13)}
SUBJECT_SENTENCE_ROWS[3] = 299
SUBJECT_3_DROPPED_ORIGINAL_IDS = tuple(range(150, 250)) + (399,)
ZUCO_SENTENCE_NUMERIC = (
    "SentLen",
    "omissionRate",
    "nFixations",
    "meanPupilSize",
    "GD",
    "TRT",
    "FFD",
    "SFD",
    "GPT",
)


@dataclass(frozen=True)
class DatasetSpec:
    """One documented table in this personal checkout."""

    key: str
    relative_path: str
    expected_rows: int | None
    required_columns: tuple[str, ...]
    notes: str


def documented_datasets() -> tuple[DatasetSpec, ...]:
    return (
        DatasetSpec(
            "sst_combined",
            "SST_data/combined_full_sst_et.csv",
            11853,
            ("sentence_id", "sentence", "sentiment_label", *FULL_SST_GAZE),
            "Full SST + standardized sentence-level gaze.",
        ),
        DatasetSpec(
            "sst_train",
            "SST_data/train_full_sst.csv",
            9482,
            ("sentence_id", "sentence", "sentiment_label", *FULL_SST_GAZE),
            "80% holdout split (random_state=42).",
        ),
        DatasetSpec(
            "sst_valid",
            "SST_data/valid_full_sst.csv",
            1185,
            ("sentence_id", "sentence", "sentiment_label", *FULL_SST_GAZE),
            "10% holdout split.",
        ),
        DatasetSpec(
            "sst_test",
            "SST_data/test_full_sst.csv",
            1186,
            ("sentence_id", "sentence", "sentiment_label", *FULL_SST_GAZE),
            "10% holdout split.",
        ),
        DatasetSpec(
            "zuco_standard",
            "ZuCo_SST_data/combined_sst_et_standard.csv",
            400,
            ("sentence_id", "sentence", "sentiment_label", "nFixations", "FFD", "GPT", "TRT", "GD"),
            "400 ZuCo sentences, standardized measured gaze.",
        ),
        DatasetSpec(
            "zuco_minmax",
            "ZuCo_SST_data/combined_sst_et_min_max.csv",
            400,
            ("sentence_id", "sentence", "sentiment_label", "nFixations"),
            "Same 400 sentences, min-max gaze.",
        ),
        DatasetSpec(
            "zuco_text",
            "ZuCo_SST_data/ssts_ZuCo.csv",
            400,
            ("sentence_id", "sentence", "sentiment_label"),
            "ZuCo SST text + integer labels only.",
        ),
        DatasetSpec(
            "zuco_word_avg",
            "ZuCo_et_csv_data/word/word_averages_v2.csv",
            7129,
            ("id", "Sent_ID", "Word_ID", "Word", "nFixations", "WordLen"),
            "12-subject mean word-level gaze.",
        ),
        DatasetSpec(
            "pred_v2",
            "gaze_prediction/data/prediction_test_v2.csv",
            191971,
            ("sentence_id", "word_id", "word", "nFix", "FFD", "GPT", "TRT", "GD"),
            "Predicted word-level gaze for full SST.",
        ),
        DatasetSpec(
            "pred_extract",
            "gaze_prediction/data/prediction_test.csv",
            1751,
            ("sentence_id", "word_id", "word", "nFix", "FFD", "GPT", "TRT", "GD"),
            "Short predicted-gaze extract (sentence_id starts at 300).",
        ),
        DatasetSpec(
            "provo",
            "gaze_prediction/data/provo.csv",
            2659,
            ("sentence_id", "word_id", "word", "nFix", "FFD", "GPT", "TRT", "fixProp"),
            "PROVO reference extract; has fixProp instead of GD.",
        ),
    )


def load_dataset(spec: DatasetSpec, root: str | Path | None = None) -> pd.DataFrame:
    path = resolve_root(root) / spec.relative_path
    if not path.is_file():
        raise FileNotFoundError(f"{spec.key}: missing {path}")
    df = pd.read_csv(path)
    missing = [c for c in spec.required_columns if c not in df.columns]
    if missing:
        raise ValueError(f"{spec.key}: missing columns {missing} in {path}")
    return df


def load_sst_raw(root: str | Path | None = None) -> pd.DataFrame:
    """Load the headerless SST dump as ``sentence`` + ``sentiment`` strings."""
    path = resolve_root(root) / "SST_data/stts_all_sentence_level.csv"
    df = pd.read_csv(path, header=None, names=["sentence", "sentiment"])
    return df


def load_full_sst_splits(
    root: str | Path | None = None,
) -> dict[str, pd.DataFrame]:
    by_key = {spec.key: spec for spec in documented_datasets()}
    return {
        name: load_dataset(by_key[f"sst_{name}"], root=root)
        for name in ("train", "valid", "test")
    }


def load_zuco_combined(
    root: str | Path | None = None, scaling: str = "standard"
) -> pd.DataFrame:
    key = "zuco_standard" if scaling == "standard" else "zuco_minmax"
    spec = next(s for s in documented_datasets() if s.key == key)
    return load_dataset(spec, root=root)


def load_zuco_word_average(root: str | Path | None = None) -> pd.DataFrame:
    spec = next(s for s in documented_datasets() if s.key == "zuco_word_avg")
    return load_dataset(spec, root=root)


def load_subject_sentence_tables(
    root: str | Path | None = None,
) -> dict[int, pd.DataFrame]:
    """Load the 12 raw sentence-level subject files (1-indexed keys).

    Subject 3 has 299 rows whose ``id`` is a *new* index, not the original
    ZuCo sentence id. Use :func:`remap_subject3_original_ids` before
    comparing that table to anyone else.
    """
    base = resolve_root(root) / "ZuCo_et_csv_data"
    tables: dict[int, pd.DataFrame] = {}
    for subject in range(1, 13):
        path = base / f"{subject}_SR.csv"
        if not path.is_file():
            raise FileNotFoundError(f"missing subject sentence table: {path}")
        df = pd.read_csv(path)
        for col in ("id",) + ZUCO_SENTENCE_NUMERIC:
            if col not in df.columns:
                raise ValueError(f"{path} missing column {col}")
        expected = SUBJECT_SENTENCE_ROWS[subject]
        if len(df) != expected:
            raise ValueError(f"{path}: expected {expected} rows, found {len(df)}")
        tables[subject] = df
    return tables


def remap_subject3_original_ids(df: pd.DataFrame) -> pd.DataFrame:
    """Map subject 3's reindexed ``id`` back to the original sentence id.

    ``DataTransformer`` skipped original sentences 150-249 and 399, then
    wrote a fresh 0..298 index. Rows 0-149 are still original ids; rows
    150-298 are original 250-398.
    """
    if len(df) != SUBJECT_SENTENCE_ROWS[3]:
        raise ValueError(
            f"subject 3 remap expects {SUBJECT_SENTENCE_ROWS[3]} rows, got {len(df)}"
        )
    original = list(range(150)) + list(range(250, 399))
    out = df.copy()
    out["reindexed_id"] = out["id"]
    out["id"] = original
    return out


def label_name_series(labels: pd.Series) -> pd.Series:
    return labels.map(LABEL_NAMES)
