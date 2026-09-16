"""Load and validate committed CSVs."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .paths import data_path
from .schema import (
    FULL_SST_GAZE_COLS,
    PREDICTED_WORD_GAZE_COLS,
    PROVO_GAZE_COLS,
    ZUCO_JOIN_GAZE_COLS,
    ZUCO_SENTENCE_GAZE_COLS,
    WORD_GAZE_COLS,
)


class SchemaError(ValueError):
    pass


def _require_columns(df: pd.DataFrame, columns: tuple[str, ...], origin: str) -> None:
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise SchemaError(f"{origin} missing columns {missing}; have {list(df.columns)}")


def _read_csv(path: Path, origin: str | None = None) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Expected CSV at {path}")
    df = pd.read_csv(path)
    if df.empty:
        raise SchemaError(f"{origin or path} is empty")
    return df


def load_zuco_text(path: Path | None = None) -> pd.DataFrame:
    path = path or data_path("ZuCo_SST_data", "ssts_ZuCo.csv")
    df = _read_csv(path, str(path))
    _require_columns(df, ("sentence_id", "sentence", "sentiment_label"), str(path))
    return df


def load_zuco_combined(scaling: str = "standard", path: Path | None = None) -> pd.DataFrame:
    if path is None:
        name = {
            "standard": "combined_sst_et_standard.csv",
            "min_max": "combined_sst_et_min_max.csv",
        }[scaling]
        path = data_path("ZuCo_SST_data", name)
    df = _read_csv(path, str(path))
    _require_columns(
        df,
        ("sentence_id", "sentence", "sentiment_label") + ZUCO_JOIN_GAZE_COLS,
        str(path),
    )
    return df


def load_zuco_split(split: str) -> pd.DataFrame:
    if split not in {"train", "valid", "test"}:
        raise ValueError(split)
    return load_zuco_combined(path=data_path("ZuCo_SST_data", f"{split}.csv"))


def load_full_sst_combined(path: Path | None = None) -> pd.DataFrame:
    path = path or data_path("SST_data", "combined_full_sst_et.csv")
    df = _read_csv(path, str(path))
    _require_columns(
        df,
        ("sentence_id", "sentence", "sentiment_label") + FULL_SST_GAZE_COLS,
        str(path),
    )
    return df


def load_full_sst_split(split: str) -> pd.DataFrame:
    names = {
        "train": "train_full_sst.csv",
        "valid": "valid_full_sst.csv",
        "test": "test_full_sst.csv",
    }
    return load_full_sst_combined(path=data_path("SST_data", names[split]))


def load_raw_sst_sentences(path: Path | None = None) -> pd.DataFrame:
    path = path or data_path("SST_data", "stts_all_sentence_level.csv")
    df = pd.read_csv(path, header=None, names=["sentence", "label_string"])
    return df


def load_subject_sentence_gaze(subject: int) -> pd.DataFrame:
    if subject < 1 or subject > 12:
        raise ValueError("subject must be 1–12 (filename index)")
    path = data_path("ZuCo_et_csv_data", f"{subject}_SR.csv")
    df = _read_csv(path, str(path))
    _require_columns(df, ("id",) + ZUCO_SENTENCE_GAZE_COLS, str(path))
    return df


def load_sentence_average(kind: str = "raw") -> pd.DataFrame:
    names = {
        "raw": "average_data.csv",
        "standard": "standard_scaled_average_data.csv",
        "min_max": "min_max_scaled_average_data.csv",
    }
    path = data_path("ZuCo_et_csv_data", names[kind])
    df = _read_csv(path, str(path))
    _require_columns(df, ("id",) + ZUCO_SENTENCE_GAZE_COLS, str(path))
    return df


def load_word_averages(version: str = "v2") -> pd.DataFrame:
    name = "word_averages_v2.csv" if version == "v2" else "word_averages.csv"
    path = data_path("ZuCo_et_csv_data", "word", name)
    df = _read_csv(path, str(path))
    _require_columns(
        df,
        ("id", "Sent_ID", "Word_ID", "Word", "WordLen") + WORD_GAZE_COLS,
        str(path),
    )
    return df


def load_subject_word_gaze(subject: int) -> pd.DataFrame:
    path = data_path("ZuCo_et_csv_data", "word", f"{subject}_SR.csv")
    df = _read_csv(path, str(path))
    _require_columns(
        df,
        ("id", "Sent_ID", "Word_ID", "Word", "WordLen") + WORD_GAZE_COLS,
        str(path),
    )
    return df


def load_predicted_word_gaze(which: str = "v2") -> pd.DataFrame:
    name = {
        "test": "prediction_test.csv",
        "v2": "prediction_test_v2.csv",
    }[which]
    path = data_path("gaze_prediction", "data", name)
    df = _read_csv(path, str(path))
    _require_columns(
        df,
        ("sentence_id", "word_id", "word") + PREDICTED_WORD_GAZE_COLS,
        str(path),
    )
    return df


def load_provo() -> pd.DataFrame:
    path = data_path("gaze_prediction", "data", "provo.csv")
    df = _read_csv(path, str(path))
    _require_columns(
        df,
        ("sentence_id", "word_id", "word") + PROVO_GAZE_COLS,
        str(path),
    )
    return df
