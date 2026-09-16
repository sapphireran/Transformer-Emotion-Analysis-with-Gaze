"""CSV loaders that pin dtypes and document header quirks."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .paths import (
    SST_COMBINED,
    SST_ET_PLACEHOLDER,
    SST_STTS,
    SST_TEST,
    SST_TRAIN,
    SST_VALID,
    ZUCO_MINMAX,
    ZUCO_SENTENCES,
    ZUCO_STANDARD,
    ZUCO_TEST,
    ZUCO_TRAIN,
    ZUCO_VALID,
    subject_sentence_csv,
    subject_word_csv,
)


def read_csv(path: Path, **kwargs) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(path)
    return pd.read_csv(path, **kwargs)


def load_sst_train() -> pd.DataFrame:
    return read_csv(SST_TRAIN)


def load_sst_valid() -> pd.DataFrame:
    return read_csv(SST_VALID)


def load_sst_test() -> pd.DataFrame:
    return read_csv(SST_TEST)


def load_sst_combined() -> pd.DataFrame:
    return read_csv(SST_COMBINED)


def load_sst_stts() -> pd.DataFrame:
    """Headerless two-column dump: sentence, POSITIVE|NEGATIVE|NEUTRAL."""
    return read_csv(SST_STTS, header=None, names=["sentence", "sentiment_name"])


def load_sst_et_placeholder() -> pd.DataFrame:
    """Word-level SST tokens with gaze columns zeroed (predictor input stub)."""
    return read_csv(SST_ET_PLACEHOLDER)


def load_zuco_standard() -> pd.DataFrame:
    return read_csv(ZUCO_STANDARD)


def load_zuco_minmax() -> pd.DataFrame:
    return read_csv(ZUCO_MINMAX)


def load_zuco_sentences() -> pd.DataFrame:
    return read_csv(ZUCO_SENTENCES)


def load_zuco_split(name: str) -> pd.DataFrame:
    mapping = {"train": ZUCO_TRAIN, "valid": ZUCO_VALID, "test": ZUCO_TEST}
    if name not in mapping:
        raise ValueError(f"unknown split {name!r}")
    return read_csv(mapping[name])


def load_subject_sentence(subject_1indexed: int) -> pd.DataFrame:
    return read_csv(subject_sentence_csv(subject_1indexed))


def load_subject_word(subject_1indexed: int) -> pd.DataFrame:
    return read_csv(subject_word_csv(subject_1indexed))


def load_all_subject_sentence() -> dict[int, pd.DataFrame]:
    return {i: load_subject_sentence(i) for i in range(1, 13)}


def load_all_subject_word() -> dict[int, pd.DataFrame]:
    return {i: load_subject_word(i) for i in range(1, 13)}
