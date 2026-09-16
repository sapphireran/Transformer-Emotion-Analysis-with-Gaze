"""CSV loaders that match the quirks documented in docs/datasets.md."""

from __future__ import annotations

from typing import Iterable

import pandas as pd

from .paths import REPO_ROOT, spec_by_key, subject_word_csv

LABEL_NAMES = {0: "negative", 1: "neutral", 2: "positive"}

# Five-feature contract used by both historical trainers.
FUSION_GAZE_COLUMNS = ("nFixations", "FFD", "GPT", "TRT", "GD")
FUSION_GAZE_COLUMNS_SST = ("nFix", "FFD", "GPT", "TRT", "GD")

ZUCO_GAZE_COLUMNS = (
    "omissionRate",
    "nFixations",
    "meanPupilSize",
    "GD",
    "TRT",
    "FFD",
    "SFD",
    "GPT",
)

STRING_POLARITY = {"NEGATIVE": 0, "NEUTRAL": 1, "POSITIVE": 2}


def load_csv(key: str) -> pd.DataFrame:
    """Load a named inventory table, including the headerless SST dump."""
    spec = spec_by_key(key)
    if not spec.has_header:
        return load_headerless_sst(spec.path)
    return pd.read_csv(spec.path)


def load_headerless_sst(path=None) -> pd.DataFrame:
    path = spec_by_key("sst_raw").path if path is None else path
    df = pd.read_csv(path, header=None, names=["sentence", "polarity"])
    df["sentiment_label"] = df["polarity"].map(STRING_POLARITY)
    return df


def load_zuco_experiment(scale: str = "standard") -> pd.DataFrame:
    """400-row Experiment A table."""
    key = {"standard": "zuco_standard", "minmax": "zuco_minmax", "min-max": "zuco_minmax"}[
        scale
    ]
    df = load_csv(key)
    df["sentiment_label"] = df["sentiment_label"].astype(int)
    df["sentence_id"] = df["sentence_id"].astype(int)
    return df


def load_full_sst_split(split: str) -> pd.DataFrame:
    key = {"train": "sst_train", "valid": "sst_valid", "test": "sst_test", "all": "sst_combined"}[
        split
    ]
    df = load_csv(key)
    df["sentiment_label"] = df["sentiment_label"].astype(int)
    return df


def fusion_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Select the five gaze columns regardless of nFix vs nFixations naming."""
    if set(FUSION_GAZE_COLUMNS).issubset(df.columns):
        cols = list(FUSION_GAZE_COLUMNS)
    elif set(FUSION_GAZE_COLUMNS_SST).issubset(df.columns):
        cols = list(FUSION_GAZE_COLUMNS_SST)
    else:
        raise KeyError(
            f"could not find fusion gaze columns in {list(df.columns)}"
        )
    out = df[cols].apply(pd.to_numeric, errors="coerce")
    out.columns = list(FUSION_GAZE_COLUMNS)
    return out


def label_counts(series: Iterable) -> dict[int, int]:
    counts = {0: 0, 1: 0, 2: 0}
    for value in series:
        counts[int(value)] += 1
    return counts


def word_rows_for_sentence(
    sentence_id: int,
    *,
    source: str = "average",
    subject: int | None = None,
) -> pd.DataFrame:
    """Word-level rows for one ZuCo sentence.

    ``source='average'`` reads ``word_averages_v2.csv``.
    ``source='subject'`` reads that subject's word CSV.
    """
    if source == "average":
        df = load_csv("zuco_word_average")
    elif source == "subject":
        if subject is None:
            raise ValueError("subject is required when source='subject'")
        df = pd.read_csv(subject_word_csv(subject))
    else:
        raise ValueError(f"unknown source {source!r}")

    tag = f"{sentence_id}_NR"
    rows = df[df["Sent_ID"].astype(str) == tag].copy()
    if rows.empty:
        # Some extracts use a bare integer in Sent_ID.
        rows = df[df["Sent_ID"].astype(str).str.startswith(f"{sentence_id}_")].copy()
    return rows.sort_values("Word_ID").reset_index(drop=True)


def sentence_text(sentence_id: int) -> tuple[str, int]:
    df = load_csv("zuco_text")
    row = df.loc[df["sentence_id"].astype(int) == int(sentence_id)]
    if row.empty:
        raise KeyError(f"sentence_id {sentence_id} not in ssts_ZuCo.csv")
    rec = row.iloc[0]
    return str(rec["sentence"]), int(rec["sentiment_label"])


def repo_relative(path) -> str:
    path = path if hasattr(path, "relative_to") else REPO_ROOT / path
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)
