"""Gaze-feature summaries that the numbered examples share."""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.feature_selection import f_classif

from .loading import LABEL_NAMES

FULL_SST_GAZE = ("nFix", "FFD", "GPT", "TRT", "GD")
ZUCO_FUSION_GAZE = ("nFixations", "FFD", "GPT", "TRT", "GD")
ZUCO_ALL_GAZE = (
    "omissionRate",
    "nFixations",
    "meanPupilSize",
    "GD",
    "TRT",
    "FFD",
    "SFD",
    "GPT",
)


def per_class_means(
    df: pd.DataFrame,
    feature_columns: Iterable[str],
    label_column: str = "sentiment_label",
) -> pd.DataFrame:
    """Mean of each gaze feature inside each polarity class."""
    cols = list(feature_columns)
    grouped = df.groupby(label_column, observed=True)[cols].mean()
    grouped.index = grouped.index.map(lambda i: LABEL_NAMES.get(int(i), str(i)))
    grouped.index.name = "class"
    return grouped


def feature_correlation(df: pd.DataFrame, feature_columns: Iterable[str]) -> pd.DataFrame:
    return df[list(feature_columns)].corr(method="pearson")


def feature_label_anova(
    df: pd.DataFrame,
    feature_columns: Iterable[str],
    label_column: str = "sentiment_label",
) -> pd.DataFrame:
    """One-way ANOVA F of each feature against the integer label."""
    cols = list(feature_columns)
    x = df[cols].to_numpy(dtype=float)
    y = df[label_column].to_numpy()
    f_values, p_values = f_classif(x, y)
    return pd.DataFrame(
        {"feature": cols, "f_statistic": f_values, "p_value": p_values}
    ).sort_values("f_statistic", ascending=False, ignore_index=True)


def subject_feature_panel(
    tables: dict[int, pd.DataFrame],
    feature_columns: Iterable[str] = ZUCO_ALL_GAZE,
    id_column: str = "id",
) -> pd.DataFrame:
    """Stack per-subject features with a ``subject`` column."""
    cols = list(feature_columns)
    frames = []
    for subject, df in tables.items():
        piece = df[[id_column, *cols]].copy()
        piece["subject"] = subject
        frames.append(piece)
    return pd.concat(frames, ignore_index=True)


def inter_subject_cv(
    panel: pd.DataFrame,
    feature_columns: Iterable[str],
    id_column: str = "id",
) -> pd.DataFrame:
    """Coefficient of variation across the 12 subjects, per sentence."""
    rows = []
    for feature in feature_columns:
        grouped = panel.groupby(id_column)[feature]
        mean = grouped.mean()
        std = grouped.std(ddof=1)
        cv = std / mean.replace(0, np.nan).abs()
        rows.append(
            pd.DataFrame(
                {
                    "sentence_id": mean.index.astype(int),
                    "feature": feature,
                    "subject_mean": mean.to_numpy(),
                    "subject_std": std.to_numpy(),
                    "subject_cv": cv.to_numpy(),
                }
            )
        )
    return pd.concat(rows, ignore_index=True)


def skip_rate_by_word_length(words: pd.DataFrame) -> pd.DataFrame:
    frame = words.copy()
    frame["skipped"] = frame["nFixations"].fillna(0).eq(0)
    grouped = frame.groupby("WordLen", observed=True)["skipped"]
    out = grouped.agg(skip_rate="mean", n_words="size").reset_index()
    out["skip_rate"] = out["skip_rate"].astype(float)
    return out.sort_values("WordLen").reset_index(drop=True)


def length_feature_correlation(
    df: pd.DataFrame,
    feature_columns: Iterable[str],
    length: pd.Series,
) -> pd.DataFrame:
    """Pearson r of each gaze feature with a length proxy (tokens or SentLen)."""
    rows = []
    length = pd.Series(length, dtype=float)
    for col in feature_columns:
        r = float(pd.Series(df[col], dtype=float).corr(length))
        rows.append({"feature": col, "r_with_length": r})
    return pd.DataFrame(rows)


def words_for_sentence(words: pd.DataFrame, sentence_id: int) -> pd.DataFrame:
    """Word-level rows whose ``Sent_ID`` prefix matches ``sentence_id``."""
    prefixes = words["Sent_ID"].astype(str).str.split("_").str[0].astype(int)
    return words.loc[prefixes == int(sentence_id)].copy()


def sentence_skip_rate(words: pd.DataFrame) -> pd.DataFrame:
    """Skip rate per ZuCo sentence, keyed by the integer prefix of ``Sent_ID``."""
    frame = words.copy()
    frame["sentence_id"] = frame["Sent_ID"].astype(str).str.split("_").str[0].astype(int)
    frame["skipped"] = frame["nFixations"].fillna(0).eq(0)
    out = (
        frame.groupby("sentence_id", observed=True)["skipped"]
        .agg(skip_rate="mean", n_words="size")
        .reset_index()
    )
    out["skip_rate"] = out["skip_rate"].astype(float)
    return out
