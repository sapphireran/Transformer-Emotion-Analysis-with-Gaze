"""Scaling, correlations, and word-to-sentence aggregation for the checked-in tables."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from .schema import ZUCO_WORD_ET_COLUMNS


@dataclass(frozen=True)
class FeatureReport:
    name: str
    n_rows: int
    summary: pd.DataFrame
    corr_with_label: pd.Series | None
    pairwise_corr: pd.DataFrame


def summarize_numeric(frame: pd.DataFrame, columns: Iterable[str] | None = None) -> pd.DataFrame:
    cols = list(columns) if columns is not None else list(frame.select_dtypes(include="number").columns)
    missing = [col for col in cols if col not in frame.columns]
    if missing:
        raise KeyError(f"Missing numeric columns: {missing}")
    return frame[cols].describe().T


def correlation_with_label(
    frame: pd.DataFrame,
    feature_columns: Iterable[str],
    label_column: str = "sentiment_label",
) -> pd.Series:
    cols = list(feature_columns)
    if label_column not in frame.columns:
        raise KeyError(f"{label_column} is not in the frame")
    return frame[cols + [label_column]].corr(numeric_only=True)[label_column].drop(label_column)


def feature_correlation_matrix(frame: pd.DataFrame, feature_columns: Iterable[str]) -> pd.DataFrame:
    cols = list(feature_columns)
    return frame[cols].corr(numeric_only=True)


def scale_numeric_frame(
    frame: pd.DataFrame,
    columns: Iterable[str],
    method: str = "standard",
) -> pd.DataFrame:
    """Return a copy with selected columns scaled. id-like columns are left alone."""
    cols = list(columns)
    missing = [col for col in cols if col not in frame.columns]
    if missing:
        raise KeyError(f"Cannot scale missing columns: {missing}")
    if method == "standard":
        scaler = StandardScaler()
    elif method in {"minmax", "min-max"}:
        scaler = MinMaxScaler()
    else:
        raise ValueError("method must be 'standard' or 'minmax'")
    out = frame.copy()
    out[cols] = scaler.fit_transform(out[cols])
    return out


def compare_scaling(frame: pd.DataFrame, columns: Iterable[str]) -> dict[str, pd.DataFrame]:
    """Rebuild min-max and standard views the same way get_average_sentence_level.py does."""
    cols = list(columns)
    return {
        "raw": frame[cols].copy(),
        "standard": scale_numeric_frame(frame, cols, "standard")[cols],
        "minmax": scale_numeric_frame(frame, cols, "minmax")[cols],
    }


def word_to_sentence_means(
    word_frame: pd.DataFrame,
    sentence_id_col: str = "Sent_ID",
    feature_columns: Iterable[str] = ZUCO_WORD_ET_COLUMNS,
) -> pd.DataFrame:
    """Average word-level ZuCo features back to one row per sentence id."""
    cols = list(feature_columns)
    missing = [col for col in [sentence_id_col, *cols] if col not in word_frame.columns]
    if missing:
        raise KeyError(f"Word table missing {missing}")
    grouped = word_frame.groupby(sentence_id_col, sort=True)[cols].mean()
    grouped = grouped.reset_index()
    grouped.insert(1, "n_words", word_frame.groupby(sentence_id_col).size().to_numpy())
    return grouped


def collinearity_pairs(corr: pd.DataFrame, threshold: float = 0.95) -> list[tuple[str, str, float]]:
    """Pairs of features whose absolute Pearson r is at or above threshold."""
    pairs: list[tuple[str, str, float]] = []
    cols = list(corr.columns)
    for i, left in enumerate(cols):
        for right in cols[i + 1 :]:
            value = float(corr.loc[left, right])
            if np.isfinite(value) and abs(value) >= threshold:
                pairs.append((left, right, value))
    return sorted(pairs, key=lambda item: -abs(item[2]))


def build_feature_report(
    name: str,
    frame: pd.DataFrame,
    feature_columns: Iterable[str],
    label_column: str | None = "sentiment_label",
) -> FeatureReport:
    cols = list(feature_columns)
    corr_label = None
    if label_column and label_column in frame.columns:
        corr_label = correlation_with_label(frame, cols, label_column)
    return FeatureReport(
        name=name,
        n_rows=int(len(frame)),
        summary=summarize_numeric(frame, cols),
        corr_with_label=corr_label,
        pairwise_corr=feature_correlation_matrix(frame, cols),
    )
