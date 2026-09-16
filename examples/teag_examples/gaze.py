"""Gaze-channel correlations and collinearity helpers."""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd


def pearson_matrix(df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    cols = list(columns)
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(missing)
    return df[cols].corr(method="pearson")


def collinear_pairs(
    corr: pd.DataFrame, threshold: float = 0.95
) -> list[tuple[str, str, float]]:
    """Upper-triangle pairs with |r| >= threshold (excludes the diagonal)."""
    cols = list(corr.columns)
    pairs: list[tuple[str, str, float]] = []
    for i, a in enumerate(cols):
        for b in cols[i + 1 :]:
            r = float(corr.loc[a, b])
            if abs(r) >= threshold:
                pairs.append((a, b, r))
    pairs.sort(key=lambda t: -abs(t[2]))
    return pairs


def feature_label_correlations(
    df: pd.DataFrame, columns: Iterable[str], label_col: str = "sentiment_label"
) -> pd.Series:
    cols = list(columns)
    return (
        df[cols + [label_col]]
        .corr(method="pearson")[label_col]
        .drop(label_col)
        .sort_values(key=np.abs, ascending=False)
    )


def almost_standardized(series: pd.Series, mean_tol: float = 1e-8, std_tol: float = 1e-2) -> bool:
    """True if mean≈0 and sample std≈1 (full-table z-score)."""
    return abs(float(series.mean())) <= mean_tol and abs(float(series.std(ddof=0)) - 1.0) <= std_tol


def in_unit_interval(series: pd.Series, atol: float = 1e-9) -> bool:
    return float(series.min()) >= -atol and float(series.max()) <= 1.0 + atol
