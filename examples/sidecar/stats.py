"""Small statistical helpers used by several atlas examples."""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd


def label_counts(series: pd.Series) -> pd.DataFrame:
    counts = series.value_counts().sort_index()
    out = counts.rename("n").to_frame()
    out["share"] = out["n"] / out["n"].sum()
    out.index.name = "label"
    return out.reset_index()


def class_conditional_means(df: pd.DataFrame, cols: Sequence[str], label_col: str = "sentiment_label") -> pd.DataFrame:
    return df.groupby(label_col)[list(cols)].mean().reset_index()


def pearson_with_label(df: pd.DataFrame, cols: Sequence[str], label_col: str = "sentiment_label") -> pd.Series:
    return df[list(cols)].corrwith(df[label_col])


def token_count(series: pd.Series) -> pd.Series:
    return series.astype(str).str.split().str.len()


def residualize(y: np.ndarray, confound: np.ndarray) -> np.ndarray:
    """Return y minus the least-squares prediction from a single confound column."""
    y = np.asarray(y, dtype=np.float64)
    x = np.asarray(confound, dtype=np.float64).reshape(-1, 1)
    A = np.concatenate([np.ones((len(x), 1)), x], axis=1)
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    return y - A @ coef


def shuffle_columns(df: pd.DataFrame, cols: Sequence[str], seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    out = df.copy()
    perm = rng.permutation(len(out))
    out.loc[:, list(cols)] = out.loc[:, list(cols)].to_numpy()[perm]
    return out
