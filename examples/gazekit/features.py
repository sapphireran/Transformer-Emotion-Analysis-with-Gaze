"""Descriptive statistics for sentence- and word-level gaze tables."""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

from .schema import CANONICAL_GAZE, LABEL_NAMES, SENTIMENT_FROM_INT


def missingness_report(df: pd.DataFrame, columns: Iterable[str] | None = None) -> pd.DataFrame:
    cols = list(columns) if columns is not None else list(df.columns)
    rows = []
    for col in cols:
        if col not in df.columns:
            rows.append({"column": col, "present": False, "n_missing": None, "pct_missing": None})
            continue
        n_missing = int(df[col].isna().sum())
        rows.append(
            {
                "column": col,
                "present": True,
                "n_missing": n_missing,
                "pct_missing": n_missing / max(len(df), 1),
            }
        )
    return pd.DataFrame(rows)


def summarize_numeric(df: pd.DataFrame, columns: Iterable[str] | None = None) -> pd.DataFrame:
    cols = [c for c in (columns or df.columns) if c in df.columns and pd.api.types.is_numeric_dtype(df[c])]
    if not cols:
        return pd.DataFrame()
    desc = df[cols].describe(percentiles=[0.05, 0.25, 0.5, 0.75, 0.95]).T
    desc["n_missing"] = df[cols].isna().sum()
    desc["n_nonzero"] = (df[cols] != 0).sum()
    return desc.reset_index().rename(columns={"index": "column"})


def correlation_matrix(df: pd.DataFrame, columns: Iterable[str] = CANONICAL_GAZE) -> pd.DataFrame:
    cols = [c for c in columns if c in df.columns]
    return df[cols].corr(numeric_only=True)


def class_balance(df: pd.DataFrame, label_col: str = "sentiment_label") -> pd.DataFrame:
    if label_col not in df.columns:
        raise KeyError(label_col)
    counts = df[label_col].value_counts(dropna=False).sort_index()
    rows = []
    n = len(df)
    for label, count in counts.items():
        name = SENTIMENT_FROM_INT.get(int(label), str(label)) if pd.notna(label) else "NaN"
        rows.append(
            {
                "sentiment_label": label,
                "sentiment_name": name,
                "count": int(count),
                "fraction": float(count) / n if n else 0.0,
            }
        )
    return pd.DataFrame(rows)


def majority_fraction(df: pd.DataFrame, label_col: str = "sentiment_label") -> float:
    if df.empty:
        return 0.0
    return float(df[label_col].value_counts(normalize=True).max())


def gaze_by_sentiment(
    df: pd.DataFrame,
    columns: Iterable[str] = CANONICAL_GAZE,
    label_col: str = "sentiment_label",
) -> pd.DataFrame:
    cols = [c for c in columns if c in df.columns]
    grouped = df.groupby(label_col)[cols].agg(["mean", "std", "median"])
    grouped.columns = [f"{a}_{b}" for a, b in grouped.columns]
    grouped = grouped.reset_index()
    grouped["sentiment_name"] = grouped[label_col].map(
        lambda x: SENTIMENT_FROM_INT.get(int(x), str(x))
    )
    return grouped


def pairwise_mean_abs_diff(df: pd.DataFrame, columns: Iterable[str] = CANONICAL_GAZE) -> pd.DataFrame:
    """Absolute difference of class-conditional means for each channel.

    A large number relative to the channel's overall std is a hint that a
    linear gaze-only model has something to grab onto.
    """
    cols = [c for c in columns if c in df.columns]
    means = df.groupby("sentiment_label")[cols].mean()
    records = []
    labels = list(means.index)
    for i, a in enumerate(labels):
        for b in labels[i + 1 :]:
            for col in cols:
                records.append(
                    {
                        "channel": col,
                        "class_a": SENTIMENT_FROM_INT.get(int(a), str(a)),
                        "class_b": SENTIMENT_FROM_INT.get(int(b), str(b)),
                        "abs_mean_diff": float(abs(means.loc[a, col] - means.loc[b, col])),
                        "overall_std": float(df[col].std(ddof=0)),
                    }
                )
    out = pd.DataFrame.from_records(records)
    if not out.empty:
        out["diff_over_std"] = out["abs_mean_diff"] / out["overall_std"].replace(0, np.nan)
    return out


def word_length_effects(words: pd.DataFrame) -> pd.DataFrame:
    """Bin word length and report mean gaze. Expect a rising nFixations curve."""
    if "WordLen" not in words.columns:
        raise KeyError("WordLen")
    tmp = words.copy()
    tmp["len_bin"] = pd.cut(
        tmp["WordLen"],
        bins=[-0.5, 2.5, 4.5, 6.5, 8.5, 12.5, 100],
        labels=["1-2", "3-4", "5-6", "7-8", "9-12", "13+"],
    )
    cols = [c for c in CANONICAL_GAZE if c in tmp.columns]
    out = tmp.groupby("len_bin", observed=False)[cols].mean().reset_index()
    out["n_words"] = tmp.groupby("len_bin", observed=False).size().values
    return out


def sentence_id_from_sent_id(sent_id) -> int:
    return int(str(sent_id).split("_")[0])


def label_name(label: int) -> str:
    return SENTIMENT_FROM_INT.get(int(label), str(label))


def expected_label_order() -> tuple[str, ...]:
    return LABEL_NAMES
