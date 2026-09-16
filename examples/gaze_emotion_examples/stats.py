"""Descriptive statistics for sentence- and word-level gaze tables."""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd


def describe_numeric(frame: pd.DataFrame, columns: Iterable[str] | None = None) -> pd.DataFrame:
    """Mean, std, min, max, and selected quantiles for numeric columns."""
    if columns is None:
        numeric = frame.select_dtypes(include=[np.number])
    else:
        numeric = frame.loc[:, list(columns)]
    if numeric.empty:
        return pd.DataFrame()
    desc = numeric.describe(percentiles=[0.25, 0.5, 0.75]).T
    desc = desc.rename(columns={"50%": "median"})
    desc["missing"] = numeric.isna().sum()
    return desc.reset_index().rename(columns={"index": "column"})


def correlation_matrix(frame: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    """Pearson correlation among the requested columns."""
    return frame.loc[:, list(columns)].corr(method="pearson")


def grouped_means(
    frame: pd.DataFrame,
    group_column: str,
    value_columns: Iterable[str],
) -> pd.DataFrame:
    """Mean of each value column inside each group."""
    return frame.groupby(group_column, dropna=False)[list(value_columns)].mean()


def high_correlations(corr: pd.DataFrame, threshold: float = 0.8) -> pd.DataFrame:
    """Pairs whose absolute Pearson correlation meets ``threshold``."""
    records = []
    cols = list(corr.columns)
    for i, left in enumerate(cols):
        for right in cols[i + 1 :]:
            value = float(corr.loc[left, right])
            if abs(value) >= threshold:
                records.append({"left": left, "right": right, "pearson": value})
    return pd.DataFrame(records).sort_values("pearson", key=np.abs, ascending=False)


def markdown_table(frame: pd.DataFrame, float_fmt: str = "{:.3f}") -> str:
    """Format a DataFrame as a markdown table with compact floats."""
    if frame.empty:
        return "_empty table_"
    columns = [str(col) for col in frame.columns]
    lines = ["| " + " | ".join(columns) + " |"]
    lines.append("| " + " | ".join("---" for _ in columns) + " |")
    for _, row in frame.iterrows():
        cells = []
        for value in row.tolist():
            if isinstance(value, (float, np.floating)):
                cells.append(float_fmt.format(float(value)))
            else:
                cells.append(str(value))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)
