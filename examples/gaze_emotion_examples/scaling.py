"""Checks that the checked-in scaled tables behave like their claimed transforms."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ScalerColumnReport:
    column: str
    mean: float
    std: float
    minimum: float
    maximum: float
    looks_standard: bool
    looks_minmax: bool


@dataclass(frozen=True)
class ScalerReport:
    """Summary of whether a table looks standard-scaled, min-max scaled, or neither."""

    columns: tuple[ScalerColumnReport, ...]
    standard_like: bool
    minmax_like: bool

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "column": item.column,
                    "mean": item.mean,
                    "std": item.std,
                    "min": item.minimum,
                    "max": item.maximum,
                    "looks_standard": item.looks_standard,
                    "looks_minmax": item.looks_minmax,
                }
                for item in self.columns
            ]
        )


def _column_report(name: str, values: np.ndarray) -> ScalerColumnReport:
    finite = values[np.isfinite(values)]
    mean = float(finite.mean()) if len(finite) else float("nan")
    std = float(finite.std(ddof=0)) if len(finite) else float("nan")
    minimum = float(finite.min()) if len(finite) else float("nan")
    maximum = float(finite.max()) if len(finite) else float("nan")
    looks_standard = abs(mean) < 0.05 and abs(std - 1.0) < 0.05
    looks_minmax = minimum >= -1e-9 and maximum <= 1.0 + 1e-9 and (maximum - minimum) > 0.5
    return ScalerColumnReport(
        column=name,
        mean=mean,
        std=std,
        minimum=minimum,
        maximum=maximum,
        looks_standard=looks_standard,
        looks_minmax=looks_minmax,
    )


def scaler_report(frame: pd.DataFrame, columns: list[str]) -> ScalerReport:
    """Inspect each column and decide which scaling family it resembles."""
    reports = tuple(_column_report(name, frame[name].to_numpy(dtype=float)) for name in columns)
    return ScalerReport(
        columns=reports,
        standard_like=all(item.looks_standard for item in reports),
        minmax_like=all(item.looks_minmax for item in reports),
    )


def rank_agreement(left: pd.Series, right: pd.Series) -> float:
    """Spearman-style rank correlation via Pearson on ranks.

    Monotone scalers (min-max, z-score) should keep this at 1.0 when they
    are applied independently per column.
    """
    left_rank = left.rank(method="average")
    right_rank = right.rank(method="average")
    return float(left_rank.corr(right_rank))


def columnwise_rank_agreement(
    raw: pd.DataFrame,
    scaled: pd.DataFrame,
    columns: list[str],
) -> pd.Series:
    """Rank correlation of each shared column between two tables."""
    return pd.Series(
        {name: rank_agreement(raw[name], scaled[name]) for name in columns},
        name="spearman",
    )
