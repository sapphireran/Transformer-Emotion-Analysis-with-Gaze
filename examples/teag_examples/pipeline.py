"""Reconstruct joined ZuCo tables from text ⨝ scaled gaze."""

from __future__ import annotations

import pandas as pd

from .io import load_sentence_average, load_zuco_combined, load_zuco_text
from .schema import ZUCO_JOIN_GAZE_COLS


def join_zuco_text_and_gaze(
    text: pd.DataFrame,
    gaze: pd.DataFrame,
    *,
    drop_cols: tuple[str, ...] = ("SentLen",),
) -> pd.DataFrame:
    """Inner-join ``ssts_ZuCo.csv`` onto a scaled average file.

    The committed ``combined_sst_et_*.csv`` files match this join with
    ``drop_cols=('SentLen',)``.
    """
    g = gaze.rename(columns={"id": "sentence_id"})
    if "sentence_id" not in g.columns:
        raise KeyError("gaze table needs an 'id' or 'sentence_id' column")
    merged = text.merge(g, on="sentence_id", how="inner", validate="one_to_one")
    drop = [c for c in drop_cols if c in merged.columns]
    if drop:
        merged = merged.drop(columns=drop)
    ordered = ["sentence_id", "sentence", "sentiment_label", *ZUCO_JOIN_GAZE_COLS]
    extra = [c for c in merged.columns if c not in ordered]
    return merged.loc[:, ordered + extra]


def reconstruct_zuco_combined(scaling: str = "standard") -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return ``(reconstructed, committed)`` for ``standard`` or ``min_max``."""
    text = load_zuco_text()
    gaze = load_sentence_average(kind=scaling)
    reconstructed = join_zuco_text_and_gaze(text, gaze)
    committed = load_zuco_combined(scaling=scaling)
    return reconstructed, committed


def max_abs_diff(a: pd.DataFrame, b: pd.DataFrame, columns: tuple[str, ...]) -> float:
    left = a.sort_values("sentence_id").reset_index(drop=True)
    right = b.sort_values("sentence_id").reset_index(drop=True)
    if len(left) != len(right):
        raise ValueError(f"row count {len(left)} != {len(right)}")
    diff = (left[list(columns)].to_numpy() - right[list(columns)].to_numpy()).astype(float)
    return float(abs(diff).max()) if diff.size else 0.0
