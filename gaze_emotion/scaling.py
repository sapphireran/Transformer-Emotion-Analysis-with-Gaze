"""Feature scaling used when averaging ZuCo subjects.

``get_average_sentence_level.py`` applies sklearn MinMaxScaler and
StandardScaler after replacing raw zeros with NaN. ``utils_ZuCo.DataTransformer``
implements the same families by hand, plus mean-normalization.
"""

from __future__ import annotations

from typing import Literal

import numpy as np

FillMethod = Literal["zeros", "mean", "min"]
ScaleMethod = Literal["min-max", "mean-norm", "standard", "raw"]


def _as_2d(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim == 1:
        return array.reshape(-1, 1)
    if array.ndim != 2:
        raise ValueError(f"Expected 1D or 2D array, got shape {array.shape}")
    return array


def fill_missing(
    values: np.ndarray,
    method: FillMethod = "zeros",
    treat_zero_as_missing: bool = False,
) -> np.ndarray:
    """Replace NaNs (and optionally zeros) column-wise."""
    filled = _as_2d(values).copy()
    if treat_zero_as_missing:
        filled[filled == 0] = np.nan
    for col in range(filled.shape[1]):
        column = filled[:, col]
        missing = np.isnan(column)
        if not missing.any():
            continue
        if method == "zeros":
            replacement = 0.0
        elif method == "mean":
            observed = column[~missing]
            replacement = float(np.mean(observed)) if observed.size else 0.0
        elif method == "min":
            observed = column[~missing]
            replacement = float(np.min(observed)) if observed.size else 0.0
        else:
            raise ValueError(f"Unsupported fill method: {method}")
        column[missing] = replacement
    return filled


def min_max_scale(values: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Scale each column to ``[0, 1]``. Constant columns become zeros."""
    data = _as_2d(values)
    col_min = np.nanmin(data, axis=0)
    col_max = np.nanmax(data, axis=0)
    denom = np.maximum(col_max - col_min, eps)
    scaled = (data - col_min) / denom
    scaled[:, col_max == col_min] = 0.0
    return scaled


def mean_normalize(values: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Center each column and divide by its range."""
    data = _as_2d(values)
    col_min = np.nanmin(data, axis=0)
    col_max = np.nanmax(data, axis=0)
    denom = np.maximum(col_max - col_min, eps)
    scaled = (data - np.nanmean(data, axis=0)) / denom
    scaled[:, col_max == col_min] = 0.0
    return scaled


def standard_scale(values: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Z-score each column. Constant columns become zeros."""
    data = _as_2d(values)
    mean = np.nanmean(data, axis=0)
    std = np.nanstd(data, axis=0)
    safe_std = np.maximum(std, eps)
    scaled = (data - mean) / safe_std
    scaled[:, std == 0] = 0.0
    return scaled


def scale_features(values: np.ndarray, method: ScaleMethod) -> np.ndarray:
    """Dispatch to the scaling used by ``DataTransformer``."""
    if method == "raw":
        return _as_2d(values).copy()
    if method == "min-max":
        return min_max_scale(values)
    if method == "mean-norm":
        return mean_normalize(values)
    if method == "standard":
        return standard_scale(values)
    raise ValueError(f"Unsupported scale method: {method}")


def average_subject_tables(tables: list[np.ndarray]) -> np.ndarray:
    """Mean-stack subject matrices that share the same row alignment."""
    if not tables:
        raise ValueError("Need at least one subject table")
    stacked = np.stack([_as_2d(table) for table in tables], axis=0)
    return np.nanmean(stacked, axis=0)
