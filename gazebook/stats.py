"""Descriptive stats used by the lab notes (correlations, ICC, rank)."""

from __future__ import annotations

from collections import Counter

import numpy as np


def pearson(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    if x.size != y.size:
        raise ValueError("pearson: length mismatch")
    if x.size < 2 or np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def corr_matrix(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=np.float64)
    return np.corrcoef(X, rowvar=False)


def eigenvalues_corr(X: np.ndarray) -> np.ndarray:
    C = corr_matrix(X)
    return np.sort(np.linalg.eigvalsh(C))


def condition_number_corr(X: np.ndarray) -> float:
    ev = eigenvalues_corr(X)
    ev = ev[ev > 0]
    if ev.size == 0:
        return float("inf")
    return float(ev.max() / ev.min())


def pairwise_reader_r(X: np.ndarray) -> tuple[float, float, float, int]:
    """Mean / min / max Pearson r across reader pairs. ``X`` is n × readers."""
    X = np.asarray(X, dtype=np.float64)
    n_readers = X.shape[1]
    rs: list[float] = []
    for i in range(n_readers):
        for j in range(i + 1, n_readers):
            r = pearson(X[:, i], X[:, j])
            if not np.isnan(r):
                rs.append(r)
    if not rs:
        return float("nan"), float("nan"), float("nan"), 0
    arr = np.array(rs)
    return float(arr.mean()), float(arr.min()), float(arr.max()), len(rs)


def icc1(X: np.ndarray) -> float:
    """One-way random ICC(1) for an n × k (targets × raters) matrix."""
    X = np.asarray(X, dtype=np.float64)
    n, k = X.shape
    grand = X.mean()
    bms = k * ((X.mean(axis=1) - grand) ** 2).sum() / (n - 1)
    wms = ((X - X.mean(axis=1, keepdims=True)) ** 2).sum() / (n * (k - 1))
    denom = bms + (k - 1) * wms
    if denom == 0:
        return float("nan")
    return float((bms - wms) / denom)


def class_counts(y: np.ndarray) -> dict[int, int]:
    c = Counter(int(v) for v in np.asarray(y).ravel())
    return {k: c.get(k, 0) for k in (0, 1, 2)}


def zscore(X: np.ndarray, ddof: int = 0) -> np.ndarray:
    X = np.asarray(X, dtype=np.float64)
    return (X - X.mean(axis=0)) / X.std(axis=0, ddof=ddof)


def minmax(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=np.float64)
    lo = X.min(axis=0)
    hi = X.max(axis=0)
    return (X - lo) / (hi - lo)


def residualize(y: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Return ``y`` after removing a least-squares line in ``x`` (with intercept)."""
    y = np.asarray(y, dtype=np.float64).ravel()
    x = np.asarray(x, dtype=np.float64).ravel()
    A = np.column_stack([np.ones(len(x)), x])
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    return y - A @ coef
