"""Metric helpers, including a reproduction of the test-loop overwrite bug."""

from __future__ import annotations

from typing import Sequence

import numpy as np


def batch_slices(n: int, batch_size: int) -> list[slice]:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    return [slice(i, min(i + batch_size, n)) for i in range(0, n, batch_size)]


def full_predictions(preds: np.ndarray) -> np.ndarray:
    """What the test loop *intended*: accumulate every batch."""
    return np.asarray(preds)


def last_batch_predictions(preds: np.ndarray, batch_size: int) -> np.ndarray:
    """What ``model_full_SST.py`` actually does:

        all_preds = preds.cpu().numpy()   # assignment, not extend

    so only the last DataLoader batch survives. For the committed test
    split (1186 rows) and ``batch_size = 256`` that is the final 162 rows
    (~13.7% of the test set).
    """
    preds = np.asarray(preds)
    n = len(preds)
    slices = batch_slices(n, batch_size)
    return preds[slices[-1]]


def weighted_scores(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_weighted": float(precision_score(y_true, y_pred, average="weighted", zero_division=0)),
        "recall_weighted": float(recall_score(y_true, y_pred, average="weighted", zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "n": int(len(y_true)),
    }


def majority_baseline(y_true: Sequence[int]) -> dict[str, float]:
    y = np.asarray(y_true)
    values, counts = np.unique(y, return_counts=True)
    maj = int(values[np.argmax(counts)])
    pred = np.full_like(y, maj)
    out = weighted_scores(y, pred)
    out["majority_label"] = maj
    out["majority_prior"] = float(counts.max() / len(y))
    return out
