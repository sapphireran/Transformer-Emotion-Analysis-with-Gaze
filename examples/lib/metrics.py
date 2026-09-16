"""Metric helpers that mirror the training scripts without importing them."""

from __future__ import annotations

from typing import Iterable

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


def weighted_classification_scores(
    labels: Iterable[int],
    preds: Iterable[int],
) -> dict[str, float]:
    """Same four scores as ``calculate_metrics`` in the training scripts."""
    y_true = np.asarray(list(labels))
    y_pred = np.asarray(list(preds))
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_weighted": float(
            precision_score(y_true, y_pred, average="weighted", zero_division=0)
        ),
        "recall_weighted": float(
            recall_score(y_true, y_pred, average="weighted", zero_division=0)
        ),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
    }


def majority_baseline(labels: Iterable[int]) -> dict[str, float]:
    """Predict the most common class everywhere — a sanity floor."""
    y = np.asarray(list(labels))
    if y.size == 0:
        raise ValueError("majority_baseline got an empty label list")
    values, counts = np.unique(y, return_counts=True)
    majority = int(values[counts.argmax()])
    preds = np.full_like(y, majority)
    scores = weighted_classification_scores(y, preds)
    scores["majority_class"] = majority
    return scores
