"""Weighted classification metrics matching the original training scripts."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


@dataclass(frozen=True)
class MetricBundle:
    accuracy: float
    precision: float
    recall: float
    f1: float

    def as_dict(self) -> dict[str, float]:
        return {
            "accuracy": self.accuracy,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
        }

    def pretty(self, digits: int = 4) -> str:
        parts = [f"{name}={value:.{digits}f}" for name, value in self.as_dict().items()]
        return ", ".join(parts)


def classification_metrics(preds: ArrayLike, labels: ArrayLike) -> MetricBundle:
    """Weighted P/R/F1 plus accuracy, same averages as model_*_SST.py."""
    y_true = np.asarray(labels)
    y_pred = np.asarray(preds)
    if y_true.shape != y_pred.shape:
        raise ValueError(f"preds/labels shape mismatch: {y_pred.shape} vs {y_true.shape}")
    if y_true.size == 0:
        raise ValueError("cannot score an empty prediction list")
    return MetricBundle(
        accuracy=float(accuracy_score(y_true, y_pred)),
        precision=float(precision_score(y_true, y_pred, average="weighted", zero_division=0)),
        recall=float(recall_score(y_true, y_pred, average="weighted", zero_division=0)),
        f1=float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
    )


def majority_baseline(labels: ArrayLike) -> tuple[int, MetricBundle]:
    """Predict the most frequent class everywhere. Useful as a dumb floor."""
    y_true = np.asarray(labels)
    if y_true.size == 0:
        raise ValueError("cannot build a majority baseline from no labels")
    values, counts = np.unique(y_true, return_counts=True)
    majority = int(values[int(np.argmax(counts))])
    preds = np.full_like(y_true, majority)
    return majority, classification_metrics(preds, y_true)
