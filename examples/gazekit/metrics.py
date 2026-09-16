"""Metric helpers that match the original scripts plus a macro variant."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


@dataclass(frozen=True)
class MetricSet:
    accuracy: float
    precision_weighted: float
    recall_weighted: float
    f1_weighted: float
    precision_macro: float
    recall_macro: float
    f1_macro: float
    n: int

    def as_dict(self) -> dict[str, float | int]:
        return {
            "accuracy": self.accuracy,
            "precision_weighted": self.precision_weighted,
            "recall_weighted": self.recall_weighted,
            "f1_weighted": self.f1_weighted,
            "precision_macro": self.precision_macro,
            "recall_macro": self.recall_macro,
            "f1_macro": self.f1_macro,
            "n": self.n,
        }


def calculate_metrics(preds, labels) -> MetricSet:
    """Same weighted trio as ``model_*.py``, plus macro averages."""
    y_true = np.asarray(labels)
    y_pred = np.asarray(preds)
    kw = dict(zero_division=0)
    return MetricSet(
        accuracy=float(accuracy_score(y_true, y_pred)),
        precision_weighted=float(precision_score(y_true, y_pred, average="weighted", **kw)),
        recall_weighted=float(recall_score(y_true, y_pred, average="weighted", **kw)),
        f1_weighted=float(f1_score(y_true, y_pred, average="weighted", **kw)),
        precision_macro=float(precision_score(y_true, y_pred, average="macro", **kw)),
        recall_macro=float(recall_score(y_true, y_pred, average="macro", **kw)),
        f1_macro=float(f1_score(y_true, y_pred, average="macro", **kw)),
        n=int(y_true.size),
    )


def mean_and_std(values) -> tuple[float, float]:
    arr = np.asarray(values, dtype="float64")
    if arr.size == 0:
        return float("nan"), float("nan")
    return float(arr.mean()), float(arr.std(ddof=1) if arr.size > 1 else 0.0)
