"""Evaluation helpers that always accumulate the full prediction list."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)


@dataclass(frozen=True)
class MetricBundle:
    accuracy: float
    precision_weighted: float
    recall_weighted: float
    f1_weighted: float
    f1_macro: float
    n: int

    def as_dict(self) -> dict[str, float | int]:
        return {
            "accuracy": self.accuracy,
            "precision_weighted": self.precision_weighted,
            "recall_weighted": self.recall_weighted,
            "f1_weighted": self.f1_weighted,
            "f1_macro": self.f1_macro,
            "n": self.n,
        }

    def format_line(self, name: str, width: int = 22) -> str:
        return (
            f"{name:<{width}} acc={self.accuracy:.4f}  "
            f"P_w={self.precision_weighted:.4f}  "
            f"R_w={self.recall_weighted:.4f}  "
            f"F1_w={self.f1_weighted:.4f}  "
            f"F1_macro={self.f1_macro:.4f}  n={self.n}"
        )


def score_predictions(
    y_true: Sequence[int] | np.ndarray,
    y_pred: Sequence[int] | np.ndarray,
    *,
    zero_division: int = 0,
) -> MetricBundle:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if y_true.shape != y_pred.shape:
        raise ValueError(f"shape mismatch: {y_true.shape} vs {y_pred.shape}")
    return MetricBundle(
        accuracy=float(accuracy_score(y_true, y_pred)),
        precision_weighted=float(
            precision_score(y_true, y_pred, average="weighted", zero_division=zero_division)
        ),
        recall_weighted=float(
            recall_score(y_true, y_pred, average="weighted", zero_division=zero_division)
        ),
        f1_weighted=float(
            f1_score(y_true, y_pred, average="weighted", zero_division=zero_division)
        ),
        f1_macro=float(f1_score(y_true, y_pred, average="macro", zero_division=zero_division)),
        n=int(y_true.size),
    )


def majority_baseline(y_true: Sequence[int] | np.ndarray) -> MetricBundle:
    y_true = np.asarray(y_true)
    values, counts = np.unique(y_true, return_counts=True)
    majority = int(values[int(np.argmax(counts))])
    pred = np.full_like(y_true, majority)
    return score_predictions(y_true, pred)


def mean_std(bundles: Iterable[MetricBundle], attr: str) -> tuple[float, float]:
    values = np.array([getattr(b, attr) for b in bundles], dtype=float)
    if values.size == 0:
        return float("nan"), float("nan")
    return float(values.mean()), float(values.std(ddof=0))
