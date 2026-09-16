"""Shared metrics for the personal sklearn baselines and reports."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable, Sequence

from tea_gaze.features import SENTIMENT_LABELS


@dataclass(frozen=True)
class MetricSet:
    accuracy: float
    precision: float
    recall: float
    f1: float
    support: int

    def as_dict(self) -> dict[str, float | int]:
        return asdict(self)

    def format_line(self, name: str) -> str:
        return (
            f"{name}: acc={self.accuracy:.4f}  p={self.precision:.4f}  "
            f"r={self.recall:.4f}  f1={self.f1:.4f}  n={self.support}"
        )


def _safe_div(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return numerator / denominator


def confusion_matrix(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    labels: Sequence[int] | None = None,
) -> list[list[int]]:
    labels = list(labels if labels is not None else SENTIMENT_LABELS.keys())
    index = {label: i for i, label in enumerate(labels)}
    matrix = [[0 for _ in labels] for _ in labels]
    for truth, pred in zip(y_true, y_pred, strict=True):
        if truth not in index or pred not in index:
            continue
        matrix[index[truth]][index[pred]] += 1
    return matrix


def _per_class_f1(matrix: list[list[int]]) -> list[float]:
    scores = []
    for i, _ in enumerate(matrix):
        tp = matrix[i][i]
        fp = sum(matrix[row][i] for row in range(len(matrix)) if row != i)
        fn = sum(matrix[i][col] for col in range(len(matrix)) if col != i)
        precision = _safe_div(tp, tp + fp)
        recall = _safe_div(tp, tp + fn)
        if precision + recall == 0:
            scores.append(0.0)
        else:
            scores.append(2 * precision * recall / (precision + recall))
    return scores


def weighted_metrics(y_true: Sequence[int], y_pred: Sequence[int]) -> MetricSet:
    """Weighted precision / recall / F1, matching the original training scripts."""
    y_true = [int(v) for v in y_true]
    y_pred = [int(v) for v in y_pred]
    labels = list(SENTIMENT_LABELS.keys())
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    support = [sum(row) for row in matrix]
    total = sum(support)
    accuracy = _safe_div(sum(matrix[i][i] for i in range(len(labels))), total)

    precisions = []
    recalls = []
    f1s = []
    for i, _ in enumerate(labels):
        tp = matrix[i][i]
        fp = sum(matrix[row][i] for row in range(len(labels)) if row != i)
        fn = sum(matrix[i][col] for col in range(len(labels)) if col != i)
        precision = _safe_div(tp, tp + fp)
        recall = _safe_div(tp, tp + fn)
        if precision + recall == 0:
            f1 = 0.0
        else:
            f1 = 2 * precision * recall / (precision + recall)
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)

    def weighted(values: Iterable[float]) -> float:
        return _safe_div(sum(value * n for value, n in zip(values, support)), total)

    return MetricSet(
        accuracy=accuracy,
        precision=weighted(precisions),
        recall=weighted(recalls),
        f1=weighted(f1s),
        support=total,
    )


def mean_metrics(sets: Sequence[MetricSet]) -> MetricSet:
    if not sets:
        raise ValueError("mean_metrics received no folds")
    n = len(sets)
    return MetricSet(
        accuracy=sum(item.accuracy for item in sets) / n,
        precision=sum(item.precision for item in sets) / n,
        recall=sum(item.recall for item in sets) / n,
        f1=sum(item.f1 for item in sets) / n,
        support=sum(item.support for item in sets),
    )


def format_confusion(matrix: list[list[int]]) -> str:
    labels = [SENTIMENT_LABELS[i] for i in range(len(matrix))]
    header = "true\\pred | " + " | ".join(f"{name:8s}" for name in labels)
    lines = [header, "-" * len(header)]
    for i, row in enumerate(matrix):
        cells = " | ".join(f"{value:8d}" for value in row)
        lines.append(f"{labels[i]:9s} | {cells}")
    return "\n".join(lines)
