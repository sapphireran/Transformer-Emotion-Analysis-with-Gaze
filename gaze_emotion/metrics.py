"""Classification metrics matching the original training scripts.

``model_ZuCo_SST.py`` and ``model_full_SST.py`` report accuracy plus
weighted precision, recall, and F1 via scikit-learn. The implementations
here stay dependency-light so examples can print the same numbers.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np


def _as_int_vector(values: Iterable[int] | np.ndarray) -> np.ndarray:
    array = np.asarray(list(values) if not isinstance(values, np.ndarray) else values)
    if array.ndim != 1:
        raise ValueError(f"Expected a 1D label vector, got shape {array.shape}")
    return array.astype(int)


def confusion_matrix(y_true: Iterable[int], y_pred: Iterable[int], num_labels: int = 3) -> np.ndarray:
    """Return a ``[num_labels, num_labels]`` matrix of true x predicted counts."""
    true = _as_int_vector(y_true)
    pred = _as_int_vector(y_pred)
    if true.shape != pred.shape:
        raise ValueError("y_true and y_pred must have the same length")
    matrix = np.zeros((num_labels, num_labels), dtype=int)
    for t, p in zip(true, pred):
        if 0 <= t < num_labels and 0 <= p < num_labels:
            matrix[t, p] += 1
    return matrix


def accuracy_score(y_true: Iterable[int], y_pred: Iterable[int]) -> float:
    true = _as_int_vector(y_true)
    pred = _as_int_vector(y_pred)
    if true.size == 0:
        return 0.0
    return float(np.mean(true == pred))


def _per_class_prf(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    support = matrix.sum(axis=1).astype(float)
    predicted = matrix.sum(axis=0).astype(float)
    tp = np.diag(matrix).astype(float)
    precision = np.divide(tp, predicted, out=np.zeros_like(tp), where=predicted > 0)
    recall = np.divide(tp, support, out=np.zeros_like(tp), where=support > 0)
    denom = precision + recall
    f1 = np.divide(2 * precision * recall, denom, out=np.zeros_like(tp), where=denom > 0)
    return precision, recall, f1, support


def weighted_scores(y_true: Iterable[int], y_pred: Iterable[int], num_labels: int = 3) -> dict[str, float]:
    """Weighted precision / recall / F1 plus accuracy."""
    true = _as_int_vector(y_true)
    pred = _as_int_vector(y_pred)
    matrix = confusion_matrix(true, pred, num_labels=num_labels)
    precision, recall, f1, support = _per_class_prf(matrix)
    total = support.sum()
    if total == 0:
        return {"accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0}
    weights = support / total
    return {
        "accuracy": accuracy_score(true, pred),
        "precision": float(np.sum(precision * weights)),
        "recall": float(np.sum(recall * weights)),
        "f1": float(np.sum(f1 * weights)),
    }


def classification_report(
    y_true: Iterable[int],
    y_pred: Iterable[int],
    num_labels: int = 3,
    label_names: dict[int, str] | None = None,
) -> str:
    """Pretty-print a small per-class table plus weighted averages."""
    from .constants import LABEL_ID_TO_NAME

    names = label_names or LABEL_ID_TO_NAME
    matrix = confusion_matrix(y_true, y_pred, num_labels=num_labels)
    precision, recall, f1, support = _per_class_prf(matrix)
    lines = [
        f"{'label':<12}{'precision':>12}{'recall':>12}{'f1':>12}{'support':>10}",
        "-" * 58,
    ]
    for idx in range(num_labels):
        name = names.get(idx, str(idx))
        lines.append(
            f"{name:<12}{precision[idx]:12.4f}{recall[idx]:12.4f}{f1[idx]:12.4f}{int(support[idx]):10d}"
        )
    scores = weighted_scores(y_true, y_pred, num_labels=num_labels)
    lines.append("-" * 58)
    lines.append(
        f"{'weighted':<12}{scores['precision']:12.4f}{scores['recall']:12.4f}"
        f"{scores['f1']:12.4f}{int(support.sum()):10d}"
    )
    lines.append(f"accuracy: {scores['accuracy']:.4f}")
    return "\n".join(lines)
