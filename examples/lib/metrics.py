"""Classification metrics without scikit-learn."""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Sequence, Tuple


def accuracy(y_true: Sequence[int], y_pred: Sequence[int]) -> float:
    _check(y_true, y_pred)
    if not y_true:
        return 0.0
    return sum(int(a == b) for a, b in zip(y_true, y_pred)) / len(y_true)


def confusion_matrix(
    y_true: Sequence[int], y_pred: Sequence[int], labels: Sequence[int]
) -> List[List[int]]:
    _check(y_true, y_pred)
    index = {label: i for i, label in enumerate(labels)}
    matrix = [[0 for _ in labels] for _ in labels]
    for truth, pred in zip(y_true, y_pred):
        if truth not in index or pred not in index:
            continue
        matrix[index[truth]][index[pred]] += 1
    return matrix


def _prf_from_counts(tp: int, fp: int, fn: int) -> Tuple[float, float, float]:
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    if precision + recall == 0.0:
        f1 = 0.0
    else:
        f1 = 2.0 * precision * recall / (precision + recall)
    return precision, recall, f1


def per_class_prf(
    y_true: Sequence[int], y_pred: Sequence[int], labels: Sequence[int]
) -> Dict[int, Dict[str, float]]:
    _check(y_true, y_pred)
    tp = defaultdict(int)
    fp = defaultdict(int)
    fn = defaultdict(int)
    support = defaultdict(int)
    for truth, pred in zip(y_true, y_pred):
        support[truth] += 1
        if truth == pred:
            tp[truth] += 1
        else:
            fp[pred] += 1
            fn[truth] += 1
    out: Dict[int, Dict[str, float]] = {}
    for label in labels:
        p, r, f1 = _prf_from_counts(tp[label], fp[label], fn[label])
        out[label] = {
            "precision": p,
            "recall": r,
            "f1": f1,
            "support": float(support[label]),
        }
    return out


def _average(
    per_class: Dict[int, Dict[str, float]],
    labels: Sequence[int],
    kind: str,
) -> Dict[str, float]:
    if kind == "macro":
        weights = {label: 1.0 for label in labels}
    elif kind == "weighted":
        weights = {label: per_class[label]["support"] for label in labels}
    else:
        raise ValueError(f"unknown average {kind!r}")
    total = sum(weights.values())
    if total == 0.0:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    return {
        metric: sum(per_class[label][metric] * weights[label] for label in labels) / total
        for metric in ("precision", "recall", "f1")
    }


def report(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    labels: Sequence[int] = (0, 1, 2),
) -> Dict[str, object]:
    per_class = per_class_prf(y_true, y_pred, labels)
    return {
        "accuracy": accuracy(y_true, y_pred),
        "macro": _average(per_class, labels, "macro"),
        "weighted": _average(per_class, labels, "weighted"),
        "per_class": per_class,
        "confusion": confusion_matrix(y_true, y_pred, labels),
    }


def _check(y_true: Sequence[int], y_pred: Sequence[int]) -> None:
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred length mismatch")
