"""Weighted accuracy / precision / recall / F1 matching the trainer recipe."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Scores:
    accuracy: float
    precision: float
    recall: float
    f1: float
    support: int


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if y_true.size == 0:
        return float("nan")
    return float(np.mean(y_true == y_pred))


def _per_class(y_true: np.ndarray, y_pred: np.ndarray, label: int) -> tuple[float, float, float, int]:
    gold = y_true == label
    pred = y_pred == label
    support = int(gold.sum())
    tp = int((gold & pred).sum())
    fp = int((~gold & pred).sum())
    fn = int((gold & ~pred).sum())
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return prec, rec, f1, support


def weighted_scores(y_true: np.ndarray, y_pred: np.ndarray, labels=(0, 1, 2)) -> Scores:
    """sklearn ``average='weighted'`` for precision / recall / F1."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    n = int(y_true.size)
    if n == 0:
        return Scores(float("nan"), float("nan"), float("nan"), float("nan"), 0)
    precs, recs, f1s, weights = [], [], [], []
    for label in labels:
        p, r, f, s = _per_class(y_true, y_pred, label)
        precs.append(p)
        recs.append(r)
        f1s.append(f)
        weights.append(s)
    w = np.array(weights, dtype=np.float64)
    if w.sum() == 0:
        return Scores(accuracy(y_true, y_pred), 0.0, 0.0, 0.0, n)
    return Scores(
        accuracy=accuracy(y_true, y_pred),
        precision=float(np.average(precs, weights=w)),
        recall=float(np.average(recs, weights=w)),
        f1=float(np.average(f1s, weights=w)),
        support=n,
    )


def confusion(y_true: np.ndarray, y_pred: np.ndarray, n_labels: int = 3) -> np.ndarray:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    mat = np.zeros((n_labels, n_labels), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        mat[int(t), int(p)] += 1
    return mat
