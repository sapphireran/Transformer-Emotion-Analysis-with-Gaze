"""Weighted classification metrics matching the training scripts."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


def calculate_metrics(preds, labels) -> tuple[float, float, float, float]:
    """Return accuracy, weighted precision, recall, F1.

    Same four values as ``calculate_metrics`` in ``model_full_SST.py`` and
    ``model_ZuCo_SST.py``.
    """
    preds = np.asarray(preds)
    labels = np.asarray(labels)
    accuracy = accuracy_score(labels, preds)
    precision = precision_score(labels, preds, average="weighted", zero_division=0)
    recall = recall_score(labels, preds, average="weighted", zero_division=0)
    f1 = f1_score(labels, preds, average="weighted", zero_division=0)
    return float(accuracy), float(precision), float(recall), float(f1)
