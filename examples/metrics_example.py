#!/usr/bin/env python3
"""Metric helpers: weighted vs macro F1, and the Track B last-batch trap.

`model_full_SST.py` replaces `all_preds` every test batch instead of
extending it. This script builds a fake 3-class loader and shows how
different last-batch metrics are from the full-set metrics.

Usage (repo root):

    python examples/metrics_example.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from examples.common import classification_report, format_report, majority_baseline


def _fake_loader(n: int = 1186, batch_size: int = 256, seed: int = 0):
    """SST-3-ish imbalance and a moderately good classifier."""
    rng = np.random.default_rng(seed)
    # Rough SST-3 prior: more negative/positive than neutral.
    labels = rng.choice([0, 1, 2], size=n, p=[0.42, 0.18, 0.40])
    preds = labels.copy()
    flip = rng.random(n) < 0.28
    preds[flip] = rng.integers(0, 3, size=int(flip.sum()))
    for start in range(0, n, batch_size):
        yield labels[start : start + batch_size], preds[start : start + batch_size]


def _score(y_true: np.ndarray, y_pred: np.ndarray, title: str) -> None:
    print(f"\n===== {title}  n={y_true.size} =====")
    print(format_report(classification_report(y_true, y_pred)))


def main() -> None:
    print("Sklearn-style weighted F1 weights each class F1 by support.")
    print("Macro F1 averages the three class F1s equally — harsher on")
    print("neutral if that class is rare and hard.")
    print()

    y = np.array([0, 0, 0, 0, 1, 1, 2, 2, 2])
    pred_good_on_majority = np.array([0, 0, 0, 0, 0, 1, 2, 2, 0])
    _score(y, pred_good_on_majority, "toy: always-ish negative, misses some")

    pred_balanced = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])
    _score(y, pred_balanced, "toy: one error per class")

    print()
    print("Majority baseline on the toy labels:")
    maj = majority_baseline(y, y.size)
    _score(y, maj, "majority")

    print("\n" + "=" * 60)
    print("Track B test-loop overwrite demo")
    print("=" * 60)
    print("Validation in model_full_SST.py does:  all_preds.extend(batch)")
    print("Test loop does:                        all_preds = batch")
    print("so the printed Test Acc is the LAST batch only.")

    full_true = []
    full_pred = []
    last_true = last_pred = None
    n_batches = 0
    for y_b, p_b in _fake_loader():
        n_batches += 1
        full_true.append(y_b)
        full_pred.append(p_b)
        last_true, last_pred = y_b, p_b

    y_full = np.concatenate(full_true)
    p_full = np.concatenate(full_pred)
    print(f"\nsimulated test n={y_full.size}  batches={n_batches}  last_batch={last_true.size}")
    _score(y_full, p_full, "correct accumulation (extend every batch)")
    _score(last_true, last_pred, "buggy test loop (last batch only)")

    print()
    print("Until model_full_SST.py is patched, quote validation metrics or")
    print("score saved logits yourself. examples/common.py:classification_report")
    print("is the intended definition (weighted / macro / per-class).")


if __name__ == "__main__":
    main()
