#!/usr/bin/env python3
"""Linear softmax baselines on Track A (no transformer).

Fits multinomial logistic regression with numpy on the 400-sentence
z-scored ZuCo table and reports 5-fold metrics for:

  * majority class
  * sentence character length only
  * the same 5 gaze features the fusion head sees
  * gaze + length

If gaze-only is near majority, concatenating those five numbers onto
RoBERTa is unlikely to move accuracy much unless the encoder *uses* them
as a gate. This is the number to write next to a GPU fusion run.

Usage (repo root):

    python examples/gaze_only_baseline.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from examples.common import (
    GAZE5_ZUCO,
    ZUCO_SST_DIR,
    classification_report,
    format_report,
    int_col,
    majority_baseline,
    matrix,
    read_dicts,
    stratified_kfold_indices,
)

N_CLASSES = 3
N_SPLITS = 5
SEED = 42
STEPS = 400
LR = 0.15
L2 = 1e-2


def softmax(logits: np.ndarray) -> np.ndarray:
    z = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(z)
    return exp / exp.sum(axis=1, keepdims=True)


def add_bias(x: np.ndarray) -> np.ndarray:
    return np.concatenate([x, np.ones((x.shape[0], 1))], axis=1)


def fit_softmax(x: np.ndarray, y: np.ndarray, steps: int = STEPS, lr: float = LR, l2: float = L2) -> np.ndarray:
    """Return weight matrix of shape (features+1, n_classes)."""
    xb = add_bias(x)
    n, f = xb.shape
    rng = np.random.default_rng(0)
    w = rng.normal(0, 0.01, size=(f, N_CLASSES))
    y_oh = np.eye(N_CLASSES)[y.astype(int)]
    for _ in range(steps):
        probs = softmax(xb @ w)
        grad = xb.T @ (probs - y_oh) / n
        grad[:-1] += l2 * w[:-1]  # do not L2-regularize bias
        w -= lr * grad
    return w


def predict(w: np.ndarray, x: np.ndarray) -> np.ndarray:
    return np.argmax(add_bias(x) @ w, axis=1)


def standardize_train(train: np.ndarray, test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = train.mean(axis=0)
    std = train.std(axis=0)
    std = np.where(std < 1e-8, 1.0, std)
    return (train - mean) / std, (test - mean) / std


def run_cv(name: str, x: np.ndarray, y: np.ndarray, kind: str) -> dict:
    fold_acc = []
    fold_f1 = []
    all_true = []
    all_pred = []
    for train_idx, test_idx in stratified_kfold_indices(y, n_splits=N_SPLITS, seed=SEED):
        x_train, x_test = x[train_idx], x[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        if kind == "majority":
            pred = majority_baseline(y_train, y_test.size)
        else:
            x_train, x_test = standardize_train(x_train, x_test)
            weights = fit_softmax(x_train, y_train)
            pred = predict(weights, x_test)
        report = classification_report(y_test, pred)
        fold_acc.append(report["accuracy"])
        fold_f1.append(report["weighted"]["f1"])
        all_true.append(y_test)
        all_pred.append(pred)
    y_true = np.concatenate(all_true)
    y_pred = np.concatenate(all_pred)
    pooled = classification_report(y_true, y_pred)
    print(f"\n===== {name} =====")
    print(
        f"5-fold mean acc={np.mean(fold_acc):.4f}  "
        f"mean weighted F1={np.mean(fold_f1):.4f}  "
        f"(seed={SEED}, same fold recipe as StratifiedKFold)"
    )
    print("pooled predictions across folds:")
    print(format_report(pooled))
    return pooled


def main() -> None:
    rows = read_dicts(ZUCO_SST_DIR / "combined_sst_et_standard.csv")
    y = int_col(rows, "sentiment_label")
    gaze = matrix(rows, GAZE5_ZUCO)
    length = np.array([[len(row["sentence"])] for row in rows], dtype=np.float64)
    both = np.concatenate([gaze, length], axis=1)

    print(f"rows={len(rows)}  gaze dim={gaze.shape[1]}  labels={ {int(k): int(v) for k, v in zip(*np.unique(y, return_counts=True))} }")
    print(f"softmax GD: steps={STEPS} lr={LR} l2={L2}")
    print("Features are re-standardized *inside each fold* (train mean/std).")

    run_cv("majority class", gaze, y, kind="majority")
    run_cv("character length only", length, y, kind="linear")
    run_cv("gaze only (nFixations, FFD, GPT, TRT, GD)", gaze, y, kind="linear")
    run_cv("gaze + character length", both, y, kind="linear")

    print()
    print("How to read this next to model_ZuCo_SST.py:")
    print("  * RoBERTa text-only should beat these numbers by a wide margin.")
    print("  * Fusion needs to beat *text-only*, not merely beat gaze-only.")
    print("  * If gaze-only ≈ majority, treat Track A fusion gains with caution.")


if __name__ == "__main__":
    main()
