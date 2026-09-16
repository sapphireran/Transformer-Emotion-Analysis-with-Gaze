#!/usr/bin/env python3
"""Linear ET-only baseline vs a majority-class dummy.

This is the cheapest question you can ask before spending a GPU hour on
RoBERTa + gaze: do the five committed eye-tracking columns carry *any*
linear signal about the 3-way sentiment label?

Two evaluations:

1. ZuCo ∩ SST (400 recorded-gaze sentences) — StratifiedKFold(5), same
   seed the training script uses.
2. Full SST projected gaze — fit on train_full_sst.csv, score on
   test_full_sst.csv.
"""

from __future__ import annotations

import sys
from pathlib import Path

_EXAMPLES = Path(__file__).resolve().parent
if str(_EXAMPLES) not in sys.path:
    sys.path.insert(0, str(_EXAMPLES))

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from paths import (
    FULL_SST_TEST,
    FULL_SST_TRAIN,
    SST_ET_COLS,
    ZUCO_ET_COLS,
    ZUCO_STANDARD,
)


def _pack(df: pd.DataFrame, cols: list[str]) -> tuple[np.ndarray, np.ndarray]:
    x = df[cols].to_numpy(dtype=np.float64)
    y = df["sentiment_label"].to_numpy(dtype=np.int64)
    return x, y


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float]:
    return (
        float(accuracy_score(y_true, y_pred)),
        float(f1_score(y_true, y_pred, average="weighted")),
    )


def _clf() -> Pipeline:
    # Standardize inside the fold so the ZuCo table (already z-scored) and
    # the SST table (also centered, but not unit-checked) share one path.
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "logreg",
                LogisticRegression(
                    max_iter=500,
                    solver="lbfgs",
                    C=1.0,
                ),
            ),
        ]
    )


def _dummy() -> DummyClassifier:
    return DummyClassifier(strategy="most_frequent")


def cv_report(name: str, x: np.ndarray, y: np.ndarray, n_splits: int = 5) -> None:
    kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    rows = {"logreg": [], "majority": []}
    for train_idx, test_idx in kf.split(x, y):
        for key, factory in (("logreg", _clf), ("majority", _dummy)):
            model = factory()
            model.fit(x[train_idx], y[train_idx])
            pred = model.predict(x[test_idx])
            rows[key].append(_metrics(y[test_idx], pred))

    print(f"=== {name}  (StratifiedKFold n={n_splits}, seed=42) ===")
    print(f"n={len(y)}  label counts={ {int(k): int(v) for k, v in zip(*np.unique(y, return_counts=True))} }")
    for key in ("majority", "logreg"):
        acc = np.mean([m[0] for m in rows[key]])
        f1 = np.mean([m[1] for m in rows[key]])
        acc_std = np.std([m[0] for m in rows[key]])
        f1_std = np.std([m[1] for m in rows[key]])
        print(
            f"  {key:<10}  acc={acc:.4f} ± {acc_std:.4f}   "
            f"weighted-F1={f1:.4f} ± {f1_std:.4f}"
        )
    delta = np.mean([m[0] for m in rows["logreg"]]) - np.mean(
        [m[0] for m in rows["majority"]]
    )
    print(f"  logreg − majority accuracy = {delta:+.4f}")
    print()


def holdout_report(
    name: str,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
) -> None:
    print(f"=== {name}  (train n={len(y_train)}, test n={len(y_test)}) ===")
    for key, factory in (("majority", _dummy), ("logreg", _clf)):
        model = factory()
        model.fit(x_train, y_train)
        pred = model.predict(x_test)
        acc, f1 = _metrics(y_test, pred)
        print(f"  {key:<10}  acc={acc:.4f}   weighted-F1={f1:.4f}")
    logreg = _clf()
    logreg.fit(x_train, y_train)
    coef = logreg.named_steps["logreg"].coef_
    # coef shape (3, 5): one row per class. Mean absolute weight per feature
    # is a coarse “which ET column did the linear model use?” view.
    mean_abs = np.mean(np.abs(coef), axis=0)
    print("  mean |coef| per feature (logreg, after StandardScaler):")
    for col, weight in zip(SST_ET_COLS, mean_abs):
        print(f"    {col:<8} {weight:.4f}")
    print()


def main() -> int:
    zuco = pd.read_csv(ZUCO_STANDARD)
    x, y = _pack(zuco, ZUCO_ET_COLS)
    cv_report("ZuCo ∩ SST recorded gaze → sentiment", x, y)

    train = pd.read_csv(FULL_SST_TRAIN)
    test = pd.read_csv(FULL_SST_TEST)
    x_tr, y_tr = _pack(train, SST_ET_COLS)
    x_te, y_te = _pack(test, SST_ET_COLS)
    holdout_report("full SST projected gaze → sentiment", x_tr, y_tr, x_te, y_te)

    print(
        "If logreg cannot beat majority, the 5-d ET vector has little "
        "linear polarity signal on its own. Late fusion could still help "
        "if the encoder uses gaze as a residual, but that is a weaker "
        "prior than a gaze-only win here."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
