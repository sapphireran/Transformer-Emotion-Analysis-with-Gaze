"""Cheap baselines that do not download transformers.

These are for personal sanity checks: if a linear model on gaze alone is already
close to the transformer, the fusion story needs more evidence than a single
run of ``model_ZuCo_SST.py``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .metrics import MetricBundle, classification_metrics, majority_baseline


@dataclass(frozen=True)
class FoldScore:
    fold: int
    metrics: MetricBundle
    n_train: int
    n_test: int


@dataclass(frozen=True)
class CvReport:
    name: str
    folds: list[FoldScore]

    @property
    def mean(self) -> MetricBundle:
        keys = ("accuracy", "precision", "recall", "f1")
        averages = {
            key: float(np.mean([getattr(fold.metrics, key) for fold in self.folds]))
            for key in keys
        }
        return MetricBundle(**averages)

    def pretty(self) -> str:
        lines = [f"{self.name} ({len(self.folds)} folds)"]
        for fold in self.folds:
            lines.append(f"  fold {fold.fold}: {fold.metrics.pretty()}  n={fold.n_train}/{fold.n_test}")
        lines.append(f"  mean: {self.mean.pretty()}")
        return "\n".join(lines)


def gaze_only_logistic_cv(
    frame: pd.DataFrame,
    feature_columns: list[str] | tuple[str, ...],
    label_column: str = "sentiment_label",
    n_splits: int = 5,
    seed: int = 42,
    C: float = 1.0,
) -> CvReport:
    """Standard-scaled multinomial logistic regression on gaze features only."""
    X = frame.loc[:, list(feature_columns)].to_numpy(dtype=np.float64)
    y = frame[label_column].to_numpy()
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    folds: list[FoldScore] = []
    for fold_id, (train_idx, test_idx) in enumerate(splitter.split(X, y), start=1):
        pipe = Pipeline(
            [
                ("scale", StandardScaler()),
                (
                    "clf",
                    LogisticRegression(
                        C=C,
                        max_iter=400,
                        solver="lbfgs",
                        random_state=seed,
                    ),
                ),
            ]
        )
        pipe.fit(X[train_idx], y[train_idx])
        preds = pipe.predict(X[test_idx])
        folds.append(
            FoldScore(
                fold=fold_id,
                metrics=classification_metrics(preds, y[test_idx]),
                n_train=int(len(train_idx)),
                n_test=int(len(test_idx)),
            )
        )
    return CvReport("gaze_only_logistic", folds)


def majority_cv(
    frame: pd.DataFrame,
    label_column: str = "sentiment_label",
    n_splits: int = 5,
    seed: int = 42,
) -> CvReport:
    y = frame[label_column].to_numpy()
    X = np.zeros((len(y), 1))
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    folds: list[FoldScore] = []
    for fold_id, (train_idx, test_idx) in enumerate(splitter.split(X, y), start=1):
        majority, _ = majority_baseline(y[train_idx])
        preds = np.full_like(y[test_idx], majority)
        metrics = classification_metrics(preds, y[test_idx])
        folds.append(
            FoldScore(
                fold=fold_id,
                metrics=metrics,
                n_train=int(len(train_idx)),
                n_test=int(len(test_idx)),
            )
        )
    return CvReport("majority", folds)
