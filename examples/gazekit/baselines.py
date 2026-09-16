"""Cheap sklearn baselines that do not download transformer weights."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .io import gaze_matrix
from .metrics import MetricSet, calculate_metrics, mean_and_std
from .schema import CANONICAL_GAZE


@dataclass
class FoldResult:
    metrics: MetricSet
    name: str
    fold: int


@dataclass
class BaselineReport:
    name: str
    feature_names: tuple[str, ...]
    folds: list[FoldResult]
    mean_accuracy: float
    std_accuracy: float
    mean_f1_weighted: float
    std_f1_weighted: float
    mean_f1_macro: float
    std_f1_macro: float

    def as_dict(self) -> dict:
        return {
            "name": self.name,
            "feature_names": list(self.feature_names),
            "n_folds": len(self.folds),
            "mean_accuracy": self.mean_accuracy,
            "std_accuracy": self.std_accuracy,
            "mean_f1_weighted": self.mean_f1_weighted,
            "std_f1_weighted": self.std_f1_weighted,
            "mean_f1_macro": self.mean_f1_macro,
            "std_f1_macro": self.std_f1_macro,
        }


def _xy(df: pd.DataFrame, columns: Sequence[str], label_col: str = "sentiment_label"):
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise KeyError(f"baseline missing columns {missing}")
    X = df.loc[:, list(columns)].to_numpy(dtype="float64")
    y = df[label_col].to_numpy()
    if np.isnan(X).any():
        raise ValueError("baseline features contain NaN; fill or drop first")
    return X, y


def _summarize(name: str, feature_names: Sequence[str], folds: list[FoldResult]) -> BaselineReport:
    acc = [f.metrics.accuracy for f in folds]
    f1w = [f.metrics.f1_weighted for f in folds]
    f1m = [f.metrics.f1_macro for f in folds]
    ma, sa = mean_and_std(acc)
    mw, sw = mean_and_std(f1w)
    mm, sm = mean_and_std(f1m)
    return BaselineReport(
        name=name,
        feature_names=tuple(feature_names),
        folds=folds,
        mean_accuracy=ma,
        std_accuracy=sa,
        mean_f1_weighted=mw,
        std_f1_weighted=sw,
        mean_f1_macro=mm,
        std_f1_macro=sm,
    )


def run_cv(
    df: pd.DataFrame,
    columns: Sequence[str],
    name: str,
    estimator,
    n_splits: int = 5,
    seed: int = 42,
    label_col: str = "sentiment_label",
) -> BaselineReport:
    X, y = _xy(df, columns, label_col)
    kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    folds: list[FoldResult] = []
    for i, (train_idx, test_idx) in enumerate(kf.split(X, y)):
        est = estimator
        # Fresh clone via sklearn if available; otherwise refit is unsafe.
        from sklearn.base import clone

        model = clone(est)
        model.fit(X[train_idx], y[train_idx])
        pred = model.predict(X[test_idx])
        folds.append(FoldResult(metrics=calculate_metrics(pred, y[test_idx]), name=name, fold=i))
    return _summarize(name, columns, folds)


def logistic_pipeline(seed: int = 42) -> Pipeline:
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    max_iter=500,
                    multi_class="auto",
                    solver="lbfgs",
                    random_state=seed,
                ),
            ),
        ]
    )


def fit_majority(
    df: pd.DataFrame,
    n_splits: int = 5,
    seed: int = 42,
    label_col: str = "sentiment_label",
) -> BaselineReport:
    dummy = DummyClassifier(strategy="most_frequent")
    # Dummy ignores X; give it a single dummy column so the CV helper works.
    tmp = df.copy()
    tmp["_ones"] = 1.0
    return run_cv(tmp, ["_ones"], "majority", dummy, n_splits=n_splits, seed=seed, label_col=label_col)


def fit_logistic(
    df: pd.DataFrame,
    columns: Sequence[str],
    name: str,
    n_splits: int = 5,
    seed: int = 42,
    label_col: str = "sentiment_label",
) -> BaselineReport:
    return run_cv(
        df,
        columns,
        name,
        logistic_pipeline(seed),
        n_splits=n_splits,
        seed=seed,
        label_col=label_col,
    )


def fit_gaze_only(df: pd.DataFrame, **kwargs) -> BaselineReport:
    return fit_logistic(df, CANONICAL_GAZE, "gaze_only_logistic", **kwargs)


def fit_length_only(df: pd.DataFrame, length_col: str = "n_tokens", **kwargs) -> BaselineReport:
    return fit_logistic(df, [length_col], "length_only_logistic", **kwargs)


def fit_gaze_plus_length(
    df: pd.DataFrame,
    length_col: str = "n_tokens",
    extra: Iterable[str] = (),
    **kwargs,
) -> BaselineReport:
    cols = list(CANONICAL_GAZE) + [length_col] + list(extra)
    return fit_logistic(df, cols, "gaze_plus_length_logistic", **kwargs)


def residualize(df: pd.DataFrame, columns: Sequence[str], on: str) -> pd.DataFrame:
    """Replace each column with the residual after a univariate linear fit on ``on``.

    Used to ask: does gaze still predict sentiment after sentence length is
    partialled out? This is an ordinary least-squares residual, not a
    multivariate mixed model.
    """
    if on not in df.columns:
        raise KeyError(on)
    out = df.copy()
    x = out[on].to_numpy(dtype="float64")
    x = np.column_stack([np.ones(len(x)), x])
    xtx_inv = np.linalg.pinv(x.T @ x)
    for col in columns:
        y = out[col].to_numpy(dtype="float64")
        beta = xtx_inv @ x.T @ y
        out[col] = y - x @ beta
    return out


def gaze_matrix_or_raise(df: pd.DataFrame):
    return gaze_matrix(df)
