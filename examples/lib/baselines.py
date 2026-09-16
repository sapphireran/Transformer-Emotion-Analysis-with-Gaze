"""Sklearn baselines that never touch a transformer.

Used to answer two personal questions before any GPU run:

1. Do the five gaze numbers predict polarity *by themselves*?
2. Does shuffling gaze vs label kill that signal?
"""

from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .gaze import feature_label_anova
from .metrics import majority_baseline, weighted_classification_scores

DEFAULT_MAX_ITER = 500


def gaze_logreg_pipeline(seed: int = 0) -> Pipeline:
    return Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    max_iter=DEFAULT_MAX_ITER,
                    solver="lbfgs",
                    random_state=seed,
                ),
            ),
        ]
    )


def stratified_cv_scores(
    features: pd.DataFrame | np.ndarray,
    labels: Iterable[int],
    n_splits: int = 5,
    seed: int = 0,
) -> dict[str, float]:
    """Out-of-fold logistic regression scores (accuracy + weighted F1)."""
    x = np.asarray(features, dtype=float)
    y = np.asarray(list(labels))
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    pipeline = gaze_logreg_pipeline(seed=seed)
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    preds = cross_val_predict(pipeline, x, y, cv=cv)
    scores = weighted_classification_scores(y, preds)
    majority = majority_baseline(y)
    scores["majority_accuracy"] = majority["accuracy"]
    scores["majority_f1_weighted"] = majority["f1_weighted"]
    scores["accuracy_minus_majority"] = scores["accuracy"] - majority["accuracy"]
    scores["n"] = int(len(y))
    scores["n_splits"] = n_splits
    return scores


def holdout_scores(
    train_x: pd.DataFrame | np.ndarray,
    train_y: Iterable[int],
    test_x: pd.DataFrame | np.ndarray,
    test_y: Iterable[int],
    seed: int = 0,
) -> dict[str, float]:
    """Fit on one split, score another (full-SST protocol)."""
    x_train = np.asarray(train_x, dtype=float)
    y_train = np.asarray(list(train_y))
    x_test = np.asarray(test_x, dtype=float)
    y_test = np.asarray(list(test_y))
    pipeline = gaze_logreg_pipeline(seed=seed)
    pipeline.fit(x_train, y_train)
    preds = pipeline.predict(x_test)
    scores = weighted_classification_scores(y_test, preds)
    majority = majority_baseline(y_test)
    scores["majority_accuracy"] = majority["accuracy"]
    scores["accuracy_minus_majority"] = scores["accuracy"] - majority["accuracy"]
    scores["n_train"] = int(len(y_train))
    scores["n_test"] = int(len(y_test))
    return scores


def permute_anova(
    df: pd.DataFrame,
    feature_columns: Sequence[str],
    label_column: str = "sentiment_label",
    n_perm: int = 100,
    seed: int = 0,
) -> pd.DataFrame:
    """Observed ANOVA F vs the distribution under shuffled labels."""
    rng = np.random.default_rng(seed)
    observed = feature_label_anova(df, feature_columns, label_column=label_column)
    labels = df[label_column].to_numpy().copy()
    perm_f = {feature: [] for feature in feature_columns}
    for _ in range(n_perm):
        shuffled = df.copy()
        shuffled[label_column] = rng.permutation(labels)
        table = feature_label_anova(shuffled, feature_columns, label_column=label_column)
        for _, row in table.iterrows():
            perm_f[row["feature"]].append(float(row["f_statistic"]))
    rows = []
    for _, row in observed.iterrows():
        feature = row["feature"]
        null = np.asarray(perm_f[feature], dtype=float)
        rows.append(
            {
                "feature": feature,
                "observed_f": float(row["f_statistic"]),
                "observed_p": float(row["p_value"]),
                "perm_mean_f": float(null.mean()),
                "perm_p": float((null >= row["f_statistic"]).mean()),
                "n_perm": n_perm,
            }
        )
    return pd.DataFrame(rows).sort_values("observed_f", ascending=False, ignore_index=True)


def residualize_on_length(
    df: pd.DataFrame,
    feature_columns: Sequence[str],
    length: pd.Series,
) -> pd.DataFrame:
    """Replace each feature with residuals after a linear fit on length."""
    x = np.asarray(length, dtype=float)
    out = df.copy()
    for col in feature_columns:
        y = np.asarray(df[col], dtype=float)
        if not np.isfinite(x).any() or np.unique(x).size < 2:
            out[col] = y
            continue
        slope, intercept = np.polyfit(x, y, 1)
        out[col] = y - (slope * x + intercept)
    return out
