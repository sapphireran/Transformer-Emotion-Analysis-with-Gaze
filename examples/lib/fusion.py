"""GPU-free stand-in for the late-fusion idea in model_ZuCo_SST.py.

The historical trainers concatenate a 768-d encoder pooler with a 16-d
linear map of five gaze features. This module keeps the *same join*:
text features and gaze features live in separate blocks and meet only
at the classifier. Text is a hashed bag-of-words so the example runs
without transformers or a GPU.

The linear gaze map here is implicit — logistic regression learns a
weight per hashed n-gram *and* per gaze column. That is enough to ask
“do these five z-scores move a linear text model on 400 rows?”
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from .loaders import FUSION_GAZE_COLUMNS, fusion_frame
from .metrics import MetricBundle, majority_baseline, mean_std, score_predictions

HASH_FEATURES = 2**12


def _as_series(X):
    if isinstance(X, pd.DataFrame):
        if X.shape[1] != 1:
            raise ValueError("expected a single text column")
        return X.iloc[:, 0].astype(str)
    if isinstance(X, pd.Series):
        return X.astype(str)
    return pd.Series(np.asarray(X).reshape(-1), dtype=str)


def hashed_text_matrix(sentences: pd.Series | list[str], *, n_features: int = HASH_FEATURES):
    """Hash character/word n-grams. Deterministic, no vocabulary file."""
    vec = HashingVectorizer(
        n_features=n_features,
        alternate_sign=False,
        ngram_range=(1, 2),
        lowercase=True,
        norm="l2",
    )
    return vec.transform(_as_series(sentences))


def _text_column(X):
    return _as_series(X["sentence"])


def _gaze_columns(X):
    return fusion_frame(X).to_numpy(dtype=float)


def _length_column(X):
    lengths = _as_series(X["sentence"]).str.split().map(len).to_numpy(dtype=float)
    return lengths.reshape(-1, 1)


def _build_pipeline(kind: str, random_state: int) -> Pipeline:
    clf = LogisticRegression(
        max_iter=400,
        solver="lbfgs",
        random_state=random_state,
    )
    text = Pipeline(
        steps=[
            ("pick", FunctionTransformer(_text_column, validate=False)),
            (
                "hash",
                HashingVectorizer(
                    n_features=HASH_FEATURES,
                    alternate_sign=False,
                    ngram_range=(1, 2),
                    lowercase=True,
                    norm="l2",
                ),
            ),
        ]
    )
    gaze = Pipeline(
        steps=[
            ("pick", FunctionTransformer(_gaze_columns, validate=False)),
            ("scale", StandardScaler(with_mean=True, with_std=True)),
        ]
    )
    length = Pipeline(
        steps=[
            ("pick", FunctionTransformer(_length_column, validate=False)),
            ("scale", StandardScaler()),
        ]
    )

    if kind == "gaze":
        features = FeatureUnion([("gaze", gaze)])
    elif kind == "length":
        features = FeatureUnion([("length", length)])
    elif kind == "text":
        features = FeatureUnion([("text", text)])
    elif kind == "fusion":
        features = FeatureUnion([("text", text), ("gaze", gaze)])
    else:
        raise ValueError(f"unknown fusion kind {kind!r}")

    return Pipeline([("features", features), ("clf", clf)])


@dataclass
class FoldResult:
    kind: str
    fold: int
    metrics: MetricBundle


@dataclass
class KindResult:
    kind: str
    folds: list[FoldResult] = field(default_factory=list)

    def mean_bundle_line(self) -> str:
        acc_m, acc_s = mean_std((f.metrics for f in self.folds), "accuracy")
        f1w_m, f1w_s = mean_std((f.metrics for f in self.folds), "f1_weighted")
        f1m_m, f1m_s = mean_std((f.metrics for f in self.folds), "f1_macro")
        return (
            f"{self.kind:<10} acc={acc_m:.4f}±{acc_s:.4f}  "
            f"F1_w={f1w_m:.4f}±{f1w_s:.4f}  "
            f"F1_macro={f1m_m:.4f}±{f1m_s:.4f}  "
            f"folds={len(self.folds)}"
        )


def run_fusion_cv(
    df: pd.DataFrame,
    *,
    kinds: tuple[str, ...] = ("length", "gaze", "text", "fusion"),
    n_splits: int = 5,
    random_state: int = 42,
) -> dict[str, KindResult]:
    """Stratified CV for the cheap late-fusion stand-in."""
    if "sentence" not in df.columns or "sentiment_label" not in df.columns:
        raise KeyError("df must have sentence and sentiment_label")

    y = df["sentiment_label"].astype(int).to_numpy()
    kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    out: dict[str, KindResult] = {kind: KindResult(kind=kind) for kind in kinds}

    for fold, (train_idx, test_idx) in enumerate(kf.split(df, y), start=1):
        train_df = df.iloc[train_idx].reset_index(drop=True)
        test_df = df.iloc[test_idx].reset_index(drop=True)
        y_train = y[train_idx]
        y_test = y[test_idx]
        for kind in kinds:
            pipe = _build_pipeline(kind, random_state=random_state + fold)
            pipe.fit(train_df, y_train)
            pred = pipe.predict(test_df)
            metrics = score_predictions(y_test, pred)
            out[kind].folds.append(FoldResult(kind=kind, fold=fold, metrics=metrics))

    out["majority"] = KindResult(kind="majority")
    # Majority is global (same prior each fold from the *fold train set*
    # would be slightly more honest; use the fold test vs train majority).
    kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    for fold, (train_idx, test_idx) in enumerate(kf.split(df, y), start=1):
        y_train = y[train_idx]
        y_test = y[test_idx]
        values, counts = np.unique(y_train, return_counts=True)
        majority = int(values[int(np.argmax(counts))])
        pred = np.full_like(y_test, majority)
        out["majority"].folds.append(
            FoldResult(kind="majority", fold=fold, metrics=score_predictions(y_test, pred))
        )

    # Keep a full-set majority too for the printed prior.
    out["majority_full"] = KindResult(kind="majority_full")
    full = majority_baseline(y)
    out["majority_full"].folds.append(FoldResult(kind="majority_full", fold=0, metrics=full))
    return out
