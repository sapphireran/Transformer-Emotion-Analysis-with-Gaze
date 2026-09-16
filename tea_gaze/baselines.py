"""Sklearn baselines that mirror the text / gaze / fusion setup.

These are intentionally smaller than the BERT/RoBERTa scripts. They exist so
the personal examples can run on CPU in a few seconds and still compare:

* text only (TF-IDF + logistic regression)
* gaze only (the five fusion features)
* early fusion (sparse TF-IDF stacked with the five gaze columns)

The original transformer training loop is unchanged in `model_ZuCo_SST.py`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence

from tea_gaze.eval import MetricSet, mean_metrics, weighted_metrics
from tea_gaze.features import CORE_FUSION_FEATURES, fusion_feature_frame
from tea_gaze.text import tokenize

ModelName = Literal["text", "gaze", "fusion"]


@dataclass(frozen=True)
class FoldResult:
    fold: int
    metrics: MetricSet
    y_true: list[int]
    y_pred: list[int]


@dataclass(frozen=True)
class ExperimentResult:
    name: ModelName
    folds: tuple[FoldResult, ...]
    mean: MetricSet

    def summary_row(self) -> str:
        return (
            f"| `{self.name}` | {self.mean.accuracy:.4f} | {self.mean.precision:.4f} "
            f"| {self.mean.recall:.4f} | {self.mean.f1:.4f} |"
        )


def _require_sklearn():
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import StratifiedKFold
        from sklearn.preprocessing import StandardScaler
        from scipy.sparse import csr_matrix, hstack
    except ImportError as exc:  # pragma: no cover - exercised when extras missing
        raise ImportError(
            "tea_gaze.baselines needs scikit-learn and scipy. "
            "Install the example extras from requirements.txt."
        ) from exc
    return (
        TfidfVectorizer,
        LogisticRegression,
        StratifiedKFold,
        StandardScaler,
        csr_matrix,
        hstack,
    )


def _classifier():
    _, LogisticRegression, *_ = _require_sklearn()
    return LogisticRegression(
        max_iter=400,
        class_weight="balanced",
        solver="lbfgs",
    )


def _vectorizer():
    TfidfVectorizer, *_ = _require_sklearn()
    return TfidfVectorizer(
        tokenizer=tokenize,
        token_pattern=None,
        lowercase=False,
        min_df=2,
        max_features=4000,
        ngram_range=(1, 2),
    )


def prepare_xy(frame):
    """Return `(texts, gaze_array, labels)` from a sentence-level table."""
    import numpy as np

    if "sentence" not in frame.columns or "sentiment_label" not in frame.columns:
        raise KeyError("sentence-level frame must include sentence and sentiment_label")
    texts = frame["sentence"].astype(str).tolist()
    gaze = fusion_feature_frame(frame).to_numpy(dtype=float)
    labels = frame["sentiment_label"].astype(int).to_numpy()
    if np.isnan(gaze).any():
        raise ValueError("fusion gaze features contain NaN values")
    return texts, gaze, labels


def _fit_predict(
    name: ModelName,
    texts_train: list[str],
    texts_test: list[str],
    gaze_train,
    gaze_test,
    y_train,
):
    TfidfVectorizer, LogisticRegression, StratifiedKFold, StandardScaler, csr_matrix, hstack = (
        _require_sklearn()
    )
    del TfidfVectorizer, LogisticRegression, StratifiedKFold

    if name == "text":
        vectorizer = _vectorizer()
        x_train = vectorizer.fit_transform(texts_train)
        x_test = vectorizer.transform(texts_test)
    elif name == "gaze":
        scaler = StandardScaler()
        x_train = scaler.fit_transform(gaze_train)
        x_test = scaler.transform(gaze_test)
    elif name == "fusion":
        vectorizer = _vectorizer()
        scaler = StandardScaler()
        text_train = vectorizer.fit_transform(texts_train)
        text_test = vectorizer.transform(texts_test)
        gaze_train_s = scaler.fit_transform(gaze_train)
        gaze_test_s = scaler.transform(gaze_test)
        x_train = hstack([text_train, csr_matrix(gaze_train_s)])
        x_test = hstack([text_test, csr_matrix(gaze_test_s)])
    else:
        raise ValueError(f"unknown model {name!r}")

    model = _classifier()
    model.fit(x_train, y_train)
    return model.predict(x_test)


def cross_validate(
    frame,
    *,
    name: ModelName,
    n_splits: int = 5,
    random_state: int = 42,
) -> ExperimentResult:
    """Stratified 5-fold CV, same split policy as `model_ZuCo_SST.py`."""
    _, _, StratifiedKFold, *_ = _require_sklearn()
    texts, gaze, labels = prepare_xy(frame)
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    folds: list[FoldResult] = []
    for fold, (train_idx, test_idx) in enumerate(splitter.split(texts, labels), start=1):
        texts_train = [texts[i] for i in train_idx]
        texts_test = [texts[i] for i in test_idx]
        y_true = [int(labels[i]) for i in test_idx]
        y_pred = [
            int(v)
            for v in _fit_predict(
                name,
                texts_train,
                texts_test,
                gaze[train_idx],
                gaze[test_idx],
                labels[train_idx],
            )
        ]
        folds.append(
            FoldResult(
                fold=fold,
                metrics=weighted_metrics(y_true, y_pred),
                y_true=y_true,
                y_pred=y_pred,
            )
        )
    return ExperimentResult(
        name=name,
        folds=tuple(folds),
        mean=mean_metrics([fold.metrics for fold in folds]),
    )


def run_suite(
    frame,
    *,
    models: Sequence[ModelName] = ("text", "gaze", "fusion"),
    n_splits: int = 5,
    random_state: int = 42,
) -> list[ExperimentResult]:
    return [
        cross_validate(frame, name=name, n_splits=n_splits, random_state=random_state)
        for name in models
    ]


def gaze_only_coefficients(frame):
    """Fit one gaze-only model on all rows and return per-feature weights."""
    import numpy as np

    _, gaze, labels = prepare_xy(frame)
    _, LogisticRegression, _, StandardScaler, *_ = _require_sklearn()
    scaler = StandardScaler()
    x = scaler.fit_transform(gaze)
    model = LogisticRegression(
        max_iter=400,
        class_weight="balanced",
        solver="lbfgs",
    )
    model.fit(x, labels)
    # shape (n_classes, n_features)
    return {
        "features": list(CORE_FUSION_FEATURES),
        "classes": [int(c) for c in model.classes_],
        "coef": np.asarray(model.coef_).tolist(),
        "intercept": np.asarray(model.intercept_).tolist(),
    }
