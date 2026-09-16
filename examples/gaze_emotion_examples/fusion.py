"""A numpy sketch of the BERT/RoBERTa + gaze fusion used in the training scripts.

``model_ZuCo_SST.py`` and ``model_full_SST.py`` both do:

1. Encode the sentence with a transformer and take ``pooler_output`` (768-d).
2. Project the gaze vector with ``Linear(gaze_dim, 16)``.
3. Concatenate, apply dropout, and classify with ``Linear(768 + 16, 3)``.

This module reproduces the linear parts in numpy so examples can run without
torch or Hugging Face weights. It also fits cheap sklearn baselines so the
docs can quote text-only vs text+gaze numbers on the 400 ZuCo sentences.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin


def softmax(logits: np.ndarray) -> np.ndarray:
    """Row-wise softmax, stable for large logits."""
    shifted = logits - logits.max(axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=-1, keepdims=True)


class GazeFusionForward:
    """Random-initialized linear stand-in for ``EyeTrackingModel``.

    Weights are not trained. The class exists to show tensor shapes and to
    let tests lock the concat / classify math.
    """

    def __init__(
        self,
        text_dim: int = 768,
        gaze_dim: int = 5,
        hidden: int = 16,
        num_labels: int = 3,
        seed: int = 0,
    ) -> None:
        rng = np.random.default_rng(seed)
        self.text_dim = text_dim
        self.gaze_dim = gaze_dim
        self.hidden = hidden
        self.num_labels = num_labels
        scale_g = 1.0 / np.sqrt(gaze_dim)
        scale_c = 1.0 / np.sqrt(text_dim + hidden)
        self.W_gaze = rng.normal(0.0, scale_g, size=(gaze_dim, hidden))
        self.b_gaze = np.zeros(hidden)
        self.W_cls = rng.normal(0.0, scale_c, size=(text_dim + hidden, num_labels))
        self.b_cls = np.zeros(num_labels)

    def project_gaze(self, gaze: np.ndarray) -> np.ndarray:
        gaze = np.asarray(gaze, dtype=float)
        if gaze.ndim == 1:
            gaze = gaze[None, :]
        if gaze.shape[-1] != self.gaze_dim:
            raise ValueError(f"expected gaze dim {self.gaze_dim}, got {gaze.shape[-1]}")
        return gaze @ self.W_gaze + self.b_gaze

    def forward(self, text: np.ndarray, gaze: np.ndarray) -> np.ndarray:
        text = np.asarray(text, dtype=float)
        if text.ndim == 1:
            text = text[None, :]
        if text.shape[-1] != self.text_dim:
            raise ValueError(f"expected text dim {self.text_dim}, got {text.shape[-1]}")
        gaze_hidden = self.project_gaze(gaze)
        if text.shape[0] != gaze_hidden.shape[0]:
            raise ValueError("batch size mismatch between text and gaze")
        combined = np.concatenate([text, gaze_hidden], axis=1)
        return combined @ self.W_cls + self.b_cls

    def predict_proba(self, text: np.ndarray, gaze: np.ndarray) -> np.ndarray:
        return softmax(self.forward(text, gaze))

    def predict(self, text: np.ndarray, gaze: np.ndarray) -> np.ndarray:
        return self.predict_proba(text, gaze).argmax(axis=1)


class _GazeBlock(BaseEstimator, TransformerMixin):
    """sklearn transformer that keeps only the numeric gaze columns."""

    def __init__(self, columns: list[str]) -> None:
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if isinstance(X, pd.DataFrame):
            return X.loc[:, self.columns].to_numpy(dtype=float)
        raise TypeError("GazeBlock expects a pandas DataFrame")


class _TextBlock(BaseEstimator, TransformerMixin):
    """sklearn transformer that keeps the sentence string column."""

    def __init__(self, column: str = "sentence") -> None:
        self.column = column

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if isinstance(X, pd.DataFrame):
            return X[self.column].astype(str).to_numpy()
        raise TypeError("TextBlock expects a pandas DataFrame")


@dataclass(frozen=True)
class FoldScore:
    fold: int
    accuracy: float
    macro_f1: float
    weighted_f1: float


@dataclass(frozen=True)
class BaselineResult:
    name: str
    folds: tuple[FoldScore, ...]

    @property
    def mean_accuracy(self) -> float:
        return float(np.mean([fold.accuracy for fold in self.folds]))

    @property
    def mean_macro_f1(self) -> float:
        return float(np.mean([fold.macro_f1 for fold in self.folds]))

    @property
    def mean_weighted_f1(self) -> float:
        return float(np.mean([fold.weighted_f1 for fold in self.folds]))

    def to_frame(self) -> pd.DataFrame:
        rows = [
            {
                "model": self.name,
                "fold": fold.fold,
                "accuracy": fold.accuracy,
                "macro_f1": fold.macro_f1,
                "weighted_f1": fold.weighted_f1,
            }
            for fold in self.folds
        ]
        rows.append(
            {
                "model": self.name,
                "fold": "mean",
                "accuracy": self.mean_accuracy,
                "macro_f1": self.mean_macro_f1,
                "weighted_f1": self.mean_weighted_f1,
            }
        )
        return pd.DataFrame(rows)


def _gaze_pipeline(columns: list[str]) -> Pipeline:
    return Pipeline(
        [
            ("select", _GazeBlock(columns)),
            ("scale", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    max_iter=400,
                    class_weight="balanced",
                    solver="lbfgs",
                ),
            ),
        ]
    )


def _text_pipeline(column: str = "sentence") -> Pipeline:
    return Pipeline(
        [
            ("select", _TextBlock(column)),
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    ngram_range=(1, 2),
                    min_df=2,
                    max_features=4000,
                ),
            ),
            (
                "clf",
                LogisticRegression(
                    max_iter=400,
                    class_weight="balanced",
                    solver="lbfgs",
                ),
            ),
        ]
    )


def _fused_pipeline(text_column: str, gaze_columns: list[str]) -> Pipeline:
    features = FeatureUnion(
        [
            (
                "text",
                Pipeline(
                    [
                        ("select", _TextBlock(text_column)),
                        (
                            "tfidf",
                            TfidfVectorizer(
                                lowercase=True,
                                ngram_range=(1, 2),
                                min_df=2,
                                max_features=4000,
                            ),
                        ),
                    ]
                ),
            ),
            (
                "gaze",
                Pipeline(
                    [
                        ("select", _GazeBlock(gaze_columns)),
                        ("scale", StandardScaler()),
                    ]
                ),
            ),
        ]
    )
    return Pipeline(
        [
            ("features", features),
            (
                "clf",
                LogisticRegression(
                    max_iter=400,
                    class_weight="balanced",
                    solver="lbfgs",
                ),
            ),
        ]
    )


def _evaluate(
    name: str,
    pipeline: Pipeline,
    frame: pd.DataFrame,
    labels: np.ndarray,
    folds: StratifiedKFold,
) -> BaselineResult:
    scores = []
    for index, (train_idx, test_idx) in enumerate(folds.split(frame, labels), start=1):
        model = pipeline
        model.fit(frame.iloc[train_idx], labels[train_idx])
        pred = model.predict(frame.iloc[test_idx])
        scores.append(
            FoldScore(
                fold=index,
                accuracy=float(accuracy_score(labels[test_idx], pred)),
                macro_f1=float(f1_score(labels[test_idx], pred, average="macro", zero_division=0)),
                weighted_f1=float(
                    f1_score(labels[test_idx], pred, average="weighted", zero_division=0)
                ),
            )
        )
    return BaselineResult(name=name, folds=tuple(scores))


def text_vs_gaze_cv(
    frame: pd.DataFrame,
    text_column: str,
    label_column: str,
    gaze_columns: list[str],
    n_splits: int = 5,
    seed: int = 42,
) -> list[BaselineResult]:
    """Compare gaze-only, text-only, and concatenated TF-IDF + gaze classifiers."""
    labels = frame[label_column].to_numpy()
    folds = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return [
        _evaluate("gaze_only", _gaze_pipeline(gaze_columns), frame, labels, folds),
        _evaluate("text_only", _text_pipeline(text_column), frame, labels, folds),
        _evaluate(
            "text_plus_gaze",
            _fused_pipeline(text_column, gaze_columns),
            frame,
            labels,
            folds,
        ),
    ]
