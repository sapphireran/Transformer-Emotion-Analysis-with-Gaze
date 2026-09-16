"""Tiny fusion stand-in so examples can show the concat idea without downloading BERT.

The real trainers in ``model_ZuCo_SST.py`` and ``model_full_SST.py`` load
``bert-base-uncased`` or ``roberta-base``, project 5 eye-tracking features to
16 dimensions, concatenate that vector with the 768-d pooler output, then
classify into 3 sentiment classes.

This module keeps the same concat geometry with hashed bag-of-words text
features so a laptop can run the example in a few seconds.
"""

from __future__ import annotations

from dataclasses import dataclass

import zlib

import numpy as np
from numpy.typing import ArrayLike
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

from .schema import HIDDEN_LAYER_SIZE, NUM_EYE_TRACKING_FEATURES, NUM_LABELS


DEFAULT_TEXT_DIM = 48


def hashed_bow_vector(text: str, dim: int = DEFAULT_TEXT_DIM) -> np.ndarray:
    """Very small hashed bag-of-words vector. Not a language model."""
    vec = np.zeros(dim, dtype=np.float64)
    tokens = [tok for tok in "".join(ch.lower() if ch.isalnum() else " " for ch in text).split() if tok]
    if not tokens:
        return vec
    for token in tokens:
        # crc32 is stable across processes; Python's hash() is not.
        digest = zlib.crc32(token.encode("utf-8"))
        index = int(digest % dim)
        sign = 1.0 if (digest // dim) % 2 == 0 else -1.0
        vec[index] += sign
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec /= norm
    return vec


def hashed_bow_matrix(texts: ArrayLike, dim: int = DEFAULT_TEXT_DIM) -> np.ndarray:
    return np.vstack([hashed_bow_vector(str(text), dim=dim) for text in texts])


@dataclass
class ToyFusionGeometry:
    """Numbers that mirror EyeTrackingModel without the transformer weights."""

    text_dim: int = DEFAULT_TEXT_DIM
    et_in: int = NUM_EYE_TRACKING_FEATURES
    et_hidden: int = HIDDEN_LAYER_SIZE
    num_labels: int = NUM_LABELS

    @property
    def concat_dim(self) -> int:
        return self.text_dim + self.et_hidden

    def describe(self) -> str:
        return (
            f"text {self.text_dim} + ET {self.et_in}->{self.et_hidden} "
            f"=> concat {self.concat_dim} => {self.num_labels} logits"
        )


class ToyEyeTrackingFusion:
    """Logistic regression on [hashed text | projected ET], for examples only.

    The linear ET projection is random-but-seeded, matching the idea of
    ``nn.Linear(5, 16)`` before concat. The classifier is then fit with
    scikit-learn so the example can report a real number without PyTorch.
    """

    def __init__(
        self,
        geometry: ToyFusionGeometry | None = None,
        seed: int = 7,
        C: float = 1.0,
    ) -> None:
        self.geometry = geometry or ToyFusionGeometry()
        self.seed = seed
        self.C = C
        rng = np.random.default_rng(seed)
        self.et_weight = rng.normal(
            0.0,
            0.3,
            size=(self.geometry.et_in, self.geometry.et_hidden),
        )
        self.et_bias = rng.normal(0.0, 0.05, size=(self.geometry.et_hidden,))
        self.classifier = LogisticRegression(
            C=C,
            max_iter=400,
            solver="lbfgs",
            random_state=seed,
        )
        self._label_encoder = LabelEncoder()
        self._fitted = False

    def project_eye_tracking(self, features: ArrayLike) -> np.ndarray:
        et = np.asarray(features, dtype=np.float64)
        if et.ndim == 1:
            et = et.reshape(1, -1)
        if et.shape[1] != self.geometry.et_in:
            raise ValueError(f"expected {self.geometry.et_in} ET features, got {et.shape[1]}")
        hidden = et @ self.et_weight + self.et_bias
        return np.tanh(hidden)

    def encode(self, texts: ArrayLike, eye_tracking: ArrayLike) -> np.ndarray:
        text_mat = hashed_bow_matrix(texts, dim=self.geometry.text_dim)
        et_hidden = self.project_eye_tracking(eye_tracking)
        return np.concatenate([text_mat, et_hidden], axis=1)

    def fit(self, texts: ArrayLike, eye_tracking: ArrayLike, labels: ArrayLike) -> "ToyEyeTrackingFusion":
        encoded = self.encode(texts, eye_tracking)
        y = self._label_encoder.fit_transform(np.asarray(labels))
        self.classifier.fit(encoded, y)
        self._fitted = True
        return self

    def predict(self, texts: ArrayLike, eye_tracking: ArrayLike) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("ToyEyeTrackingFusion.fit() must be called first")
        encoded = self.encode(texts, eye_tracking)
        pred = self.classifier.predict(encoded)
        return self._label_encoder.inverse_transform(pred)

    def predict_proba(self, texts: ArrayLike, eye_tracking: ArrayLike) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("ToyEyeTrackingFusion.fit() must be called first")
        return self.classifier.predict_proba(self.encode(texts, eye_tracking))
