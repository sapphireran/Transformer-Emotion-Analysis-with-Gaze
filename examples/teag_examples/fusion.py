"""NumPy clone of EyeTrackingModel's gaze concat (no transformer weights).

Matches ``model_full_SST.py`` / ``model_ZuCo_SST.py``:

* ``Linear(n_gaze, gaze_hidden)`` with **no** activation
* concat with pooled text vector
* dropout on the concat (inverted, train-time only)
* ``Linear(hidden + gaze_hidden, n_labels)``
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class FusionConfig:
    hidden_size: int = 768
    n_gaze_features: int = 5
    gaze_hidden: int = 16
    n_labels: int = 3
    dropout: float = 0.1


def _xavier_uniform(rng: np.random.Generator, fan_in: int, fan_out: int) -> np.ndarray:
    limit = np.sqrt(6.0 / (fan_in + fan_out))
    return rng.uniform(-limit, limit, size=(fan_in, fan_out))


def softmax(logits: np.ndarray, axis: int = -1) -> np.ndarray:
    shifted = logits - np.max(logits, axis=axis, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=axis, keepdims=True)


def cross_entropy(logits: np.ndarray, labels: np.ndarray) -> float:
    """Mean NLL of integer labels; numerically stable via log-softmax."""
    labels = np.asarray(labels)
    log_probs = logits - np.logaddexp.reduce(logits, axis=-1, keepdims=True)
    n = labels.shape[0]
    return float(-np.mean(log_probs[np.arange(n), labels]))


class NumpyEyeTrackingFusion:
    def __init__(
        self,
        config: FusionConfig | None = None,
        rng: np.random.Generator | None = None,
        seed: int = 0,
    ) -> None:
        self.config = config or FusionConfig()
        self.rng = rng or np.random.default_rng(seed)
        c = self.config
        self.W_gaze = _xavier_uniform(self.rng, c.n_gaze_features, c.gaze_hidden)
        self.b_gaze = np.zeros(c.gaze_hidden, dtype=np.float64)
        self.W_cls = _xavier_uniform(
            self.rng, c.hidden_size + c.gaze_hidden, c.n_labels
        )
        self.b_cls = np.zeros(c.n_labels, dtype=np.float64)

    @property
    def concat_width(self) -> int:
        return self.config.hidden_size + self.config.gaze_hidden

    def gaze_hidden(self, gaze: np.ndarray) -> np.ndarray:
        gaze = np.asarray(gaze, dtype=np.float64)
        if gaze.ndim == 1:
            gaze = gaze[None, :]
        if gaze.shape[-1] != self.config.n_gaze_features:
            raise ValueError(
                f"expected {self.config.n_gaze_features} gaze dims, got {gaze.shape}"
            )
        return gaze @ self.W_gaze + self.b_gaze

    def forward(
        self,
        pooled: np.ndarray,
        gaze: np.ndarray,
        *,
        train: bool = False,
        dropout_rng: np.random.Generator | None = None,
    ) -> np.ndarray:
        pooled = np.asarray(pooled, dtype=np.float64)
        if pooled.ndim == 1:
            pooled = pooled[None, :]
        if pooled.shape[-1] != self.config.hidden_size:
            raise ValueError(
                f"expected pooled width {self.config.hidden_size}, got {pooled.shape}"
            )
        gaze_h = self.gaze_hidden(gaze)
        if pooled.shape[0] != gaze_h.shape[0]:
            raise ValueError("batch size mismatch between pooled and gaze")
        combined = np.concatenate([pooled, gaze_h], axis=1)
        if train and self.config.dropout > 0:
            rng = dropout_rng or self.rng
            keep = 1.0 - self.config.dropout
            mask = rng.random(combined.shape) >= self.config.dropout
            combined = combined * mask / keep
        return combined @ self.W_cls + self.b_cls

    def predict_proba(self, pooled: np.ndarray, gaze: np.ndarray) -> np.ndarray:
        return softmax(self.forward(pooled, gaze, train=False))

    def predict(self, pooled: np.ndarray, gaze: np.ndarray) -> np.ndarray:
        return np.argmax(self.predict_proba(pooled, gaze), axis=-1)
