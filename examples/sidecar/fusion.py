"""NumPy reconstruction of the gaze-sidecar fusion used in the training scripts.

Both ``model_full_SST.py`` and ``model_ZuCo_SST.py`` define:

    eye = Linear(5 -> 16)(gaze)          # no activation
    h = concat(pooler_output, eye)    # 768 + 16 = 784
    logits = Linear(784 -> 3)(Dropout(h))

This module does **not** load BERT/RoBERTa. It exists so the personal
docs can talk about the geometry with a tiny, deterministic forward pass
that unit tests can pin down.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

POOLER_DIM = 768
GAZE_IN_DIM = 5
GAZE_HIDDEN_DIM = 16
NUM_LABELS = 3
FUSION_CONCAT_DIM = POOLER_DIM + GAZE_HIDDEN_DIM  # 784


def _kaiming_uniform(rng: np.random.Generator, fan_in: int, fan_out: int) -> np.ndarray:
    bound = np.sqrt(6.0 / fan_in)
    return rng.uniform(-bound, bound, size=(fan_out, fan_in)).astype(np.float64)


@dataclass
class SidecarFusion:
    """Tiny stand-in for ``EyeTrackingModel.forward`` (linear sidecar, concat, dropout, classifier)."""

    W_eye: np.ndarray
    b_eye: np.ndarray
    W_cls: np.ndarray
    b_cls: np.ndarray
    dropout_p: float = 0.1
    seed: int = 0

    @classmethod
    def from_seed(
        cls,
        seed: int = 0,
        hidden: int = POOLER_DIM,
        gaze_in: int = GAZE_IN_DIM,
        gaze_h: int = GAZE_HIDDEN_DIM,
        n_labels: int = NUM_LABELS,
        dropout_p: float = 0.1,
    ) -> "SidecarFusion":
        rng = np.random.default_rng(seed)
        return cls(
            W_eye=_kaiming_uniform(rng, gaze_in, gaze_h),
            b_eye=np.zeros(gaze_h, dtype=np.float64),
            W_cls=_kaiming_uniform(rng, hidden + gaze_h, n_labels),
            b_cls=np.zeros(n_labels, dtype=np.float64),
            dropout_p=dropout_p,
            seed=seed,
        )

    @property
    def concat_dim(self) -> int:
        """Classifier input size: pooler dim + gaze hidden dim (784 in the training scripts)."""
        return int(self.W_cls.shape[1])

    def project_gaze(self, gaze: np.ndarray) -> np.ndarray:
        gaze = np.asarray(gaze, dtype=np.float64)
        if gaze.ndim == 1:
            gaze = gaze[None, :]
        if gaze.shape[-1] != self.W_eye.shape[1]:
            raise ValueError(f"expected gaze dim {self.W_eye.shape[1]}, got {gaze.shape[-1]}")
        return gaze @ self.W_eye.T + self.b_eye

    def concat(self, pooler: np.ndarray, gaze: np.ndarray) -> np.ndarray:
        pooler = np.asarray(pooler, dtype=np.float64)
        if pooler.ndim == 1:
            pooler = pooler[None, :]
        eye = self.project_gaze(gaze)
        if pooler.shape[0] != eye.shape[0]:
            raise ValueError("batch mismatch between pooler and gaze")
        return np.concatenate([pooler, eye], axis=1)

    def _dropout(self, h: np.ndarray, training: bool) -> np.ndarray:
        if (not training) or self.dropout_p <= 0.0:
            return h
        rng = np.random.default_rng(self.seed + 17)
        mask = rng.random(h.shape) >= self.dropout_p
        return h * mask / (1.0 - self.dropout_p)

    def logits(self, pooler: np.ndarray, gaze: np.ndarray, training: bool = False) -> np.ndarray:
        h = self.concat(pooler, gaze)
        if h.shape[1] != FUSION_CONCAT_DIM:
            # allow alternative hidden sizes in tests
            pass
        h = self._dropout(h, training=training)
        return h @ self.W_cls.T + self.b_cls

    def predict(self, pooler: np.ndarray, gaze: np.ndarray) -> np.ndarray:
        return np.argmax(self.logits(pooler, gaze, training=False), axis=1)


def softmax(logits: np.ndarray) -> np.ndarray:
    z = np.asarray(logits, dtype=np.float64)
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)
