"""NumPy stand-in for the late-fusion classifier in the training scripts.

The original ``EyeTrackingModel`` does:

1. Run BERT or RoBERTa and take ``pooler_output`` (hidden size 768).
2. Project 5 gaze features through a linear layer to ``hidden_layer_size`` (16).
3. Concatenate the two vectors, apply dropout, and classify into 3 labels.

This module keeps the same shapes and math so examples can walk through a
forward pass and a tiny SGD loop without downloading transformers weights.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .constants import (
    DEFAULT_HIDDEN_LAYER_SIZE,
    DEFAULT_NUM_EYE_TRACKING_FEATURES,
    DEFAULT_NUM_LABELS,
)


def softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits, axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=-1, keepdims=True)


def cross_entropy(logits: np.ndarray, labels: np.ndarray) -> float:
    probs = softmax(logits)
    gathered = probs[np.arange(len(labels)), labels]
    return float(-np.mean(np.log(np.clip(gathered, 1e-12, 1.0))))


def one_hot(labels: np.ndarray, num_labels: int) -> np.ndarray:
    encoded = np.zeros((labels.size, num_labels), dtype=float)
    encoded[np.arange(labels.size), labels] = 1.0
    return encoded


@dataclass
class FusionShapes:
    text_dim: int
    gaze_in: int
    gaze_hidden: int
    combined_dim: int
    num_labels: int

    def as_lines(self) -> list[str]:
        return [
            f"text pooled vector:          {self.text_dim}",
            f"raw gaze features:           {self.gaze_in}",
            f"projected gaze vector:       {self.gaze_hidden}",
            f"concatenated classifier in:  {self.combined_dim}",
            f"sentiment logits:            {self.num_labels}",
        ]


class GazeFusionClassifier:
    """Late fusion of a text embedding and a 5-D gaze vector."""

    def __init__(
        self,
        text_dim: int = 32,
        gaze_in: int = DEFAULT_NUM_EYE_TRACKING_FEATURES,
        gaze_hidden: int = DEFAULT_HIDDEN_LAYER_SIZE,
        num_labels: int = DEFAULT_NUM_LABELS,
        rng: np.random.Generator | None = None,
    ) -> None:
        self.text_dim = text_dim
        self.gaze_in = gaze_in
        self.gaze_hidden = gaze_hidden
        self.num_labels = num_labels
        self.rng = rng or np.random.default_rng(7)
        # Xavier-style init keeps the toy loop numerically stable.
        self.W_gaze = self.rng.normal(0, 1 / np.sqrt(gaze_in), size=(gaze_in, gaze_hidden))
        self.b_gaze = np.zeros(gaze_hidden)
        combined = text_dim + gaze_hidden
        self.W_cls = self.rng.normal(0, 1 / np.sqrt(combined), size=(combined, num_labels))
        self.b_cls = np.zeros(num_labels)

    @property
    def shapes(self) -> FusionShapes:
        return FusionShapes(
            text_dim=self.text_dim,
            gaze_in=self.gaze_in,
            gaze_hidden=self.gaze_hidden,
            combined_dim=self.text_dim + self.gaze_hidden,
            num_labels=self.num_labels,
        )

    def project_gaze(self, gaze: np.ndarray) -> np.ndarray:
        gaze = np.asarray(gaze, dtype=float)
        if gaze.ndim == 1:
            gaze = gaze.reshape(1, -1)
        return gaze @ self.W_gaze + self.b_gaze

    def forward(self, text: np.ndarray, gaze: np.ndarray) -> np.ndarray:
        text = np.asarray(text, dtype=float)
        if text.ndim == 1:
            text = text.reshape(1, -1)
        gaze_hidden = self.project_gaze(gaze)
        combined = np.concatenate([text, gaze_hidden], axis=1)
        return combined @ self.W_cls + self.b_cls

    def predict(self, text: np.ndarray, gaze: np.ndarray) -> np.ndarray:
        return np.argmax(self.forward(text, gaze), axis=1)

    def step(
        self,
        text: np.ndarray,
        gaze: np.ndarray,
        labels: np.ndarray,
        lr: float = 0.05,
    ) -> float:
        """One minibatch of vanilla SGD on cross-entropy."""
        text = np.asarray(text, dtype=float)
        gaze = np.asarray(gaze, dtype=float)
        labels = np.asarray(labels, dtype=int)
        gaze_hidden = self.project_gaze(gaze)
        combined = np.concatenate([text, gaze_hidden], axis=1)
        logits = combined @ self.W_cls + self.b_cls
        loss = cross_entropy(logits, labels)

        probs = softmax(logits)
        dlogits = (probs - one_hot(labels, self.num_labels)) / len(labels)
        dW_cls = combined.T @ dlogits
        db_cls = dlogits.sum(axis=0)
        dcombined = dlogits @ self.W_cls.T
        dtext, dgaze_hidden = np.split(dcombined, [self.text_dim], axis=1)
        del dtext  # text vectors are treated as frozen encoder outputs
        dW_gaze = gaze.T @ dgaze_hidden
        db_gaze = dgaze_hidden.sum(axis=0)

        self.W_cls -= lr * dW_cls
        self.b_cls -= lr * db_cls
        self.W_gaze -= lr * dW_gaze
        self.b_gaze -= lr * db_gaze
        return loss


def make_separable_batch(
    n: int = 96,
    text_dim: int = 32,
    gaze_in: int = 5,
    num_labels: int = 3,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build a batch where both text and gaze carry a class signal."""
    rng = np.random.default_rng(seed)
    labels = rng.integers(0, num_labels, size=n)
    text = rng.normal(0, 0.3, size=(n, text_dim))
    gaze = rng.normal(0, 0.3, size=(n, gaze_in))
    for cls in range(num_labels):
        mask = labels == cls
        text[mask, cls] += 2.5
        gaze[mask, cls % gaze_in] += 2.5
    return text, gaze, labels
