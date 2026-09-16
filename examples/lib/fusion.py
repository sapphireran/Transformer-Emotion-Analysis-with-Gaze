"""Numpy walkthrough of ``EyeTrackingModel`` fusion shapes.

Random weights, no pretrained encoder. The point is to see concat
widths and a valid 3-way softmax before touching PyTorch.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

DEFAULT_HIDDEN = 768
DEFAULT_GAZE_HIDDEN = 16
DEFAULT_NUM_LABELS = 3
DEFAULT_NUM_GAZE = 5


def softmax(logits: np.ndarray, axis: int = -1) -> np.ndarray:
    shifted = logits - logits.max(axis=axis, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=axis, keepdims=True)


@dataclass
class FusionWalkthrough:
    """Linear gaze projection + concat + classification, matching the scripts."""

    num_gaze_features: int = DEFAULT_NUM_GAZE
    encoder_hidden: int = DEFAULT_HIDDEN
    gaze_hidden: int = DEFAULT_GAZE_HIDDEN
    num_labels: int = DEFAULT_NUM_LABELS
    seed: int = 0

    def __post_init__(self) -> None:
        rng = np.random.default_rng(self.seed)
        self.gaze_weight = rng.normal(
            scale=0.1, size=(self.num_gaze_features, self.gaze_hidden)
        )
        self.gaze_bias = rng.normal(scale=0.01, size=(self.gaze_hidden,))
        concat = self.encoder_hidden + self.gaze_hidden
        self.classifier_weight = rng.normal(scale=0.1, size=(concat, self.num_labels))
        self.classifier_bias = rng.normal(scale=0.01, size=(self.num_labels,))

    @property
    def concat_width(self) -> int:
        return self.encoder_hidden + self.gaze_hidden

    def project_gaze(self, gaze: np.ndarray) -> np.ndarray:
        gaze = np.asarray(gaze, dtype=float)
        if gaze.ndim == 1:
            gaze = gaze[None, :]
        if gaze.shape[-1] != self.num_gaze_features:
            raise ValueError(
                f"expected gaze width {self.num_gaze_features}, got {gaze.shape[-1]}"
            )
        return gaze @ self.gaze_weight + self.gaze_bias

    def forward(
        self,
        pooled: np.ndarray,
        gaze: np.ndarray,
        dropout_rate: float = 0.0,
    ) -> dict[str, np.ndarray]:
        pooled = np.asarray(pooled, dtype=float)
        if pooled.ndim == 1:
            pooled = pooled[None, :]
        if pooled.shape[-1] != self.encoder_hidden:
            raise ValueError(
                f"expected encoder width {self.encoder_hidden}, got {pooled.shape[-1]}"
            )
        gaze_hidden = self.project_gaze(gaze)
        if pooled.shape[0] != gaze_hidden.shape[0]:
            raise ValueError("batch size mismatch between pooled and gaze")
        combined = np.concatenate([pooled, gaze_hidden], axis=1)
        if dropout_rate:
            rng = np.random.default_rng(self.seed + 1)
            keep = rng.random(combined.shape) >= dropout_rate
            combined = combined * keep / max(1.0 - dropout_rate, 1e-6)
        logits = combined @ self.classifier_weight + self.classifier_bias
        probs = softmax(logits, axis=1)
        return {
            "pooled": pooled,
            "gaze_hidden": gaze_hidden,
            "concat": combined,
            "logits": logits,
            "probs": probs,
        }


def demo_batch(
    batch_size: int = 4,
    num_gaze_features: int = DEFAULT_NUM_GAZE,
    encoder_hidden: int = DEFAULT_HIDDEN,
    seed: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    pooled = rng.normal(size=(batch_size, encoder_hidden))
    gaze = rng.normal(size=(batch_size, num_gaze_features))
    return pooled, gaze
