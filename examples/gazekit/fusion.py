"""NumPy clone of ``EyeTrackingModel``'s affine math (no pretrained weights).

The real model lives in ``model_ZuCo_SST.py`` / ``model_full_SST.py``. This
module exists so docs and tests can lock the concat width and the 5→16
projection without downloading BERT.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .schema import CANONICAL_GAZE

ENCODER_HIDDEN = 768
GAZE_IN = len(CANONICAL_GAZE)
GAZE_HIDDEN = 16
NUM_LABELS = 3
FUSED_WIDTH = ENCODER_HIDDEN + GAZE_HIDDEN  # 784
DROPOUT_P = 0.1  # documented only; this NumPy path is eval-style (no drop)


def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(x, 0.0)


@dataclass
class GazeFusionForward:
    """Random affine stand-in for the late-fusion head."""

    hidden_size: int = ENCODER_HIDDEN
    et_in: int = GAZE_IN
    et_hidden: int = GAZE_HIDDEN
    num_labels: int = NUM_LABELS
    seed: int = 0
    activated: bool = False

    def __post_init__(self) -> None:
        rng = np.random.default_rng(self.seed)
        scale_et = 0.02
        scale_cls = 0.02
        self.W_et = rng.normal(0.0, scale_et, (self.et_in, self.et_hidden))
        self.b_et = np.zeros(self.et_hidden)
        self.W_cls = rng.normal(0.0, scale_cls, (self.hidden_size + self.et_hidden, self.num_labels))
        self.b_cls = np.zeros(self.num_labels)

    def project_gaze(self, et: np.ndarray) -> np.ndarray:
        et = np.asarray(et, dtype="float64")
        if et.ndim == 1:
            et = et[None, :]
        if et.shape[-1] != self.et_in:
            raise ValueError(f"expected last dim {self.et_in}, got {et.shape}")
        hidden = et @ self.W_et + self.b_et
        return relu(hidden) if self.activated else hidden

    def fuse(self, pooled: np.ndarray, et: np.ndarray) -> np.ndarray:
        pooled = np.asarray(pooled, dtype="float64")
        if pooled.ndim == 1:
            pooled = pooled[None, :]
        gaze = self.project_gaze(et)
        if pooled.shape[0] != gaze.shape[0]:
            raise ValueError("batch size mismatch between pooled and gaze")
        if pooled.shape[1] != self.hidden_size:
            raise ValueError(f"expected pooled width {self.hidden_size}, got {pooled.shape[1]}")
        combined = np.concatenate([pooled, gaze], axis=1)
        return combined @ self.W_cls + self.b_cls

    def __call__(self, pooled: np.ndarray, et: np.ndarray) -> np.ndarray:
        return self.fuse(pooled, et)


def shapes_report(
    batch_size: int = 4,
    seed: int = 0,
    activated: bool = False,
) -> dict[str, tuple[int, ...]]:
    """Run a dummy batch and return every intermediate shape."""
    rng = np.random.default_rng(seed)
    model = GazeFusionForward(seed=seed, activated=activated)
    pooled = rng.normal(0, 1, (batch_size, ENCODER_HIDDEN))
    et = rng.normal(0, 1, (batch_size, GAZE_IN))
    gaze_h = model.project_gaze(et)
    logits = model.fuse(pooled, et)
    return {
        "pooled": pooled.shape,
        "eye_tracking_features": et.shape,
        "gaze_hidden": gaze_h.shape,
        "fused": (batch_size, FUSED_WIDTH),
        "logits": logits.shape,
    }


def assert_architecture_constants() -> None:
    """Lock the numbers documented in docs/model-architecture.md."""
    if GAZE_IN != 5:
        raise AssertionError("fusion still consumes exactly five gaze channels")
    if GAZE_HIDDEN != 16:
        raise AssertionError("gaze projection width is 16")
    if FUSED_WIDTH != 784:
        raise AssertionError("768 + 16 must be 784")
    if NUM_LABELS != 3:
        raise AssertionError("three-way sentiment")
