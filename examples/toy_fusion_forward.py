#!/usr/bin/env python3
"""Replicate the EyeTrackingModel tensor shapes without downloading BERT.

The fusion head in model_ZuCo_SST.py / model_full_SST.py is:

    pooler_output (B, 768)
        concat
    Linear(5, 16)(gaze) (B, 16)
        → dropout → Linear(784, 3) → logits

This script uses numpy random weights so you can see the shapes and a
deterministic numerical check of the concat math.

Usage (repo root):

    python examples/toy_fusion_forward.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

HIDDEN = 768
GAZE_IN = 5
GAZE_H = 16
NUM_LABELS = 3
SEQ_LEN = 128


def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(x, 0.0)


def softmax(logits: np.ndarray) -> np.ndarray:
    z = logits - logits.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


class ToyEyeTrackingModel:
    """Affine gaze projection + concat classifier (no encoder)."""

    def __init__(self, rng: np.random.Generator):
        self.W_gaze = rng.normal(0, 0.02, size=(GAZE_IN, GAZE_H))
        self.b_gaze = np.zeros(GAZE_H)
        self.W_cls = rng.normal(0, 0.02, size=(HIDDEN + GAZE_H, NUM_LABELS))
        self.b_cls = np.zeros(NUM_LABELS)

    def forward(self, pooled: np.ndarray, gaze: np.ndarray, dropout_p: float = 0.0, rng=None):
        if pooled.shape[-1] != HIDDEN:
            raise ValueError(f"pooled last dim {pooled.shape[-1]} != {HIDDEN}")
        if gaze.shape[-1] != GAZE_IN:
            raise ValueError(f"gaze last dim {gaze.shape[-1]} != {GAZE_IN}")
        gaze_h = gaze @ self.W_gaze + self.b_gaze
        concat = np.concatenate([pooled, gaze_h], axis=-1)
        if dropout_p and rng is not None:
            keep = rng.random(concat.shape) >= dropout_p
            concat = concat * keep / (1.0 - dropout_p)
        logits = concat @ self.W_cls + self.b_cls
        return {
            "pooled": pooled,
            "gaze": gaze,
            "gaze_h": gaze_h,
            "concat": concat,
            "logits": logits,
            "probs": softmax(logits),
        }


def _print_shapes(batch: int, tensors: dict) -> None:
    print(f"batch={batch}  seq_len={SEQ_LEN} (tokenizer pad; unused in this toy)")
    print(f"{'tensor':<12} {'shape':<16} {'min':>10} {'max':>10}")
    print("-" * 50)
    for name in ("pooled", "gaze", "gaze_h", "concat", "logits", "probs"):
        arr = tensors[name]
        print(f"{name:<12} {str(tuple(arr.shape)):<16} {arr.min():10.4f} {arr.max():10.4f}")


def _deterministic_check() -> None:
    """Hand-check concat + linear with tiny integers, then the real sizes."""
    pooled = np.ones((2, HIDDEN))
    gaze = np.arange(10, dtype=np.float64).reshape(2, 5)
    model = ToyEyeTrackingModel(np.random.default_rng(0))
    model.W_gaze[:] = 0.0
    model.W_gaze[0, 0] = 1.0  # copy nFixations into gaze_h[0]
    model.b_gaze[:] = 0.0
    model.W_cls[:] = 0.0
    model.W_cls[0, 0] = 1.0       # logit[0] sees pooled[0] == 1
    model.W_cls[HIDDEN, 1] = 1.0  # logit[1] sees gaze_h[0] == nFixations
    out = model.forward(pooled, gaze)
    assert out["concat"].shape == (2, HIDDEN + GAZE_H)
    assert np.allclose(out["gaze_h"][:, 0], gaze[:, 0])
    assert np.allclose(out["logits"][:, 0], 1.0)
    assert np.allclose(out["logits"][:, 1], gaze[:, 0])
    print("deterministic concat check: OK")
    print(f"  concat shape {out['concat'].shape} == {(2, HIDDEN + GAZE_H)}")
    print(f"  logits[:, 0] (from pooled ones) = {out['logits'][:, 0]}")
    print(f"  logits[:, 1] (from nFixations)  = {out['logits'][:, 1]}")


def main() -> None:
    rng = np.random.default_rng(42)
    model = ToyEyeTrackingModel(rng)

    print("=== random batch (mirrors EyeTrackingModel) ===")
    batch = 4
    pooled = rng.normal(0, 1, size=(batch, HIDDEN))
    gaze = rng.normal(0, 1, size=(batch, GAZE_IN))
    out = model.forward(pooled, gaze, dropout_p=0.1, rng=rng)
    _print_shapes(batch, out)
    print()
    print("probs (rows=batch, cols=neg/neu/pos):")
    np.set_printoptions(precision=3, suppress=True)
    print(out["probs"])
    print(f"rows sum to 1: {np.allclose(out['probs'].sum(axis=1), 1.0)}")

    print()
    print("=== text-only sibling (no gaze concat) ===")
    w_text = rng.normal(0, 0.02, size=(HIDDEN, NUM_LABELS))
    logits_text = pooled @ w_text
    print(f"text-only logits shape {logits_text.shape} vs fusion {out['logits'].shape}")
    print("The only extra capacity in fusion is 16 affine gaze dims plus")
    print("the classifier columns that mix them into the 3 logits.")

    print()
    print("=== deterministic math check ===")
    _deterministic_check()

    print()
    print("Real training still needs BertModel/RobertaModel.pooler_output;")
    print("this toy only documents the concat head.")


if __name__ == "__main__":
    main()
