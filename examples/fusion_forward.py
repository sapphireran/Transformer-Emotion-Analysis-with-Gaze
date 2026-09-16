#!/usr/bin/env python3
"""Numpy clone of EyeTrackingModel.forward — shapes only, no pretrained weights.

Mirrors the Linear(5→16) → concat(pooler, gaze) → Linear(784→3) graph in
model_ZuCo_SST.py and model_full_SST.py.
"""

from __future__ import annotations

import argparse

import numpy as np


ENCODER_HIDDEN = 768
GAZE_FEATURES = 5
GAZE_HIDDEN = 16
NUM_LABELS = 3


def linear(x: np.ndarray, weight: np.ndarray, bias: np.ndarray) -> np.ndarray:
    """PyTorch nn.Linear: y = x @ W.T + b with W shaped (out, in)."""
    return x @ weight.T + bias


def softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


class EyeTrackingFusion:
    def __init__(self, rng: np.random.Generator, encoder_hidden: int = ENCODER_HIDDEN):
        self.encoder_hidden = encoder_hidden
        # Small init so logits stay in a readable range for the demo.
        self.W_gaze = rng.normal(0, 0.1, size=(GAZE_HIDDEN, GAZE_FEATURES))
        self.b_gaze = np.zeros(GAZE_HIDDEN)
        self.W_cls = rng.normal(0, 0.1, size=(NUM_LABELS, encoder_hidden + GAZE_HIDDEN))
        self.b_cls = np.zeros(NUM_LABELS)

    def gaze_hidden(self, gaze: np.ndarray) -> np.ndarray:
        return linear(gaze, self.W_gaze, self.b_gaze)

    def forward(self, pooled: np.ndarray, gaze: np.ndarray) -> np.ndarray:
        if pooled.ndim != 2 or pooled.shape[1] != self.encoder_hidden:
            raise ValueError(f"pooled should be (B, {self.encoder_hidden}), got {pooled.shape}")
        if gaze.ndim != 2 or gaze.shape[1] != GAZE_FEATURES:
            raise ValueError(f"gaze should be (B, {GAZE_FEATURES}), got {gaze.shape}")
        if pooled.shape[0] != gaze.shape[0]:
            raise ValueError("batch sizes of pooled and gaze do not match")
        combined = np.concatenate([pooled, self.gaze_hidden(gaze)], axis=1)
        return linear(combined, self.W_cls, self.b_cls)


class TextOnlyHead:
    def __init__(self, rng: np.random.Generator, encoder_hidden: int = ENCODER_HIDDEN):
        self.W = rng.normal(0, 0.1, size=(NUM_LABELS, encoder_hidden))
        self.b = np.zeros(NUM_LABELS)

    def forward(self, pooled: np.ndarray) -> np.ndarray:
        return linear(pooled, self.W, self.b)


class GazeOnlyHead:
    def __init__(self, rng: np.random.Generator):
        self.W = rng.normal(0, 0.1, size=(NUM_LABELS, GAZE_FEATURES))
        self.b = np.zeros(NUM_LABELS)

    def forward(self, gaze: np.ndarray) -> np.ndarray:
        return linear(gaze, self.W, self.b)


def random_batch(rng: np.random.Generator, batch_size: int) -> tuple[np.ndarray, np.ndarray]:
    pooled = rng.normal(0, 1, size=(batch_size, ENCODER_HIDDEN))
    gaze = rng.normal(0, 1, size=(batch_size, GAZE_FEATURES))
    return pooled, gaze


def run_demo(seed: int = 0, batch_size: int = 8) -> dict:
    rng = np.random.default_rng(seed)
    pooled, gaze = random_batch(rng, batch_size)
    fusion = EyeTrackingFusion(rng)
    text = TextOnlyHead(rng)
    gaze_only = GazeOnlyHead(rng)

    fusion_logits = fusion.forward(pooled, gaze)
    text_logits = text.forward(pooled)
    gaze_logits = gaze_only.forward(gaze)

    # Same text vector, zero gaze vs real gaze — fusion must move.
    zero_gaze = np.zeros_like(gaze)
    moved = fusion.forward(pooled, gaze) - fusion.forward(pooled, zero_gaze)

    return {
        "batch_size": batch_size,
        "fusion_logits_shape": tuple(fusion_logits.shape),
        "text_logits_shape": tuple(text_logits.shape),
        "gaze_logits_shape": tuple(gaze_logits.shape),
        "concat_width": ENCODER_HIDDEN + GAZE_HIDDEN,
        "fusion_probs": softmax(fusion_logits),
        "mean_abs_move_vs_zero_gaze": float(np.mean(np.abs(moved))),
        "fusion_argmax": fusion_logits.argmax(axis=1).tolist(),
        "text_argmax": text_logits.argmax(axis=1).tolist(),
        "gaze_argmax": gaze_logits.argmax(axis=1).tolist(),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    result = run_demo(seed=args.seed, batch_size=args.batch_size)
    expected = (args.batch_size, NUM_LABELS)
    if result["fusion_logits_shape"] != expected:
        print(f"shape error: {result['fusion_logits_shape']} != {expected}")
        return 1
    if result["mean_abs_move_vs_zero_gaze"] <= 0:
        print("fusion ignored the gaze vector")
        return 1

    if not args.quiet:
        print("EyeTracking fusion forward (numpy)")
        print(f"  pooled:     ({args.batch_size}, {ENCODER_HIDDEN})")
        print(f"  gaze:       ({args.batch_size}, {GAZE_FEATURES})")
        print(f"  gaze hid:   ({args.batch_size}, {GAZE_HIDDEN})")
        print(f"  concat:     ({args.batch_size}, {result['concat_width']})")
        print(f"  logits:     {result['fusion_logits_shape']}")
        print(f"  mean |fusion(x,g) - fusion(x,0)| = {result['mean_abs_move_vs_zero_gaze']:.4f}")
        print(f"  fusion argmax: {result['fusion_argmax']}")
        print(f"  text-only argmax: {result['text_argmax']}")
        print(f"  gaze-only argmax: {result['gaze_argmax']}")
        print("  first-row fusion probs:", np.round(result["fusion_probs"][0], 4))

    print("fusion forward OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
