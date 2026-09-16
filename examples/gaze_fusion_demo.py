#!/usr/bin/env python3
"""Tiny, dependency-free sketch of EyeTrackingModel's tensor shapes.

The real model lives in model_ZuCo_SST.py / model_full_SST.py and needs
PyTorch + a BERT/RoBERTa download. This script only shows:

    gaze (5) ──► Linear(5, 16) ──┐
                                 ├─► concat (784) ──► Linear(784, 3) ──► logits
    text pooler (768) ───────────┘

It also runs the personal sanity ablation: *shuffle the gaze rows* and
confirm the logits change. If they did not change, the fusion branch
would be dead.

    python3 examples/gaze_fusion_demo.py
"""

from __future__ import annotations

import math
import random
import sys
from typing import List, Sequence, Tuple

from common import ZUCO_GAZE_COLS, as_int, read_rows

HIDDEN_TEXT = 768
HIDDEN_GAZE = 16
NUM_GAZE = 5
NUM_LABELS = 3


def matvec(matrix: Sequence[Sequence[float]], vector: Sequence[float]) -> List[float]:
    out = []
    for row in matrix:
        out.append(sum(w * x for w, x in zip(row, vector)))
    return out


def softmax(logits: Sequence[float]) -> List[float]:
    m = max(logits)
    exps = [math.exp(x - m) for x in logits]
    s = sum(exps)
    return [e / s for e in exps]


class TinyFusion:
    """Untrained linear fusion with the same shapes as EyeTrackingModel."""

    def __init__(self, rng: random.Random) -> None:
        # Small random weights so logits are not identically zero.
        scale_g = 0.3 / math.sqrt(NUM_GAZE)
        scale_c = 0.3 / math.sqrt(HIDDEN_TEXT + HIDDEN_GAZE)
        self.W_gaze = [
            [rng.uniform(-scale_g, scale_g) for _ in range(NUM_GAZE)]
            for _ in range(HIDDEN_GAZE)
        ]
        self.b_gaze = [0.0] * HIDDEN_GAZE
        self.W_cls = [
            [rng.uniform(-scale_c, scale_c) for _ in range(HIDDEN_TEXT + HIDDEN_GAZE)]
            for _ in range(NUM_LABELS)
        ]
        self.b_cls = [0.0] * NUM_LABELS

    def gaze_hidden(self, gaze: Sequence[float]) -> List[float]:
        h = matvec(self.W_gaze, gaze)
        return [a + b for a, b in zip(h, self.b_gaze)]

    def forward(self, text: Sequence[float], gaze: Sequence[float]) -> List[float]:
        fused = list(text) + self.gaze_hidden(gaze)
        logits = matvec(self.W_cls, fused)
        return [a + b for a, b in zip(logits, self.b_cls)]


def load_zuco_gaze(limit: int = 32) -> Tuple[List[List[float]], List[int], List[str]]:
    _c, rows = read_rows("ZuCo_SST_data/combined_sst_et_standard.csv")
    gaze = []
    labels = []
    ids = []
    for row in rows[:limit]:
        gaze.append([float(row[c]) for c in ZUCO_GAZE_COLS])
        labels.append(as_int(row["sentiment_label"]))
        ids.append(row["sentence_id"])
    return gaze, labels, ids


def fake_pooler(rng: random.Random, n: int) -> List[List[float]]:
    """Stand-in for BERT/RoBERTa pooler_output. Not a real encoder."""
    return [[rng.gauss(0.0, 0.15) for _ in range(HIDDEN_TEXT)] for _ in range(n)]


def mean_abs_diff(a: Sequence[Sequence[float]], b: Sequence[Sequence[float]]) -> float:
    total = 0.0
    count = 0
    for u, v in zip(a, b):
        for x, y in zip(u, v):
            total += abs(x - y)
            count += 1
    return total / count


def argmax(xs: Sequence[float]) -> int:
    return max(range(len(xs)), key=lambda i: xs[i])


def main() -> int:
    rng = random.Random(42)
    gaze, labels, ids = load_zuco_gaze(32)
    text = fake_pooler(rng, len(gaze))
    model = TinyFusion(rng)

    aligned = [model.forward(t, g) for t, g in zip(text, gaze)]
    shuffled_gaze = gaze[1:] + gaze[:1]
    shuffled = [model.forward(t, g) for t, g in zip(text, shuffled_gaze)]
    text_only_gaze = [[0.0] * NUM_GAZE for _ in gaze]
    text_only = [model.forward(t, g) for t, g in zip(text, text_only_gaze)]

    print("EyeTrackingModel shape sketch (untrained weights, fake pooler)")
    print(f"  batch                 {len(gaze)}")
    print(f"  gaze features         {NUM_GAZE}  {list(ZUCO_GAZE_COLS)}")
    print(f"  gaze hidden           {HIDDEN_GAZE}")
    print(f"  text pooler           {HIDDEN_TEXT}")
    print(f"  concat                {HIDDEN_TEXT + HIDDEN_GAZE}")
    print(f"  logits                {NUM_LABELS}")
    print()
    print("First three sentences (aligned gaze)")
    print(f"{'id':>4} {'gold':>4} {'pred':>4}  {'p_neg':>8} {'p_neu':>8} {'p_pos':>8}  logits")
    for i in range(3):
        probs = softmax(aligned[i])
        print(
            f"{ids[i]:>4} {labels[i]:>4} {argmax(aligned[i]):>4}  "
            f"{probs[0]:8.3f} {probs[1]:8.3f} {probs[2]:8.3f}  "
            f"[{aligned[i][0]:+.3f}, {aligned[i][1]:+.3f}, {aligned[i][2]:+.3f}]"
        )

    print()
    print("Ablation: mean |Δlogit| vs aligned fusion")
    print(f"  shuffled gaze (row k gets gaze of k+1) : {mean_abs_diff(aligned, shuffled):.4f}")
    print(f"  zero gaze (text branch only)           : {mean_abs_diff(aligned, text_only):.4f}")
    print()
    if mean_abs_diff(aligned, shuffled) <= 0.0:
        print("ERROR: shuffling gaze did not change logits — fusion is dead")
        return 1
    print("Shuffling gaze changed the logits, so the 5→16 branch is wired.")
    print("This is not a trained model and the predicted labels are meaningless.")
    print("Use model_ZuCo_SST.py / model_full_SST.py for real fine-tuning.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
