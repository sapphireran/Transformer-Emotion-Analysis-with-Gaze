#!/usr/bin/env python3
"""Dependency-free sketch of EyeTrackingModel's late-fusion arithmetic.

The real model lives in model_ZuCo_SST.py / model_full_SST.py and needs
torch + transformers. This script only checks the shapes I quote in
docs/04-model-architecture.md:

    gaze  [5]  --Linear(5, 16)-->  [16]
    text  [768]                    [768]
    concat                         [784]
    --Linear(784, 3)-->            [3] logits

It also loads sentence 0's z-scored gaze vector and shows that a
*random* 5→16 remix still preserves the obvious pattern (nFix and GPT
are the large positive coordinates). That is not a trained result; it
is a reminder that the gaze tower has only 96 parameters and cannot
invent a new modality.
"""

from __future__ import annotations

import csv
import math
import os
import random
import sys


HIDDEN_TEXT = 768
GAZE_IN = 5
GAZE_HIDDEN = 16
NUM_LABELS = 3


def param_count() -> None:
    gaze_w = GAZE_IN * GAZE_HIDDEN + GAZE_HIDDEN
    clf_w = (HIDDEN_TEXT + GAZE_HIDDEN) * NUM_LABELS + NUM_LABELS
    print("Parameter counts (gaze path only)")
    print(f"  Linear({GAZE_IN} -> {GAZE_HIDDEN})     = {gaze_w:6d}")
    print(f"  Linear({HIDDEN_TEXT + GAZE_HIDDEN} -> {NUM_LABELS})   = {clf_w:6d}")
    print(f"  fusion extras total     = {gaze_w + clf_w:6d}")
    print(f"  roberta-base (approx)   = {125_000_000:6d}")
    print(
        "  The gaze path is "
        f"{(gaze_w + clf_w) / 125_000_000:.6%} of the text checkpoint."
    )


def _dot(row: list[float], vec: list[float]) -> float:
    return sum(a * b for a, b in zip(row, vec))


def affine(matrix: list[list[float]], bias: list[float], vec: list[float]) -> list[float]:
    return [_dot(row, vec) + b for row, b in zip(matrix, bias)]


def load_sentence0_gaze() -> list[float]:
    path = "ZuCo_SST_data/combined_sst_et_standard.csv"
    if not os.path.isfile(path):
        print(f"Missing {path}. Run from the repository root.", file=sys.stderr)
        sys.exit(2)
    with open(path, newline="", encoding="utf-8") as handle:
        row = next(r for r in csv.DictReader(handle) if r["sentence_id"] == "0")
    # Same column order as model_ZuCo_SST.py:
    #   df[['nFixations', 'FFD', 'GPT', 'TRT', 'GD']]
    keys = ["nFixations", "FFD", "GPT", "TRT", "GD"]
    values = [float(row[k]) for k in keys]
    print("Sentence 0 z-scored gaze (Track A input)")
    print(f"  text : {row['sentence']}")
    print(f"  label: {row['sentiment_label']}  (1 = NEUTRAL)")
    for key, value in zip(keys, values):
        print(f"  {key:12s} {value:+.4f}")
    return values


def remix_demo(gaze: list[float], seed: int = 0) -> None:
    rng = random.Random(seed)
    # Small init, similar in spirit to a linear layer that has not
    # been trained: we only want to see that large input coordinates
    # still dominate a 16-D remix.
    matrix = [
        [rng.uniform(-0.3, 0.3) for _ in range(GAZE_IN)]
        for _ in range(GAZE_HIDDEN)
    ]
    bias = [0.0] * GAZE_HIDDEN
    hidden = affine(matrix, bias, gaze)
    print()
    print(f"Random Linear(5, 16) remix (seed={seed}, weights in [-0.3, 0.3])")
    print("  first 8 of 16:", " ".join(f"{v:+.3f}" for v in hidden[:8]))
    energy = math.sqrt(sum(v * v for v in hidden))
    print(f"  L2 of 16-D hidden: {energy:.3f}")
    print(
        "  Concat shape would be "
        f"[{HIDDEN_TEXT} + {GAZE_HIDDEN}] = [{HIDDEN_TEXT + GAZE_HIDDEN}]"
    )
    # Which input feature has the largest |weight| mass?
    mass = [sum(abs(matrix[j][i]) for j in range(GAZE_HIDDEN)) for i in range(GAZE_IN)]
    names = ["nFixations", "FFD", "GPT", "TRT", "GD"]
    print("  abs-weight mass per input (random, not trained):")
    for name, value in zip(names, mass):
        print(f"    {name:12s} {value:.3f}")
    print(
        "  On *this* sentence the large positive inputs are nFixations "
        "and GPT. A trained layer can rotate them, not enlarge the 5-D "
        "family into something unrelated to reading time."
    )


def main() -> int:
    if not os.path.isdir("ZuCo_SST_data"):
        print("Run from the repository root.", file=sys.stderr)
        return 2
    param_count()
    print()
    gaze = load_sentence0_gaze()
    remix_demo(gaze)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
