#!/usr/bin/env python3
"""Show the late-fusion shapes used by EyeTrackingModel, without Hugging Face."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gaze_emotion.constants import (
    DEFAULT_HIDDEN_LAYER_SIZE,
    DEFAULT_NUM_EYE_TRACKING_FEATURES,
    FUSION_GAZE_FEATURES,
    label_name,
)
from gaze_emotion.datasets import feature_matrix, load_csv_rows
from gaze_emotion.fusion import GazeFusionClassifier, softmax


def main() -> int:
    print("Late-fusion architecture (NumPy stand-in for EyeTrackingModel)")
    print("=" * 72)
    print()
    print("Original forward pass:")
    print("  pooled = BERT/RoBERTa(...).pooler_output          # [batch, 768]")
    print("  gaze_h = Linear(5, 16)(nFix, FFD, GPT, TRT, GD)   # [batch, 16]")
    print("  logits = Linear(784, 3)(Dropout(concat(pooled, gaze_h)))")
    print()

    # Use the real 768-d width so the printed shapes match the trainers.
    model = GazeFusionClassifier(text_dim=768, rng=np.random.default_rng(0))
    print("Implemented shapes")
    for line in model.shapes.as_lines():
        print(f"  {line}")
    print(f"  W_gaze: {model.W_gaze.shape}   W_cls: {model.W_cls.shape}")
    print()

    rows = load_csv_rows("ZuCo_SST_data/combined_sst_et_standard.csv")[:4]
    gaze = feature_matrix(rows, FUSION_GAZE_FEATURES)
    # Frozen-encoder stand-in: a deterministic hash of the sentence length.
    text = np.zeros((len(rows), 768))
    for i, row in enumerate(rows):
        text[i, :8] = len(row["sentence"]) / 100.0
        text[i, 8:16] = int(row["sentiment_label"]) / 2.0

    logits = model.forward(text, gaze)
    probs = softmax(logits)
    preds = np.argmax(logits, axis=1)

    print(f"Mini-batch of {len(rows)} real ZuCo sentences")
    print("-" * 72)
    for i, row in enumerate(rows):
        gaze_bits = ", ".join(
            f"{name}={gaze[i, j]:+.2f}" for j, name in enumerate(FUSION_GAZE_FEATURES)
        )
        print(f"{i}. [{label_name(row['sentiment_label'])}] {row['sentence'][:70]}")
        print(f"   gaze: {gaze_bits}")
        print(
            f"   logits: {np.array2string(logits[i], precision=3)}  "
            f"probs: {np.array2string(probs[i], precision=3)}  "
            f"pred={label_name(int(preds[i]))}"
        )

    print()
    print("Parameter counts (toy head only; encoder is external):")
    gaze_params = model.W_gaze.size + model.b_gaze.size
    cls_params = model.W_cls.size + model.b_cls.size
    print(f"  gaze projection Linear(5, {DEFAULT_HIDDEN_LAYER_SIZE}): {gaze_params}")
    print(f"  classifier Linear(784, 3): {cls_params}")
    print(f"  fusion head total: {gaze_params + cls_params}")
    print(
        f"  (RoBERTa-base itself is ~125M parameters; the head is "
        f"{DEFAULT_NUM_EYE_TRACKING_FEATURES} -> 16 -> 3.)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
