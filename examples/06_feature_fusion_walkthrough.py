#!/usr/bin/env python3
"""Numpy walkthrough of the EyeTrackingModel concat head."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from examples.lib.fusion import FusionWalkthrough, demo_batch
from examples.lib.reporting import banner


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-gaze-features", type=int, default=5)
    parser.add_argument("--encoder-hidden", type=int, default=768)
    parser.add_argument("--gaze-hidden", type=int, default=16)
    parser.add_argument("--num-labels", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print shapes and a single probability row as JSON.",
    )
    args = parser.parse_args()

    model = FusionWalkthrough(
        num_gaze_features=args.num_gaze_features,
        encoder_hidden=args.encoder_hidden,
        gaze_hidden=args.gaze_hidden,
        num_labels=args.num_labels,
        seed=args.seed,
    )
    pooled, gaze = demo_batch(
        batch_size=args.batch_size,
        num_gaze_features=args.num_gaze_features,
        encoder_hidden=args.encoder_hidden,
        seed=args.seed + 7,
    )
    out = model.forward(pooled, gaze)

    shapes = {name: list(arr.shape) for name, arr in out.items()}
    shapes["concat_width"] = model.concat_width
    row0 = out["probs"][0]
    payload = {
        "shapes": shapes,
        "probs_row0": row0.tolist(),
        "probs_row0_sum": float(row0.sum()),
        "note": (
            "Random weights, not a trained encoder. Shapes match "
            "EyeTrackingModel: pooler (H) || Linear(gaze→16) → H+16 → labels."
        ),
    }

    if args.json:
        print(json.dumps(payload, indent=2))
        return 0

    banner("Fusion shapes (numpy stand-in for EyeTrackingModel)")
    print(f"pooled:      {tuple(out['pooled'].shape)}   # BERT/RoBERTa pooler_output")
    print(f"gaze_hidden: {tuple(out['gaze_hidden'].shape)}    # Linear({args.num_gaze_features} → {args.gaze_hidden})")
    print(f"concat:      {tuple(out['concat'].shape)}   # {args.encoder_hidden} + {args.gaze_hidden}")
    print(f"logits:      {tuple(out['logits'].shape)}     # num_labels = {args.num_labels}")
    print(f"probs:       {tuple(out['probs'].shape)}")
    print()
    print("softmax(row 0):", " ".join(f"{p:.4f}" for p in row0))
    print(f"sum(row 0):     {row0.sum():.6f}")
    print()
    print(payload["note"])
    print(
        "There is no nonlinearity on the gaze projection in the original "
        "scripts. Dropout (0.1) is applied after concat during training."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
