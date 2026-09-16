#!/usr/bin/env python3
"""Hashed-text vs gaze vs late-fusion logistic regression on 400 ZuCo rows.

This is not RoBERTa. It is the cheap question in docs/design-notes.md:
do the five z-scored gaze columns move a linear text model at all?

Folds match the historical trainer: StratifiedKFold(5, seed=42).

Usage:
    python3 examples/toy_text_gaze_fusion.py
    python3 examples/toy_text_gaze_fusion.py --folds 5 --out examples/output
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from examples.lib.fusion import run_fusion_cv
from examples.lib.loaders import LABEL_NAMES, label_counts, load_zuco_experiment


ORDER = ("majority", "length", "gaze", "text", "fusion")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)

    df = load_zuco_experiment("standard")
    counts = label_counts(df["sentiment_label"])
    n = sum(counts.values())
    print("Toy late fusion on ZuCo ∩ SST (measured gaze, z-scored)")
    print(f"rows: {n}")
    print(
        "labels: "
        + ", ".join(f"{LABEL_NAMES[k]}={counts[k]} ({counts[k] / n:.1%})" for k in (0, 1, 2))
    )
    print(f"folds: StratifiedKFold({args.folds}, random_state={args.seed})")
    print("text features: HashingVectorizer 4096-d, word 1-2 grams")
    print("gaze features: nFixations, FFD, GPT, TRT, GD")
    print("classifier: multinomial logistic regression")
    print()

    results = run_fusion_cv(df, n_splits=args.folds, random_state=args.seed)

    lines = [
        "Mean ± std over folds",
        "---------------------",
    ]
    for kind in ORDER:
        lines.append(results[kind].mean_bundle_line())
    lines.append("")
    lines.append("Per-fold accuracy")
    lines.append("-----------------")
    header = f"{'fold':>6}" + "".join(f"{k:>12}" for k in ORDER)
    lines.append(header)
    n_folds = len(results["text"].folds)
    for i in range(n_folds):
        cells = "".join(f"{results[k].folds[i].metrics.accuracy:12.4f}" for k in ORDER)
        lines.append(f"{i + 1:6d}{cells}")
    lines.append("")
    lines.append("How to read this")
    lines.append("----------------")
    lines.append("* majority / length: is the label just the most common class or review length?")
    lines.append("* gaze: five reading-time z-scores, no words.")
    lines.append("* text: hashed n-grams, no gaze.")
    lines.append("* fusion: concat of text block and gaze block (late fusion).")
    lines.append("* If fusion ≈ text and gaze ≈ majority, RoBERTa+gaze is unlikely to be magic.")
    lines.append("* If gaze is strong, look for length leakage before celebrating.")
    lines.append("* This is not a substitute for model_ZuCo_SST.py.")

    text = "\n".join(lines)
    print(text)

    if args.out:
        args.out.mkdir(parents=True, exist_ok=True)
        dest = args.out / "toy_text_gaze_fusion.txt"
        dest.write_text(
            f"rows={n} folds={args.folds} seed={args.seed}\n\n" + text + "\n",
            encoding="utf-8",
        )
        print(f"\nWrote {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
