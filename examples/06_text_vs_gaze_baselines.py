#!/usr/bin/env python3
"""Cheap sklearn stand-ins for the transformer fusion idea.

Fits three logistic models on the 400 ZuCo ∩ SST sentences:

* gaze features only
* TF-IDF of the review text only
* TF-IDF concatenated with the gaze vector

This is not a replacement for BERT/RoBERTa. It answers a smaller question:
does sentence-level gaze add anything on top of a linear bag of n-grams?
"""

from __future__ import annotations

import argparse

import pandas as pd

import _bootstrap  # noqa: F401

from gaze_emotion_examples.fusion import GazeFusionForward, text_vs_gaze_cv
from gaze_emotion_examples.io import load_dataset
from gaze_emotion_examples.labels import majority_accuracy
from gaze_emotion_examples.paths import repo_root
from gaze_emotion_examples.stats import markdown_table


DEFAULT_GAZE = ["nFixations", "GD", "TRT", "FFD", "GPT"]


def _show_shapes() -> None:
    model = GazeFusionForward(text_dim=768, gaze_dim=5, hidden=16, num_labels=3, seed=0)
    import numpy as np

    text = np.zeros((2, 768))
    gaze = np.zeros((2, 5))
    logits = model.forward(text, gaze)
    print("EyeTrackingModel shape sketch (numpy stand-in)")
    print(f"  text         {text.shape}")
    print(f"  gaze         {gaze.shape}")
    print(f"  gaze hidden  {model.project_gaze(gaze).shape}")
    print(f"  logits       {logits.shape}")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument(
        "--gaze-columns",
        nargs="+",
        default=DEFAULT_GAZE,
        help="Subset of gaze channels to concatenate (default matches the training scripts).",
    )
    args = parser.parse_args()

    _show_shapes()
    frame = load_dataset("zuco_sst_standard")
    majority = majority_accuracy(frame["sentiment_label"])
    print(f"majority-class baseline on 400 ZuCo sentences: {majority:.3f}")
    print(f"5-fold stratified CV, gaze columns: {args.gaze_columns}\n")

    results = text_vs_gaze_cv(
        frame,
        text_column="sentence",
        label_column="sentiment_label",
        gaze_columns=args.gaze_columns,
        n_splits=args.folds,
    )
    tables = [result.to_frame() for result in results]
    combined = pd.concat(tables, ignore_index=True)
    print(markdown_table(combined))
    dest = repo_root() / "examples" / "output" / "text_vs_gaze_baselines.csv"
    dest.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(dest, index=False)
    print(f"\nwrote {dest}")
    print("\nMeans:")
    for result in results:
        print(
            f"  {result.name:15} acc={result.mean_accuracy:.3f}  "
            f"macro-F1={result.mean_macro_f1:.3f}  weighted-F1={result.mean_weighted_f1:.3f}"
        )


if __name__ == "__main__":
    main()
