#!/usr/bin/env python3
"""Summarize sentence-level gaze on the ZuCo and full-SST tables."""

from __future__ import annotations

import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import _bootstrap  # noqa: F401

from gaze_emotion_examples.catalog import get_dataset
from gaze_emotion_examples.io import load_dataset
from gaze_emotion_examples.paths import repo_root
from gaze_emotion_examples.stats import (
    correlation_matrix,
    describe_numeric,
    grouped_means,
    high_correlations,
    markdown_table,
)


def _plot_corr(corr, title: str, destination) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 6.0))
    image = ax.imshow(corr.to_numpy(), cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr.columns)), corr.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(corr.index)), corr.index)
    ax.set_title(title)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(destination, dpi=140)
    plt.close(fig)


def _run_one(key: str, plot_dir) -> None:
    spec = get_dataset(key)
    frame = load_dataset(spec)
    print(f"\n== {spec.title} ==")
    print(markdown_table(describe_numeric(frame, spec.gaze_columns)))
    if spec.label_column:
        means = grouped_means(frame, spec.label_column, spec.gaze_columns).reset_index()
        print("\nMeans by sentiment_label:")
        print(markdown_table(means))
    corr = correlation_matrix(frame, spec.gaze_columns)
    pairs = high_correlations(corr, threshold=0.8)
    if not pairs.empty:
        print("\n|r| >= 0.80:")
        print(markdown_table(pairs))
    plot_path = plot_dir / f"corr_{key}.png"
    _plot_corr(corr, spec.title, plot_path)
    print(f"wrote {plot_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--keys",
        nargs="+",
        default=["zuco_sst_standard", "zuco_sentence_raw", "sst_full"],
    )
    args = parser.parse_args()
    plot_dir = repo_root() / "examples" / "output"
    plot_dir.mkdir(parents=True, exist_ok=True)
    for key in args.keys:
        _run_one(key, plot_dir)


if __name__ == "__main__":
    main()
