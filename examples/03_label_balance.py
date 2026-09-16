#!/usr/bin/env python3
"""Show SST-3 label mix for ZuCo and the full Stanford Sentiment Treebank."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

import _bootstrap  # noqa: F401

from gaze_emotion_examples.catalog import get_dataset
from gaze_emotion_examples.io import load_dataset
from gaze_emotion_examples.labels import label_counts, label_table_markdown, majority_accuracy
from gaze_emotion_examples.paths import repo_root


KEYS = (
    "zuco_sst_standard",
    "zuco_sst_train",
    "zuco_sst_valid",
    "zuco_sst_test",
    "sst_full",
    "sst_train",
    "sst_valid",
    "sst_test",
)


def main() -> None:
    plot_dir = repo_root() / "examples" / "output"
    plot_dir.mkdir(parents=True, exist_ok=True)
    summary_rows = []
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=False)
    for ax, key, title in (
        (axes[0], "zuco_sst_standard", "ZuCo ∩ SST (n=400)"),
        (axes[1], "sst_full", "Full SST (n=11,853)"),
    ):
        spec = get_dataset(key)
        frame = load_dataset(spec)
        counts = label_counts(frame[spec.label_column])
        print(label_table_markdown(counts, spec.title))
        print(f"majority baseline accuracy: {majority_accuracy(frame[spec.label_column]):.3f}\n")
        ax.bar(counts["name"], counts["count"], color=["#b4533a", "#6b7280", "#2f6f4e"])
        ax.set_title(title)
        ax.set_ylabel("sentences")
    fig.tight_layout()
    dest = plot_dir / "label_balance.png"
    fig.savefig(dest, dpi=140)
    plt.close(fig)
    print(f"wrote {dest}")

    for key in KEYS:
        spec = get_dataset(key)
        frame = load_dataset(spec)
        counts = label_counts(frame[spec.label_column])
        row = {"key": key, "n": len(frame)}
        for item in counts.itertuples(index=False):
            row[item.name] = item.count
        row["majority_acc"] = majority_accuracy(frame[spec.label_column])
        summary_rows.append(row)
    summary = pd.DataFrame(summary_rows).fillna(0)
    out = plot_dir / "label_balance.csv"
    summary.to_csv(out, index=False)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
