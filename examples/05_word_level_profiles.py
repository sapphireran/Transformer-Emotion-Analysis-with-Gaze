#!/usr/bin/env python3
"""Walk one ZuCo sentence at word level and rank tokens by reading time."""

from __future__ import annotations

import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import _bootstrap  # noqa: F401

from gaze_emotion_examples.io import load_dataset
from gaze_emotion_examples.paths import repo_root
from gaze_emotion_examples.profiles import (
    profile_frame,
    reconstruct_sentence,
    sentence_profile,
    top_words_by_measure,
)
from gaze_emotion_examples.stats import markdown_table


def _plot_profile(frame, title: str, destination) -> None:
    fig, ax = plt.subplots(figsize=(11, 4.2))
    x = range(len(frame))
    ax.bar(x, frame["TRT"], color="#3f6b8a", label="TRT (ms)")
    ax.plot(x, frame["GD"], color="#c47b2b", marker="o", linewidth=1.4, label="GD (ms)")
    ax.set_xticks(list(x), frame["word"], rotation=55, ha="right")
    ax.set_ylabel("milliseconds")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(destination, dpi=140)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sentence-id", default="0_NR", help="ZuCo Sent_ID, e.g. 0_NR")
    args = parser.parse_args()

    words = load_dataset("zuco_word_raw")
    sentences = load_dataset("zuco_sst_text")
    profile = sentence_profile(words, args.sentence_id)
    frame = profile_frame(profile)
    numeric_id = int(str(args.sentence_id).split("_")[0])
    sst_row = sentences.loc[sentences["sentence_id"] == numeric_id]
    label = int(sst_row["sentiment_label"].iloc[0]) if not sst_row.empty else None
    original = sst_row["sentence"].iloc[0] if not sst_row.empty else "(not in SST join)"

    print("original SST sentence:")
    print(f"  {original}")
    print(f"label: {label}")
    print("reconstructed from word table:")
    print(f"  {reconstruct_sentence(profile)}")
    print()
    print(markdown_table(frame.drop(columns=["skipped"])))
    print("\nLongest total reading time:")
    for item in top_words_by_measure(profile, "trt", k=5):
        print(f"  {item.word!r:12} TRT={item.trt:7.1f} ms  nFix={item.n_fixations:.2f}")
    print("\nLongest go-past time:")
    for item in top_words_by_measure(profile, "gpt", k=5):
        print(f"  {item.word!r:12} GPT={item.gpt:7.1f} ms")

    dest = repo_root() / "examples" / "output" / f"word_profile_{numeric_id}.png"
    dest.parent.mkdir(parents=True, exist_ok=True)
    _plot_profile(frame, f"Sentence {args.sentence_id} total reading time", dest)
    print(f"\nwrote {dest}")


if __name__ == "__main__":
    main()
