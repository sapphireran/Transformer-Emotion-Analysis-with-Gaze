#!/usr/bin/env python3
"""Word-level skips on ZuCo: by length and by sentence polarity."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from examples.lib.gaze import sentence_skip_rate, skip_rate_by_word_length
from examples.lib.loading import LABEL_NAMES, load_zuco_combined, load_zuco_word_average
from examples.lib.paths import resolve_root
from examples.lib.reporting import banner, print_frame


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=None)
    args = parser.parse_args()
    root = resolve_root(args.root)

    words = load_zuco_word_average(root=root)
    zuco = load_zuco_combined(root=root)[["sentence_id", "sentiment_label", "sentence"]]

    skipped = words["nFixations"].fillna(0).eq(0)
    banner("Overall skip rate (12-subject word average)")
    print(f"words: {len(words)}")
    print(f"skipped (nFixations == 0): {int(skipped.sum())}")
    print(f"skip rate: {float(skipped.mean()):.4f}")

    banner("Skip rate by character length")
    by_len = skip_rate_by_word_length(words)
    print_frame(by_len)

    sent = sentence_skip_rate(words).merge(zuco, on="sentence_id", how="left")
    banner("Skip rate by sentence polarity")
    by_label = (
        sent.groupby("sentiment_label", observed=True)["skip_rate"]
        .agg(mean_skip_rate="mean", median_skip_rate="median", n_sentences="size")
        .reset_index()
    )
    by_label["class"] = by_label["sentiment_label"].map(LABEL_NAMES)
    print_frame(by_label[["class", "n_sentences", "mean_skip_rate", "median_skip_rate"]])

    banner("Highest-skip sentences")
    hardest = sent.sort_values("skip_rate", ascending=False).head(8)
    print_frame(hardest[["sentence_id", "skip_rate", "n_words", "sentiment_label", "sentence"]])

    print(
        "\nShort function words are skipped most. That is expected reading "
        "behavior, not a data bug. Sentence-level omissionRate in the "
        "ZuCo tables is the same idea, computed by the original ZuCo "
        "release rather than from these averages."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
