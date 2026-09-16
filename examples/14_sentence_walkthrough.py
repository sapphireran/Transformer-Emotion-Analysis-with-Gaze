#!/usr/bin/env python3
"""Walk a handful of ZuCo sentences from text to word-level gaze.

Picks a few ids that illustrate different reading patterns: a long
hedged review, a short three-adjective line, a high-skip stub, and a
clear positive. For each sentence prints polarity, sentence-level
standardized ET, and the averaged word table (skips included).

This is the closest the personal examples get to 'what would a fused
model actually see' without loading RoBERTa.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from examples.lib.gaze import ZUCO_ALL_GAZE, words_for_sentence
from examples.lib.loading import LABEL_NAMES, load_zuco_combined, load_zuco_word_average
from examples.lib.paths import resolve_root
from examples.lib.reporting import banner, print_frame

DEFAULT_IDS = (0, 3, 4, 316)


def _sentence_block(zuco, words, sentence_id: int) -> None:
    row = zuco.loc[zuco["sentence_id"] == sentence_id]
    if row.empty:
        print(f"sentence_id {sentence_id} not in ZuCo combined table")
        return
    rec = row.iloc[0]
    label = int(rec["sentiment_label"])
    banner(f"sentence {sentence_id} — {LABEL_NAMES[label]}")
    print(rec["sentence"])
    print()
    feat = rec[list(ZUCO_ALL_GAZE)].to_frame().T
    feat.index = ["standardized ET"]
    print_frame(feat)
    tokens = words_for_sentence(words, sentence_id)
    if tokens.empty:
        print("\nNo word-level rows for this sentence_id.")
        return
    show = tokens[
        ["Word_ID", "Word", "WordLen", "nFixations", "FFD", "GD", "TRT", "GPT"]
    ].sort_values("Word_ID")
    skipped = int(show["nFixations"].fillna(0).eq(0).sum())
    print(f"\nword-level average (n={len(show)}, skipped={skipped})")
    print_frame(show, max_rows=80)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=None)
    parser.add_argument(
        "--ids",
        default=",".join(str(i) for i in DEFAULT_IDS),
        help="Comma-separated ZuCo sentence_id values.",
    )
    args = parser.parse_args()
    root = resolve_root(args.root)
    ids = tuple(int(part.strip()) for part in args.ids.split(",") if part.strip())

    zuco = load_zuco_combined(root=root)
    words = load_zuco_word_average(root=root)
    print(
        f"Walking {len(ids)} ZuCo sentences. Word rows are 12-subject "
        "averages (word_averages_v2.csv), sentence ET is z-scored."
    )
    for sentence_id in ids:
        _sentence_block(zuco, words, sentence_id)
    print(
        "\nZeros in nFixations/FFD/GD/TRT/GPT are skips, not missing files. "
        "The fusion head never sees this word table — it only gets the five "
        "sentence numbers above. Token-level fusion would have to align "
        "these words to RoBERTa BPE first (docs/known_issues.md item 15)."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
