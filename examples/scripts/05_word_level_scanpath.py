#!/usr/bin/env python3
"""Word-length effects and a short per-sentence scanpath dump from ZuCo averages."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "examples"))

from gazekit.features import word_length_effects
from gazekit.io import load_sentence_table, load_word_table
from gazekit.paths import default_paths
from gazekit.report import format_frame, section
from gazekit.schema import CANONICAL_GAZE


def build_report(root: Path | None = None, example_sentence_id: int = 0, top_n: int = 12) -> dict:
    paths = default_paths(root)
    words = load_word_table(paths.word_averages)
    sentences = load_sentence_table(paths.zuco_combined_standard)
    effects = word_length_effects(words)
    sample = words[words["sentence_id"] == example_sentence_id].sort_values("Word_ID")
    text_row = sentences.loc[sentences["sentence_id"] == example_sentence_id]
    text = text_row["sentence"].iloc[0] if len(text_row) else ""
    label = int(text_row["sentiment_label"].iloc[0]) if len(text_row) else None
    # Longest-TRT words overall — a cheap "where did readers linger?" list.
    linger = (
        words.nlargest(top_n, "TRT")[["sentence_id", "Word_ID", "word", "WordLen", *CANONICAL_GAZE]]
        if "TRT" in words.columns
        else words.head(0)
    )
    return {
        "n_words": len(words),
        "n_sentences": int(words["sentence_id"].nunique()),
        "effects": effects,
        "example_sentence_id": example_sentence_id,
        "example_text": text,
        "example_label": label,
        "example_words": sample[["Word_ID", "word", "WordLen", *CANONICAL_GAZE]],
        "linger": linger,
        "unknown_rate": float((words["word"].astype(str).str.lower() == "unknown").mean()),
    }


def render(report: dict) -> str:
    return "\n".join(
        [
            section(
                "Word-level ZuCo averages",
                f"n_words={report['n_words']}  n_sentences={report['n_sentences']}\n"
                f"unknown-token rate={report['unknown_rate']:.4f}",
            ),
            section("Mean gaze by word-length bin", format_frame(report["effects"])),
            section(
                f"Scanpath for sentence_id={report['example_sentence_id']} "
                f"(label={report['example_label']})",
                f"{report['example_text']}\n\n{format_frame(report['example_words'])}",
            ),
            section("Highest-TRT words (corpus)", format_frame(report["linger"])),
        ]
    )


def main() -> None:
    print(render(build_report()))


if __name__ == "__main__":
    main()
