#!/usr/bin/env python3
"""Show word-level ZuCo gaze for a few labeled sentences.

Joins `word_averages_v2.csv` (tokens + reading measures) to
`combined_sst_et_standard.csv` (sentence text + sentiment) so you can
see which words carried the longest first-pass times.
"""

from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from examples.lib import cli, io_csv, paths, report, schema


SENTENCE_IDS = (0, 3, 5)  # one mixed / one short / one positive from the head of the file
WORD_MEASURES = ("nFixations", "FFD", "GD", "TRT", "GPT")


def sentence_index(sent_id: str) -> int:
    return int(sent_id.split("_", 1)[0])


def load() -> tuple[dict[int, dict], dict[int, list[dict]]]:
    sentences = {
        int(row["sentence_id"]): row
        for row in io_csv.read_rows(paths.table_path("zuco_standard"))
    }
    words_by_sent: dict[int, list[dict]] = defaultdict(list)
    for row in io_csv.read_rows(paths.table_path("zuco_word_avg")):
        words_by_sent[sentence_index(row["Sent_ID"])].append(row)
    for rows in words_by_sent.values():
        rows.sort(key=lambda r: int(float(r["Word_ID"])))
    return sentences, words_by_sent


def format_sentence(sent: dict, words: list[dict]) -> str:
    sid = int(sent["sentence_id"])
    label = int(sent["sentiment_label"])
    title = (
        f"## sentence_id={sid}  label={label} ({schema.LABEL_NAMES[label]})\n\n"
        f"{sent['sentence']}"
    )
    table_rows = []
    for word in words:
        table_rows.append(
            [
                int(float(word["Word_ID"])),
                word["Word"] or "unknown",
                float(word["nFixations"]),
                float(word["FFD"]),
                float(word["GD"]),
                float(word["TRT"]),
                float(word["GPT"]),
            ]
        )
    ranked = sorted(words, key=lambda r: float(r["TRT"]), reverse=True)
    top = ", ".join(
        f"{w['Word']} (TRT={float(w['TRT']):.1f})" for w in ranked[:5] if float(w["TRT"]) > 0
    )
    return report.join_sections(
        [
            title,
            report.ascii_table(
                ("word_id", "word") + WORD_MEASURES,
                table_rows,
            ),
            f"Longest total reading time: {top or '(all zero)'}.",
        ]
    )


def main() -> int:
    args = cli.parser("Preview word-level ZuCo gaze on labeled sentences.").parse_args()
    sentences, words_by_sent = load()
    parts = [
        "# Word-level gaze preview",
        "Measures are subject-averaged milliseconds from "
        "`ZuCo_et_csv_data/word/word_averages_v2.csv`. "
        "Zeros usually mean the word was skipped or missing for that reader.",
    ]
    for sid in SENTENCE_IDS:
        if sid not in sentences or sid not in words_by_sent:
            parts.append(f"## sentence_id={sid}\n\nmissing from the join.")
            continue
        parts.append(format_sentence(sentences[sid], words_by_sent[sid]))
    body = report.join_sections(parts)
    print(body)
    written = cli.maybe_write(args, "word_level_preview.md", body)
    if written:
        print(f"\nwrote {written}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
