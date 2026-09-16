#!/usr/bin/env python3
"""Walk the word-level ZuCo averages the way a reader would.

Shows:
  - how WordLen relates to nFixations / TRT / FFD
  - a closed-class vs open-class split (tiny hand list, not a tagger)
  - one reconstructed sentence with per-word TRT so the sentence-level
    5-d vector is less abstract

    python3 examples/05_word_level_gaze.py
"""

from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np

from common import as_float_column, read_csv, table, write_text

WORD_TABLE = "ZuCo_et_csv_data/word/word_averages_v2.csv"
SENTENCE_TABLE = "ZuCo_SST_data/combined_sst_et_standard.csv"

# Intentionally tiny. This is a teaching split, not POS tagging.
CLOSED_CLASS = {
    "a",
    "an",
    "the",
    "and",
    "or",
    "but",
    "if",
    "to",
    "of",
    "in",
    "on",
    "at",
    "for",
    "from",
    "with",
    "as",
    "by",
    "is",
    "are",
    "was",
    "were",
    "be",
    "been",
    "it",
    "its",
    "this",
    "that",
    "these",
    "those",
    "he",
    "she",
    "they",
    "we",
    "you",
    "i",
    "not",
    "no",
    "so",
    "than",
    "then",
}


def load_words() -> tuple[list[str], list[list[str]]]:
    return read_csv(WORD_TABLE)


def group_by_sentence(header: list[str], rows: list[list[str]]) -> dict[str, list[list[str]]]:
    idx = header.index("Sent_ID")
    grouped: dict[str, list[list[str]]] = defaultdict(list)
    for row in rows:
        grouped[row[idx]].append(row)
    for sid in grouped:
        grouped[sid].sort(key=lambda r: int(float(r[header.index("Word_ID")])))
    return grouped


def length_bins(header: list[str], rows: list[list[str]]) -> str:
    wordlen = as_float_column(rows, header, "WordLen")
    nfix = as_float_column(rows, header, "nFixations")
    trt = as_float_column(rows, header, "TRT")
    ffd = as_float_column(rows, header, "FFD")
    gd = as_float_column(rows, header, "GD")
    bins = [(1, 2, "1-2 chars"), (3, 4, "3-4"), (5, 6, "5-6"), (7, 9, "7-9"), (10, 40, "10+")]
    out_rows = []
    for lo, hi, name in bins:
        mask = (wordlen >= lo) & (wordlen <= hi)
        if not np.any(mask):
            continue
        out_rows.append(
            (
                name,
                str(int(mask.sum())),
                f"{nfix[mask].mean():.2f}",
                f"{ffd[mask].mean():.1f}",
                f"{gd[mask].mean():.1f}",
                f"{trt[mask].mean():.1f}",
            )
        )
    return table(["WordLen", "n", "mean nFix", "mean FFD", "mean GD", "mean TRT"], out_rows)


def closed_vs_open(header: list[str], rows: list[list[str]]) -> str:
    widx = header.index("Word")
    nfix = as_float_column(rows, header, "nFixations")
    trt = as_float_column(rows, header, "TRT")
    ffd = as_float_column(rows, header, "FFD")
    gpt = as_float_column(rows, header, "GPT")
    wordlen = as_float_column(rows, header, "WordLen")
    words = [row[widx].strip().lower() for row in rows]
    closed = np.array([w in CLOSED_CLASS for w in words])
    open_m = ~closed
    rows_out = []
    for name, mask in (("closed-class (tiny list)", closed), ("everything else", open_m)):
        rows_out.append(
            (
                name,
                str(int(mask.sum())),
                f"{wordlen[mask].mean():.2f}",
                f"{nfix[mask].mean():.2f}",
                f"{ffd[mask].mean():.1f}",
                f"{trt[mask].mean():.1f}",
                f"{gpt[mask].mean():.1f}",
            )
        )
    return table(
        ["group", "n", "mean WordLen", "mean nFix", "mean FFD", "mean TRT", "mean GPT"],
        rows_out,
    )


def render_sentence(header: list[str], word_rows: list[list[str]], label_line: str) -> str:
    w = header.index("Word")
    nfix = header.index("nFixations")
    ffd = header.index("FFD")
    trt = header.index("TRT")
    gpt = header.index("GPT")
    gd = header.index("GD")
    body = [
        label_line,
        table(
            ["word", "nFix", "FFD", "GD", "TRT", "GPT"],
            [
                (
                    row[w] or "unknown",
                    f"{float(row[nfix]):.2f}",
                    f"{float(row[ffd]):.1f}",
                    f"{float(row[gd]):.1f}",
                    f"{float(row[trt]):.1f}",
                    f"{float(row[gpt]):.1f}",
                )
                for row in word_rows
            ],
        ),
        "",
        "reconstructed: " + " ".join(row[w] or "unknown" for row in word_rows),
        "",
    ]
    return "\n".join(body)


def pick_sentence_ids(grouped: dict[str, list[list[str]]]) -> list[str]:
    """Prefer 0_NR (first ZuCo sentence) plus a short and a long one."""
    ids = []
    if "0_NR" in grouped:
        ids.append("0_NR")
    lengths = sorted(((sid, len(rows)) for sid, rows in grouped.items()), key=lambda t: t[1])
    if lengths:
        short = lengths[0][0]
        long = lengths[-1][0]
        if short not in ids:
            ids.append(short)
        if long not in ids:
            ids.append(long)
    return ids[:3]


def main() -> int:
    header, rows = read_csv(WORD_TABLE)
    sent_header, sent_rows = read_csv(SENTENCE_TABLE)
    sent_by_id = {int(float(r[sent_header.index("sentence_id")])): r for r in sent_rows}

    grouped = group_by_sentence(header, rows)
    nfix = as_float_column(rows, header, "nFixations")
    wordlen = as_float_column(rows, header, "WordLen")
    trt = as_float_column(rows, header, "TRT")

    never = int(np.sum(nfix == 0))
    corr_len_nfix = float(np.corrcoef(wordlen, nfix)[0, 1])
    corr_len_trt = float(np.corrcoef(wordlen, trt)[0, 1])

    lines = [
        "# Word-level ZuCo averages",
        f"path: {WORD_TABLE}",
        f"tokens: {len(rows)}  sentences: {len(grouped)}",
        f"rows with nFixations == 0: {never} ({100 * never / len(rows):.1f}%)",
        f"corr(WordLen, nFixations) = {corr_len_nfix:.3f}",
        f"corr(WordLen, TRT) = {corr_len_trt:.3f}",
        "",
        "### mean gaze by character length",
        length_bins(header, rows),
        "",
        "### closed-class list vs the rest",
        "The closed-class list is a handful of function words, not a tagger.",
        closed_vs_open(header, rows),
        "",
        "### reconstructed sentences",
        "Gaze values are raw-ish subject averages (milliseconds / counts),",
        "not the z-scores on the sentence-level training table.",
        "",
    ]

    for sid in pick_sentence_ids(grouped):
        numeric = int(sid.split("_")[0])
        if numeric in sent_by_id:
            row = sent_by_id[numeric]
            label = int(float(row[sent_header.index("sentiment_label")]))
            sentence = row[sent_header.index("sentence")]
            caption = f"Sent_ID {sid}  label={label}  SST: {sentence}"
        else:
            caption = f"Sent_ID {sid}  (no matching SST row)"
        lines.append(render_sentence(header, grouped[sid], caption))

    text = "\n".join(lines)
    print(text)
    write_text("05_word_level_gaze.txt", text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
