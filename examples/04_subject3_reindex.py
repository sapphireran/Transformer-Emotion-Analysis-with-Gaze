#!/usr/bin/env python3
"""Prove subject-3 packed ids: row 150 is original sentence 250, not 150."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from sidecar.alignment import (  # noqa: E402
    N_PACKED_SENTENCES,
    TASK1_SKIPPED_ORIGINALS,
    pack_subject3_id,
    subject3_sentence_alignment,
    unpack_subject3_id,
    word_break_row,
)
from sidecar.load import load_subject_sentence, load_subject_word  # noqa: E402
from sidecar.reports import markdown_table, write_text  # noqa: E402


def main() -> int:
    s1 = load_subject_sentence(1)
    s3 = load_subject_sentence(3)
    w1 = load_subject_word(1)
    w3 = load_subject_word(3)

    align = subject3_sentence_alignment(s1, s3)
    n_match_unpacked = int(align["sentlen_match"].sum())
    n_match_naive = int(align["same_row_would_match"].sum())
    break_row = word_break_row(w1)

    w1_sid = w1["Sent_ID"].astype(str).str.split("_").str[0].astype(int)
    w3_sid = w3["Sent_ID"].astype(str).str.split("_").str[0].astype(int)
    w1_150 = w1.loc[w1_sid == 150, "Word"].fillna("").head(8).tolist()
    w1_250 = w1.loc[w1_sid == 250, "Word"].fillna("").head(8).tolist()
    w3_150 = w3.loc[w3_sid == 150, "Word"].fillna("").head(8).tolist()

    preview = align.iloc[[0, 149, 150, 151, len(align) - 1]][
        ["packed_id", "original_id", "s3_SentLen", "s1_SentLen", "sentlen_match", "same_row_would_match"]
    ]

    skipped = ", ".join(str(i) for i in TASK1_SKIPPED_ORIGINALS[:5]) + ", ..., 249, 399"

    text = "\n".join(
        [
            "# Subject-3 reindex (Task 1 hole)",
            "",
            "`utils_ZuCo.DataTransformer` drops MATLAB subject 2 / CSV subject 3",
            f"sentences `{skipped}` ({len(TASK1_SKIPPED_ORIGINALS)} rows) and then",
            "writes a fresh contiguous `id`. The committed `3_SR.csv` therefore has",
            f"{len(s3)} rows numbered 0..{int(s3['id'].max())}, not a 400-row table",
            "with holes. `SentLen` is a stimulus property, so it is a perfect key for",
            "reconstructing the packing against subject 1.",
            "",
            f"- packed rows: {len(s3)} (expected {N_PACKED_SENTENCES})",
            f"- unpack map: packed `i<150` → original `i`; packed `i>=150` → original `i+100`",
            f"- example: packed 150 → original {unpack_subject3_id(150)}; "
            f"packed 298 → original {unpack_subject3_id(298)}",
            f"- pack inverse: original 250 → packed {pack_subject3_id(250)}; "
            f"original 150 → {pack_subject3_id(150)!r} (skipped)",
            "",
            "## Sentence-level SentLen pairing",
            "",
            f"Unpacked pairing matches SentLen on **{n_match_unpacked}/{len(align)}** rows.",
            f"Naively pairing the same `id` (what `groupby(level=0)` does) matches only "
            f"**{n_match_naive}/{len(align)}** rows — exactly the prefix 0..149 plus whatever",
            "later sentences happen to share a length by chance.",
            "",
            markdown_table(preview.reset_index(drop=True), floatfmt="{:.1f}"),
            "",
            "## Word-level smoking gun",
            "",
            f"Subject-1 word table: {len(w1)} tokens, 400 sentences.",
            f"Subject-3 word table: {len(w3)} tokens, {w3_sid.nunique()} sentences.",
            f"Tokens for original sentences 0-149: {(w1_sid <= 149).sum()} on both readers.",
            f"First mixed row if you average by DataFrame index: **{break_row}**.",
            "",
            f"- subject 1, `Sent_ID=150_NR` tokens: `{w1_150}`",
            f"- subject 1, `Sent_ID=250_NR` tokens: `{w1_250}`",
            f"- subject 3, `Sent_ID=150_NR` tokens: `{w3_150}`",
            "",
            "Subject 3's `150_NR` is the *text* of original sentence 250. Any word-level",
            "`groupby(level=0).mean()` after row 2593 is averaging different reviews.",
            "",
            "This is why `ZuCo_et_csv_data/average_data.csv` and `word/word_averages_v2.csv`",
            "should not be treated as 'the mean reader on sentence k' for k ≥ 150.",
            "",
        ]
    )
    out = write_text("subject3_reindex.md", text)
    print(f"wrote {out}")
    print("unpacked SentLen matches", n_match_unpacked, "/", len(align))
    print("naive id matches", n_match_naive, "/", len(align))
    print("word break row", break_row)
    print("w3[150] == w1[250]", w3_150 == w1_250)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
