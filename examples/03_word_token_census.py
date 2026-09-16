#!/usr/bin/env python3
"""List punctuation-strip artifacts in the word-average table."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gazebook.csvio import read_dicts
from gazebook.paths import repo_root
from gazebook.reports import md_table, write_text
from gazebook.tokens import (
    DIGIT_TOKEN_COUNT,
    KNOWN_GLUED,
    WORDLEN_MISMATCH_COUNT,
    digit_tokens,
    glued_hits,
    load_word_averages,
    sentence_words,
    unknown_tokens,
    wordlen_mismatches,
)


def main() -> int:
    root = repo_root()
    words = load_word_averages(root)
    _, text_rows = read_dicts(root / "ZuCo_SST_data/ssts_ZuCo.csv")
    text = {int(r["sentence_id"]): r["sentence"] for r in text_rows}

    glued = glued_hits(words)
    print("Known glued / mangled tokens")
    intended = {tok: orig for _, tok, orig in KNOWN_GLUED}
    print(
        md_table(
            ["Sent_ID", "token in CSV", "intended", "WordLen", "mean nFix"],
            [
                [hit.sent_id, hit.word, intended[hit.word], hit.word_len, hit.n_fixations]
                for hit in glued
            ],
        )
    )

    mismatches = wordlen_mismatches(words)
    digits = digit_tokens(words)
    unknowns = unknown_tokens(words)
    print()
    print(f"WordLen != len(Word): {len(mismatches)} (target {WORDLEN_MISMATCH_COUNT})")
    print(f"Tokens containing a digit: {len(digits)} (target {DIGIT_TOKEN_COUNT})")
    print(f"empty/unknown placeholders: {len(unknowns)}")

    print()
    print("Sentence 4 (empty → emp11111ty)")
    print(" ", text[4])
    print(
        md_table(
            ["Word_ID", "Word", "nFixations", "WordLen"],
            [[w["Word_ID"], w["Word"], float(w["nFixations"]), w["WordLen"]] for w in sentence_words("4_NR", words)],
        )
    )
    print()
    print("Sentence 80 (hyphen stripped)")
    print(" ", text[80])
    print(
        md_table(
            ["Word_ID", "Word", "nFixations"],
            [[w["Word_ID"], w["Word"], float(w["nFixations"])] for w in sentence_words("80_NR", words)],
        )
    )

    lines = [
        "# Word token census",
        "",
        f"Glued tokens recovered: {len(glued)} / {len(KNOWN_GLUED)}",
        f"WordLen mismatches: {len(mismatches)}",
        f"Digit tokens: {len(digits)}",
        f"unknown placeholders: {len(unknowns)}",
        "",
        "Sentence 4 text:",
        text[4],
        "",
        "Sentence 80 text:",
        text[80],
        "",
    ]
    write_text(root / "examples/output/03_word_token_census.md", "\n".join(lines))

    if len(mismatches) != WORDLEN_MISMATCH_COUNT or len(digits) != DIGIT_TOKEN_COUNT:
        return 1
    if {h.word for h in glued} != {tok for _, tok, _ in KNOWN_GLUED}:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
