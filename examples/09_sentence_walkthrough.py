#!/usr/bin/env python3
"""Walk a negative / neutral / positive sentence at word level."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gazebook.csvio import read_dicts
from gazebook.paths import repo_root
from gazebook.reports import md_table, write_text
from gazebook.tokens import sentence_words, load_word_averages

# One id per class, chosen because each has a documented token artifact or a
# short enough line to read in a terminal.
WALK = (
    (4, "empty → emp11111ty"),
    (3, "short neutral, extreme nFixations"),
    (80, "hyphen stripped: murder-on-campus"),
)


def main() -> int:
    root = repo_root()
    _, combined = read_dicts(root / "ZuCo_SST_data/combined_sst_et_standard.csv")
    by = {int(r["sentence_id"]): r for r in combined}
    words = load_word_averages(root)
    chunks = []

    for sid, note in WALK:
        row = by[sid]
        tokens = sentence_words(f"{sid}_NR", words)
        print(f"## sentence {sid}  label={row['sentiment_label']}  ({note})")
        print(row["sentence"])
        print(
            "z-scored sentence gaze: "
            + ", ".join(f"{k}={float(row[k]):+.3f}" for k in ("nFixations", "FFD", "GPT", "TRT", "GD"))
        )
        table = md_table(
            ["Word_ID", "Word", "nFix", "FFD", "TRT", "GPT", "WordLen"],
            [
                [
                    w["Word_ID"],
                    w["Word"],
                    float(w["nFixations"]),
                    float(w["FFD"]),
                    float(w["TRT"]),
                    float(w["GPT"]),
                    w["WordLen"],
                ]
                for w in tokens
            ],
        )
        print(table)
        print()
        chunks.append(f"## sentence {sid}\n\n{row['sentence']}\n\n{table}\n")

    write_text(root / "examples/output/09_sentence_walkthrough.md", "\n".join(chunks))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
