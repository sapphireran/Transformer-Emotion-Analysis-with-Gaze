#!/usr/bin/env python3
"""Word-level skip (nFixations==0) rates per ZuCo reader, plus empty tokens."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from sidecar.load import load_all_subject_word  # noqa: E402
from sidecar.reports import markdown_table, write_text  # noqa: E402


def main() -> int:
    words = load_all_subject_word()
    rows = []
    for i, df in words.items():
        nfix = df["nFixations"]
        empty = df["Word"].isna() | (df["Word"].astype(str).str.len() == 0)
        rows.append(
            {
                "subject": i,
                "n_tokens": len(df),
                "n_sentences": df["Sent_ID"].nunique(),
                "skip_rate": float((nfix == 0).mean()),
                "mean_nFix": float(nfix.mean()),
                "mean_TRT": float(df["TRT"].mean()),
                "mean_WordLen": float(df["WordLen"].mean()),
                "empty_tokens": int(empty.sum()),
                "r_nFix_WordLen": float(df["nFixations"].corr(df["WordLen"])),
            }
        )
    table = pd.DataFrame(rows)
    text = "\n".join(
        [
            "# Word-level skips",
            "",
            "A skip here is `nFixations == 0` on the committed word-level `*_SR.csv`",
            "files. Subject 3 is the short table (299 sentences, packed ids) *and* the",
            "highest skip rate. That is a second reason not to row-average it with the",
            "other eleven readers.",
            "",
            markdown_table(table),
            "",
            f"Mean skip rate excluding subject 3: "
            f"{table.loc[table.subject != 3, 'skip_rate'].mean():.4f}.",
            f"Subject 3 skip rate: {float(table.loc[table.subject == 3, 'skip_rate'].iloc[0]):.4f}.",
            "",
            "Empty `Word` cells come from `re.sub('[^\\w\\s]', '', word.content)` in",
            "`utils_ZuCo.py` wiping tokens that were only punctuation. `WordLen` is 0",
            "on those rows. They are harmless for sentence-level means but they do",
            "show up as `'unknown'` once `word/get_average.py` fills nulls.",
            "",
        ]
    )
    out = write_text("word_skips.md", text)
    print(f"wrote {out}")
    print(table.loc[table.subject == 3, ["n_tokens", "skip_rate"]].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
