#!/usr/bin/env python3
"""Re-aggregate word_averages_v2.csv and compare to the sentence-level means.

The historical word averager groups by row index. This example groups by
Sent_ID instead, which is the join key I would use if I rebuilt the table.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from examples._common import banner
from tea_gaze.features import word_to_sentence_means
from tea_gaze.io import load_word_averages
from tea_gaze.paths import DataPaths
from tea_gaze.reports import markdown_table
from tea_gaze.schema import ZUCO_WORD_ET_COLUMNS


def main() -> None:
    paths = DataPaths.from_cwd()
    words = load_word_averages(paths).frame
    sentence_raw = pd.read_csv(paths.sentence_averages_raw)

    banner("Word table")
    print(f"rows={len(words)}  sentences={words['Sent_ID'].nunique()}  "
          f"zero_nFix={(words['nFixations'] == 0).sum()}  "
          f"empty_or_unknown_word={(words['Word'].fillna('').isin(['', 'unknown'])).sum()}")

    rebuilt = word_to_sentence_means(words)
    rebuilt["sentence_index"] = rebuilt["Sent_ID"].astype(str).str.split("_").str[0].astype(int)
    rebuilt = rebuilt.sort_values("sentence_index")

    banner("Rebuilt sentence means from word_averages_v2.csv")
    print(markdown_table(rebuilt.head(8).rename(columns={"n_words": "n_words"})))

    overlap_cols = [col for col in ZUCO_WORD_ET_COLUMNS if col in sentence_raw.columns]
    merged = rebuilt.merge(
        sentence_raw.reset_index().rename(columns={"id": "sentence_index"})
        if "id" not in sentence_raw.columns
        else sentence_raw.rename(columns={"id": "sentence_index"}),
        on="sentence_index",
        suffixes=("_from_words", "_from_sent"),
    )

    banner("How close is word-mean vs committed sentence average?")
    rows = []
    for col in overlap_cols:
        left = merged[f"{col}_from_words"].to_numpy(dtype=float)
        right = merged[f"{col}_from_sent"].to_numpy(dtype=float)
        mask = np.isfinite(left) & np.isfinite(right)
        if mask.sum() == 0:
            continue
        delta = np.abs(left[mask] - right[mask])
        corr = np.corrcoef(left[mask], right[mask])[0, 1] if mask.sum() > 1 else float("nan")
        rows.append(
            {
                "feature": col,
                "n": int(mask.sum()),
                "mean_abs_diff": float(delta.mean()),
                "max_abs_diff": float(delta.max()),
                "r": float(corr),
            }
        )
    print(markdown_table(pd.DataFrame(rows)))
    print()
    print("Large diffs are expected: sentence CSVs average only fixated words")
    print("and then average readers; this rebuild averages already-averaged words.")
    print("Use this as a join-key check, not as a bit-identical repro.")


if __name__ == "__main__":
    main()
