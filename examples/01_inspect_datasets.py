#!/usr/bin/env python3
"""List every trainer-facing CSV: shape, columns, and a one-line preview."""

from __future__ import annotations

import pandas as pd

from examples._common import banner
from tea_gaze.io import (
    load_full_sst_splits,
    load_gaze_prediction_sample,
    load_subject_sentence_et,
    load_word_averages,
    load_zuco_combined,
    load_zuco_sentiment,
    load_zuco_splits,
)
from tea_gaze.paths import DataPaths
from tea_gaze.reports import markdown_table


def _row(name: str, path, frame: pd.DataFrame) -> dict[str, object]:
    preview = frame.iloc[0]
    sentence = preview["sentence"] if "sentence" in frame.columns else preview.get("Word", "")
    if isinstance(sentence, str) and len(sentence) > 48:
        sentence = sentence[:45] + "..."
    return {
        "name": name,
        "rows": len(frame),
        "cols": frame.shape[1],
        "first_id": preview.get("sentence_id", preview.get("id", "")),
        "preview": sentence,
        "file": path.name,
    }


def main() -> None:
    paths = DataPaths.from_cwd()
    banner("Trainer-facing tables")
    rows = []

    text = load_zuco_sentiment(paths)
    rows.append(_row(text.name, text.path, text.frame))

    for scaling in ("standard", "minmax"):
        bundle = load_zuco_combined(paths, scaling=scaling)
        rows.append(_row(bundle.name, bundle.path, bundle.frame))

    for name, bundle in load_zuco_splits(paths).items():
        rows.append(_row(bundle.name, bundle.path, bundle.frame))

    for name, bundle in load_full_sst_splits(paths).items():
        rows.append(_row(bundle.name, bundle.path, bundle.frame))

    print(markdown_table(pd.DataFrame(rows)))

    banner("Per-subject sentence coverage (ZuCo task 1)")
    coverage = []
    for subject in range(1, 13):
        bundle = load_subject_sentence_et(subject, paths)
        coverage.append({"subject": subject, "sentence_rows": bundle.n_rows, "file": bundle.path.name})
    print(markdown_table(pd.DataFrame(coverage), digits=0))

    banner("Word-level and predicted-gaze samples")
    word = load_word_averages(paths)
    pred = load_gaze_prediction_sample(paths)
    extra = pd.DataFrame(
        [
            {
                "name": word.name,
                "rows": word.n_rows,
                "unique_Sent_ID": word.frame["Sent_ID"].nunique(),
                "columns": ", ".join(word.frame.columns),
            },
            {
                "name": pred.name,
                "rows": pred.n_rows,
                "unique_Sent_ID": pred.frame["sentence_id"].nunique(),
                "columns": ", ".join(pred.frame.columns),
            },
        ]
    )
    print(markdown_table(extra))
    print()
    print("Repo root:", paths.root)
    print("Subject 3 is expected to be short (DataTransformer skips a bad block).")


if __name__ == "__main__":
    main()
