#!/usr/bin/env python3
"""Label priors, majority baselines, and gaze-vs-label correlations on both tracks."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from sidecar.load import (  # noqa: E402
    load_sst_combined,
    load_sst_test,
    load_sst_train,
    load_sst_valid,
    load_zuco_split,
    load_zuco_standard,
)
from sidecar.metrics import majority_baseline  # noqa: E402
from sidecar.paths import LABEL_NAMES, SST_GAZE_COLS, ZUCO_ALL_GAZE_COLS  # noqa: E402
from sidecar.reports import markdown_table, write_text  # noqa: E402
from sidecar.stats import class_conditional_means, label_counts, pearson_with_label  # noqa: E402


def _label_block(name: str, df: pd.DataFrame) -> str:
    counts = label_counts(df["sentiment_label"])
    counts["name"] = counts["label"].map(LABEL_NAMES)
    maj = majority_baseline(df["sentiment_label"].to_numpy())
    lines = [
        f"## {name}",
        "",
        f"n = {len(df)}",
        "",
        markdown_table(counts[["label", "name", "n", "share"]]),
        "",
        (
            f"Majority baseline: always predict label {maj['majority_label']} "
            f"({LABEL_NAMES[maj['majority_label']]}) → accuracy {maj['accuracy']:.4f}, "
            f"weighted F1 {maj['f1_weighted']:.4f}."
        ),
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    sst_train = load_sst_train()
    sst_valid = load_sst_valid()
    sst_test = load_sst_test()
    sst_all = load_sst_combined()
    zuco = load_zuco_standard()

    chunks = [
        "# Label atlas",
        "",
        "Both experiment tracks use `sentiment_label ∈ {0, 1, 2}` for negative / neutral / positive.",
        "The full-SST tables are z-scored gaze joined onto ~11.8k SST sentences.",
        "The ZuCo-SST table is 400 actually-read reviews with *measured* gaze.",
        "",
        _label_block("Full SST — train", sst_train),
        _label_block("Full SST — valid", sst_valid),
        _label_block("Full SST — test", sst_test),
        _label_block("Full SST — combined", sst_all),
        _label_block("ZuCo-SST combined (standard scaled)", zuco),
        _label_block("ZuCo-SST train.csv (80/10/10, unused by model_ZuCo_SST.py)", load_zuco_split("train")),
        _label_block("ZuCo-SST valid.csv", load_zuco_split("valid")),
        _label_block("ZuCo-SST test.csv", load_zuco_split("test")),
    ]

    sst_corr = pearson_with_label(sst_train, SST_GAZE_COLS).rename("r_with_label").to_frame()
    sst_corr = sst_corr.reset_index().rename(columns={"index": "feature"})
    zuco_corr = pearson_with_label(zuco, ZUCO_ALL_GAZE_COLS).rename("r_with_label").to_frame()
    zuco_corr = zuco_corr.reset_index().rename(columns={"index": "feature"})

    chunks += [
        "## Pearson r(feature, sentiment_label)",
        "",
        "Weak associations on both tracks. Gaze is not a substitute for the text.",
        "",
        "### Full SST train",
        "",
        markdown_table(sst_corr),
        "",
        "### ZuCo-SST combined",
        "",
        markdown_table(zuco_corr),
        "",
        "## Class-conditional gaze means",
        "",
        "### Full SST train",
        "",
        markdown_table(class_conditional_means(sst_train, SST_GAZE_COLS)),
        "",
        "### ZuCo-SST combined",
        "",
        markdown_table(class_conditional_means(zuco, ZUCO_ALL_GAZE_COLS)),
        "",
        "On full SST, the **neutral** class sits at the *highest* mean nFix/GD/TRT,",
        "and the **positive** class sits below zero (the z-scored mean). That is the",
        "opposite of a simple 'harder text gets more fixations' story once the",
        "sentences are already standardized.",
        "",
    ]
    out = write_text("label_atlas.md", "\n".join(chunks))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
