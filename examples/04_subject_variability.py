#!/usr/bin/env python3
"""Inter-subject disagreement on ZuCo sentence-level gaze."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from examples.lib.gaze import ZUCO_ALL_GAZE, inter_subject_cv, subject_feature_panel
from examples.lib.loading import (
    SUBJECT_SENTENCE_ROWS,
    load_subject_sentence_tables,
    load_zuco_combined,
    remap_subject3_original_ids,
)
from examples.lib.paths import resolve_root
from examples.lib.reporting import banner, print_frame


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=None)
    parser.add_argument(
        "--top",
        type=int,
        default=8,
        help="How many highest-CV sentences to print per feature.",
    )
    args = parser.parse_args()
    root = resolve_root(args.root)

    tables = load_subject_sentence_tables(root=root)
    banner("Per-subject row counts")
    for subject, df in tables.items():
        print(f"  subject {subject:2d}: {len(df):3d} rows (expected {SUBJECT_SENTENCE_ROWS[subject]})")
    print(
        "Subject 3 is DataTransformer subject 2: 101 sentences dropped, "
        "then reindexed. Remapping to original ids before the CV panel "
        "(see examples/10_zuco_index_alignment.py)."
    )
    tables[3] = remap_subject3_original_ids(tables[3])
    panel = subject_feature_panel(tables)
    cv = inter_subject_cv(panel, ZUCO_ALL_GAZE)
    zuco = load_zuco_combined(root=root)[["sentence_id", "sentence", "sentiment_label"]]

    banner("Mean coefficient of variation across 12 subjects")
    summary = (
        cv.groupby("feature", observed=True)["subject_cv"]
        .agg(mean_cv="mean", median_cv="median")
        .sort_values("mean_cv", ascending=False)
    )
    print_frame(summary)

    banner("Subject means of raw sentence-level features")
    subject_means = (
        panel.groupby("subject", observed=True)[list(ZUCO_ALL_GAZE)].mean().round(3)
    )
    print_frame(subject_means)

    banner(f"Highest-CV sentences (top {args.top} per late measure)")
    late = cv[cv["feature"].isin(["TRT", "GPT", "nFixations"])]
    top = (
        late.sort_values("subject_cv", ascending=False)
        .groupby("feature", observed=True, group_keys=False)
        .head(args.top)
    )
    top = top.merge(zuco, on="sentence_id", how="left")
    print_frame(
        top[["feature", "sentence_id", "subject_cv", "sentiment_label", "sentence"]],
        max_rows=40,
    )
    print(
        "\nAveraging twelve readers is what the training CSVs do. A high "
        "CV means that average is a blur — the fusion layer sees a "
        "compromise, not any one person's scanpath. Pupil size and "
        "omissionRate often move together across subjects; first-pass "
        "durations (FFD, GD) usually disagree more."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
