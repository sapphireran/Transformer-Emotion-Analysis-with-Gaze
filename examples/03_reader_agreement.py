#!/usr/bin/env python3
"""Inter-reader agreement on the clean 0–149 window."""

from __future__ import annotations

from _util import maybe_write, parser
from zuco_lab.reports import join_sections, md_heading, md_table
from zuco_lab.subjects import SENTENCE_FEATURES, reader_agreement


def build() -> str:
    rows = []
    for feature in SENTENCE_FEATURES:
        if feature == "SentLen":
            continue
        report = reader_agreement(feature, 0, 149)
        rows.append(
            (
                feature,
                report.mean_cv,
                report.mean_pairwise_r,
                report.min_pairwise_r,
                report.max_pairwise_r,
                report.n_pairs,
            )
        )
    return join_sections(
        [
            md_heading("Twelve-reader agreement (ids 0–149)", 1),
            (
                "Only the first 150 sentences are used. After that, reader 3's ids "
                "are compacted and pairwise correlations would mix different texts."
            ),
            md_table(
                ("feature", "mean CV", "mean pairwise r", "min r", "max r", "pairs"),
                rows,
            ),
            md_heading("Notes"),
            (
                "SentLen is omitted because it is a property of the text, not the "
                "reader: every aligned file has the same length, so correlation is "
                "undefined or trivial. Duration measures (GD, TRT, GPT) usually "
                "agree more than pupil size. A mean pairwise r around 0.3–0.5 means "
                "the readers are not interchangeable, which is why averaging them "
                "is a modelling choice rather than a free lunch."
            ),
        ]
    )


def main() -> None:
    args = parser("Measure inter-reader CV and Pearson r on aligned sentences.").parse_args()
    text = build()
    print(text)
    maybe_write(args, "03_reader_agreement.md", text)


if __name__ == "__main__":
    main()
