#!/usr/bin/env python3
"""Compare raw, standard-scaled, and min-max ZuCo sentence gaze."""

from __future__ import annotations

import _bootstrap  # noqa: F401

from gaze_emotion_examples.io import load_dataset
from gaze_emotion_examples.scaling import columnwise_rank_agreement, scaler_report
from gaze_emotion_examples.stats import markdown_table


RAW_COLUMNS = [
    "omissionRate",
    "nFixations",
    "meanPupilSize",
    "GD",
    "TRT",
    "FFD",
    "SFD",
    "GPT",
]


def _show(title: str, frame, columns) -> None:
    report = scaler_report(frame, columns)
    print(f"\n== {title} ==")
    print(markdown_table(report.to_frame()))
    print(
        "fingerprint:",
        "standard-like" if report.standard_like else "min-max-like" if report.minmax_like else "raw/mixed",
    )


def main() -> None:
    raw = load_dataset("zuco_sentence_raw").rename(columns={"id": "sentence_id"})
    standard = load_dataset("zuco_sst_standard")
    minmax = load_dataset("zuco_sst_minmax")

    _show("raw subject-averaged gaze", raw, RAW_COLUMNS)
    _show("combined_sst_et_standard.csv", standard, RAW_COLUMNS)
    _show("combined_sst_et_min_max.csv", minmax, RAW_COLUMNS)

    aligned_raw = raw.set_index("sentence_id").loc[standard["sentence_id"], RAW_COLUMNS]
    aligned_std = standard.set_index("sentence_id")[RAW_COLUMNS]
    aligned_mm = minmax.set_index("sentence_id")[RAW_COLUMNS]

    print("\nRank agreement raw vs standard (should be ~1.0 for monotone column scaling):")
    print(columnwise_rank_agreement(aligned_raw, aligned_std, RAW_COLUMNS).round(6).to_string())
    print("\nRank agreement raw vs min-max:")
    print(columnwise_rank_agreement(aligned_raw, aligned_mm, RAW_COLUMNS).round(6).to_string())


if __name__ == "__main__":
    main()
