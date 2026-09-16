#!/usr/bin/env python3
"""Class balance and gaze–label correlations for both experiment tracks."""

from __future__ import annotations

from examples._common import banner
from tea_gaze.features import build_feature_report, collinearity_pairs
from tea_gaze.io import load_full_sst_splits, load_zuco_combined, load_zuco_splits
from tea_gaze.reports import label_counts, markdown_table, series_to_frame
from tea_gaze.schema import SST_MODEL_FEATURES, ZUCO_MODEL_FEATURES, ZUCO_SENTENCE_ET_COLUMNS


def _print_labels(title: str, frame) -> None:
    banner(title)
    print(markdown_table(label_counts(frame)))


def main() -> None:
    zuco = load_zuco_combined(scaling="standard").frame
    sst = load_full_sst_splits()["combined"].frame

    _print_labels("ZuCo combined (n=400) label mix", zuco)
    _print_labels("Full SST combined (n=11853) label mix", sst)

    banner("ZuCo 80/10/10 files (not used by model_ZuCo_SST.py)")
    for name, bundle in load_zuco_splits().items():
        counts = label_counts(bundle.frame)
        counts.insert(0, "split", name)
        print(f"\n{name} n={len(bundle.frame)}")
        print(markdown_table(counts))

    banner("ZuCo: correlation of ET features with sentiment_label")
    zuco_report = build_feature_report("zuco", zuco, ZUCO_SENTENCE_ET_COLUMNS)
    assert zuco_report.corr_with_label is not None
    print(markdown_table(series_to_frame(zuco_report.corr_with_label.sort_values(), "r")))
    print("\nTrainer-used subset:", ", ".join(ZUCO_MODEL_FEATURES))

    banner("Full SST: correlation of predicted ET with sentiment_label")
    sst_report = build_feature_report("sst", sst, SST_MODEL_FEATURES)
    assert sst_report.corr_with_label is not None
    print(markdown_table(series_to_frame(sst_report.corr_with_label.sort_values(), "r")))

    banner("Collinear pairs (|r| >= 0.95)")
    print("ZuCo trainer features:")
    zuco_pairs = collinearity_pairs(zuco[list(ZUCO_MODEL_FEATURES)].corr())
    if zuco_pairs:
        for left, right, value in zuco_pairs:
            print(f"  {left:>12}  {right:>12}  r={value:.3f}")
    else:
        print("  (none)")
    print("Full SST predicted features:")
    for left, right, value in collinearity_pairs(sst[list(SST_MODEL_FEATURES)].corr()):
        print(f"  {left:>12}  {right:>12}  r={value:.3f}")

    banner("ZuCo pairwise correlation (all stored ET columns)")
    print(zuco_report.pairwise_corr.round(3).to_string())


if __name__ == "__main__":
    main()
