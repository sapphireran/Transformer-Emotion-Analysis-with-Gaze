#!/usr/bin/env python3
"""Majority and gaze-only linear floors on the 400-row ZuCo table."""

from __future__ import annotations

from _util import maybe_write, parser
from zuco_lab import csvio, labels, paths
from zuco_lab.baselines import evaluate_gaze_linear, evaluate_majority, stratified_holdout
from zuco_lab.catalogs import FUSION_FEATURES
from zuco_lab.reports import join_sections, md_heading, md_table


def _xy(rows):
    x = [[float(row[name]) for name in FUSION_FEATURES] for row in rows]
    y = [labels.parse_label(row["sentiment_label"]) for row in rows]
    return x, y


def build() -> str:
    rows = csvio.read_dicts(paths.ZUCO_COMBINED_STANDARD)
    x, y = _xy(rows)
    majority = evaluate_majority(y)
    train_idx, test_idx = stratified_holdout(y, frac=0.2, seed=42)
    train_x = [x[i] for i in train_idx]
    train_y = [y[i] for i in train_idx]
    test_x = [x[i] for i in test_idx]
    test_y = [y[i] for i in test_idx]
    gaze = evaluate_gaze_linear(train_x, train_y, test_x, test_y)
    # Also report in-sample so the linear solver is visibly doing something.
    insample = evaluate_gaze_linear(x, y, x, y)
    return join_sections(
        [
            md_heading("Gaze-only baselines (ZuCo 400)", 1),
            (
                "If the five fusion features already separated sentiment, a linear "
                "readout would show it. They do not, at least not on this z-scored "
                "table. That is useful: it means any gain from "
                "``roberta_eye_tracking`` has to come from *interaction* with the "
                "text encoder, not from gaze as a standalone classifier."
            ),
            md_table(
                ("model", "n", "accuracy", "weighted F1"),
                [
                    (majority.name, majority.n, majority.accuracy, majority.weighted_f1),
                    (f"{gaze.name} (stratified 20% holdout)", gaze.n, gaze.accuracy, gaze.weighted_f1),
                    (f"{insample.name} (in-sample, optimistic)", insample.n, insample.accuracy, insample.weighted_f1),
                ],
            ),
            md_heading("Protocol"),
            (
                f"Features: {', '.join(FUSION_FEATURES)}. "
                "Holdout is stratified on the integer label with seed 42. "
                "The linear model is one-vs-rest least squares with a bias term, "
                "implemented in stdlib (no sklearn)."
            ),
        ]
    )


def main() -> None:
    args = parser("Fit majority and gaze-only linear floors on ZuCo.").parse_args()
    text = build()
    print(text)
    maybe_write(args, "09_gaze_baselines.md", text)


if __name__ == "__main__":
    main()
