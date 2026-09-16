#!/usr/bin/env python3
"""Per-class gaze means and a simple association ranking."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from examples.lib import cli, features, io_csv, paths, report, schema, stats


JOBS = (
    ("zuco_standard", "zuco5", "ZuCo 5-d set used by the training scripts"),
    ("zuco_standard", "zuco_full", "ZuCo full sentence-level set"),
    ("sst_train", "sst5", "Full-SST projected 5-d set (train split)"),
)


def section(table_name: str, feature_set: str, title: str) -> str:
    rows = io_csv.read_rows(paths.table_path(table_name))
    means = features.class_means(rows, feature_set)
    names = features.FEATURE_SETS[feature_set]
    mean_rows = []
    for name in names:
        mean_rows.append(
            [
                name,
                *(means[label][name] for label in (0, 1, 2) if label in means),
                stats.mean([float(row[name]) for row in rows]),
                stats.std([float(row[name]) for row in rows]),
            ]
        )
    assoc = features.association_scores(rows, feature_set)
    corr = features.label_correlations(rows, feature_set)
    header = ["feature", "mean neg", "mean neu", "mean pos", "overall mean", "std"]
    return report.join_sections(
        [
            f"## {title}",
            f"Table `{table_name}` ({len(rows)} rows), feature set `{feature_set}`.",
            report.ascii_table(header, mean_rows),
            "Class-mean gap ranking (descriptive only):",
            report.ascii_table(
                ("feature", "mean |class gap|"),
                [(name, score) for name, score in assoc],
            ),
            "Pearson(feature, integer label) — treat as a weak monotone check:",
            report.ascii_table(
                ("feature", "r"),
                [(name, r) for name, r in corr],
            ),
        ]
    )


def main() -> int:
    args = cli.parser("Summarize gaze features by sentiment class.").parse_args()
    parts = [
        "# Gaze feature report",
        "Integer labels: "
        + ", ".join(f"{k}={v}" for k, v in schema.LABEL_NAMES.items())
        + ". Full-SST gaze columns are projected, not human recordings.",
    ]
    for table_name, feature_set, title in JOBS:
        parts.append(section(table_name, feature_set, title))
    body = report.join_sections(parts)
    print(body)
    written = cli.maybe_write(args, "gaze_feature_report.md", body)
    if written:
        print(f"\nwrote {written}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
