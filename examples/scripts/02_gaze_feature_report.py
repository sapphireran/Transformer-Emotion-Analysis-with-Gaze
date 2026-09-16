#!/usr/bin/env python3
"""Channel summaries, correlations, and class-conditional means on ZuCo SST."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "examples"))

from gazekit.features import (
    correlation_matrix,
    gaze_by_sentiment,
    missingness_report,
    pairwise_mean_abs_diff,
    summarize_numeric,
)
from gazekit.io import load_sentence_table
from gazekit.paths import default_paths
from gazekit.report import format_frame, section
from gazekit.schema import CANONICAL_GAZE, EXTRA_SENTENCE_ET


def build_report(root: Path | None = None) -> dict:
    paths = default_paths(root)
    df = load_sentence_table(paths.zuco_combined_standard)
    cols = list(CANONICAL_GAZE) + [c for c in EXTRA_SENTENCE_ET if c in df.columns]
    corr = correlation_matrix(df, CANONICAL_GAZE)
    return {
        "n": len(df),
        "missing": missingness_report(df, cols),
        "summary": summarize_numeric(df, cols),
        "corr": corr,
        "by_class": gaze_by_sentiment(df),
        "mean_diffs": pairwise_mean_abs_diff(df),
        "trt_nfix_corr": float(corr.loc["TRT", "nFixations"]) if "TRT" in corr.index else float("nan"),
    }


def render(report: dict) -> str:
    return "\n".join(
        [
            section(
                "ZuCo SST gaze channels (standard-scaled)",
                f"n={report['n']}\nTRT vs nFixations r={report['trt_nfix_corr']:.4f}",
            ),
            section("Missingness", format_frame(report["missing"])),
            section("Numeric summary", format_frame(report["summary"])),
            section("Correlation (canonical five)", format_frame(report["corr"].reset_index())),
            section("Class-conditional means", format_frame(report["by_class"])),
            section(
                "Pairwise |Δmean| / std",
                format_frame(report["mean_diffs"].sort_values("diff_over_std", ascending=False)),
            ),
        ]
    )


def main() -> None:
    print(render(build_report()))


if __name__ == "__main__":
    main()
