"""Pearson correlations, collinear pairs, and scaling checks for gaze channels."""

from __future__ import annotations

import _bootstrap  # noqa: F401

import sys

from teag_examples.gaze import (
    almost_standardized,
    collinear_pairs,
    feature_label_correlations,
    in_unit_interval,
    pearson_matrix,
)
from teag_examples.io import load_full_sst_combined, load_zuco_combined
from teag_examples.paths import docs_assets, examples_output
from teag_examples.reports import corr_to_markdown, write_json, write_markdown
from teag_examples.schema import FULL_SST_GAZE_COLS, ZUCO_JOIN_GAZE_COLS, ZUCO_MODEL_GAZE_COLS
from teag_examples.viz import save_corr_heatmap


def _series_block(title: str, series) -> str:
    lines = [f"### {title}", "", "| Feature | r vs label |", "| --- | ---: |"]
    for name, value in series.items():
        lines.append(f"| `{name}` | {float(value):.4f} |")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    del argv
    zuco = load_zuco_combined("standard")
    sst = load_full_sst_combined()
    zuco_mm = load_zuco_combined("min_max")

    zuco_corr = pearson_matrix(zuco, ZUCO_JOIN_GAZE_COLS)
    sst_corr = pearson_matrix(sst, FULL_SST_GAZE_COLS)
    zuco_pairs = collinear_pairs(zuco_corr, threshold=0.90)
    sst_pairs = collinear_pairs(sst_corr, threshold=0.95)

    zuco_std_ok = {
        col: almost_standardized(zuco[col], mean_tol=1e-8, std_tol=0.02)
        for col in ZUCO_JOIN_GAZE_COLS
    }
    sst_std_ok = {
        col: almost_standardized(sst[col], mean_tol=1e-8, std_tol=0.02)
        for col in FULL_SST_GAZE_COLS
    }
    mm_ok = {col: in_unit_interval(zuco_mm[col]) for col in ZUCO_JOIN_GAZE_COLS}

    save_corr_heatmap(
        zuco_corr,
        "ZuCo sentence gaze (standard combined, n=400)",
        docs_assets() / "zuco_gaze_corr.png",
    )
    save_corr_heatmap(
        sst_corr,
        "Full SST predicted gaze (z-scored, n=11853)",
        docs_assets() / "sst_predicted_gaze_corr.png",
    )

    md = [
        "# Gaze feature analysis",
        "",
        "Computed from committed CSVs. Model concat uses "
        f"ZuCo `{list(ZUCO_MODEL_GAZE_COLS)}` and SST `{list(FULL_SST_GAZE_COLS)}`.",
        "",
        "## ZuCo correlations",
        "",
        corr_to_markdown(zuco_corr),
        _series_block("ZuCo r(feature, sentiment_label)", feature_label_correlations(zuco, ZUCO_JOIN_GAZE_COLS)),
        f"Pairs with |r| ≥ 0.90: {zuco_pairs}",
        "",
        "## Full SST predicted correlations",
        "",
        corr_to_markdown(sst_corr),
        _series_block(
            "SST r(feature, sentiment_label)",
            feature_label_correlations(sst, FULL_SST_GAZE_COLS),
        ),
        f"Pairs with |r| ≥ 0.95: {sst_pairs}",
        "",
        "## Scaling checks",
        "",
        f"- ZuCo standard columns look z-scored: {zuco_std_ok}",
        f"- SST predicted columns look z-scored: {sst_std_ok}",
        f"- ZuCo min-max columns in [0, 1]: {mm_ok}",
        "",
    ]
    text = "\n".join(md)
    write_markdown(text, examples_output() / "gaze_analysis.md")
    write_markdown(text, docs_assets() / "gaze_analysis.md")
    write_json(
        {
            "zuco_collinear_r90": zuco_pairs,
            "sst_collinear_r95": sst_pairs,
            "zuco_standardized": zuco_std_ok,
            "sst_standardized": sst_std_ok,
            "minmax_unit_interval": mm_ok,
        },
        examples_output() / "gaze_analysis.json",
    )
    sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
