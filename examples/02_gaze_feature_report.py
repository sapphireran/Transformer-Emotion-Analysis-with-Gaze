#!/usr/bin/env python3
"""Summarize measured ZuCo gaze: ranges, correlations, reader agreement."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tea_gaze.features import CORE_FUSION_FEATURES, SENTIMENT_LABELS, describe_features
from tea_gaze.io import load_all_subject_et_frame, load_frame
from tea_gaze.paths import repo_root
from tea_gaze.reports import write_text
from tea_gaze.stats import (
    correlation_matrix,
    group_means,
    mean_pairwise,
    pairwise_subject_correlation,
    summarize_column,
)


def _md_table(headers: list[str], rows: list[list[object]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join("---" for _ in headers) + "|",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(cell) for cell in row) + " |")
    return "\n".join(lines)


def main() -> int:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    raw = load_frame("zuco_et_average")
    labeled = load_frame("zuco_sst_standard")
    subjects = load_all_subject_et_frame()
    out_dir = repo_root() / "examples" / "output"
    out_dir.mkdir(parents=True, exist_ok=True)

    summaries = [summarize_column(name, raw[name].astype(float)) for name in CORE_FUSION_FEATURES]
    summary_rows = [
        [
            item.name,
            item.count,
            f"{item.mean:.3f}",
            f"{item.stdev:.3f}",
            f"{item.minimum:.3f}",
            f"{item.maximum:.3f}",
        ]
        for item in summaries
    ]

    corr = correlation_matrix(
        {name: raw[name].astype(float).tolist() for name in CORE_FUSION_FEATURES}
    )
    corr_rows = []
    for i, name in enumerate(CORE_FUSION_FEATURES):
        corr_rows.append([name] + [f"{corr[i][j]:.2f}" for j in range(len(CORE_FUSION_FEATURES))])

    per_label_lines = []
    for name in CORE_FUSION_FEATURES:
        means = group_means(
            labeled["sentiment_label"].astype(int).tolist(),
            labeled[name].astype(float).tolist(),
        )
        pretty = ", ".join(
            f"{SENTIMENT_LABELS[label]}={means[label]:.3f}" for label in sorted(means)
        )
        per_label_lines.append(f"- `{name}` (z-scored): {pretty}")

    series: dict[int, dict[int, float]] = {}
    for subject, group in subjects.groupby("subject"):
        series[int(subject)] = {
            int(row.id): float(row.nFixations) for row in group.itertuples(index=False)
        }
    pairs = pairwise_subject_correlation(series)
    pair_mean = mean_pairwise(pairs)
    weakest = min(pairs, key=lambda item: item[2])
    strongest = max(pairs, key=lambda item: item[2])

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    axes[0].hist(raw["TRT"], bins=24, color="#2f5d50", alpha=0.85)
    axes[0].set_title("Sentence-mean TRT (raw ms)")
    axes[0].set_xlabel("Total reading time")
    axes[0].set_ylabel("Sentences")
    im = axes[1].imshow(corr, cmap="PuOr", vmin=-1, vmax=1)
    axes[1].set_xticks(range(len(CORE_FUSION_FEATURES)), CORE_FUSION_FEATURES, rotation=45)
    axes[1].set_yticks(range(len(CORE_FUSION_FEATURES)), CORE_FUSION_FEATURES)
    axes[1].set_title("Fusion-feature correlations")
    fig.colorbar(im, ax=axes[1], fraction=0.046)
    fig.tight_layout()
    plot_path = out_dir / "gaze_feature_report.png"
    fig.savefig(plot_path, dpi=140)
    plt.close(fig)

    report = "\n".join(
        [
            "# ZuCo gaze feature report",
            "",
            "Raw units come from `ZuCo_et_csv_data/average_data.csv` (mean of 12 readers). "
            "Label-conditioned means use the z-scored join in "
            "`ZuCo_SST_data/combined_sst_et_standard.csv`.",
            "",
            "## Fusion features (raw sentence means)",
            "",
            _md_table(
                ["Feature", "n", "mean", "stdev", "min", "max"],
                summary_rows,
            ),
            "",
            "## Correlations on raw sentence means",
            "",
            _md_table([" "] + list(CORE_FUSION_FEATURES), corr_rows),
            "",
            "## Mean z-scored gaze by sentiment label",
            "",
            *per_label_lines,
            "",
            "## Inter-reader agreement on nFixations",
            "",
            f"- Mean pairwise Pearson r: **{pair_mean:.3f}** (aligned on sentence id).",
            f"- Strongest pair: readers {strongest[0]} & {strongest[1]} (r={strongest[2]:.3f}, n={strongest[3]}).",
            f"- Weakest pair: readers {weakest[0]} & {weakest[1]} (r={weakest[2]:.3f}, n={weakest[3]}).",
            "- Reader 3 only has 299 sentences; pairs involving that reader use the intersection.",
            "",
            "## Glossary",
            "",
            describe_features(),
            "",
            f"![Gaze histograms and correlations]({plot_path.name})",
            "",
        ]
    )
    md_path = write_text(out_dir / "gaze_feature_report.md", report)
    print(f"Wrote {md_path.relative_to(repo_root())}")
    print(f"Wrote {plot_path.relative_to(repo_root())}")
    print(f"Mean pairwise nFixations r = {pair_mean:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
