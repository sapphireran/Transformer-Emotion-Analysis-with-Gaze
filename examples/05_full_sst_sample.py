#!/usr/bin/env python3
"""Show how the larger SST table differs from measured ZuCo gaze."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tea_gaze.features import CORE_FUSION_FEATURES, SENTIMENT_LABELS, fusion_feature_frame
from tea_gaze.io import load_frame
from tea_gaze.paths import repo_root
from tea_gaze.reports import markdown_label_table, write_text
from tea_gaze.schema import summarize_labels
from tea_gaze.stats import correlation_matrix, summarize_column


def _md_table(headers: list[str], rows: list[list[object]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join("---" for _ in headers) + "|",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(cell) for cell in row) + " |")
    return "\n".join(lines)


def _feature_table(frame, title: str) -> str:
    gaze = fusion_feature_frame(frame)
    rows = []
    for name in CORE_FUSION_FEATURES:
        summary = summarize_column(name, gaze[name].tolist())
        rows.append(
            [
                name,
                f"{summary.mean:.3f}",
                f"{summary.stdev:.3f}",
                f"{summary.minimum:.3f}",
                f"{summary.maximum:.3f}",
            ]
        )
    corr = correlation_matrix({name: gaze[name].tolist() for name in CORE_FUSION_FEATURES})
    corr_rows = [
        [name] + [f"{corr[i][j]:.2f}" for j in range(5)]
        for i, name in enumerate(CORE_FUSION_FEATURES)
    ]
    return "\n".join(
        [
            f"### {title}",
            "",
            _md_table(["Feature", "mean", "stdev", "min", "max"], rows),
            "",
            _md_table([" "] + list(CORE_FUSION_FEATURES), corr_rows),
            "",
        ]
    )


def main() -> int:
    zuco = load_frame("zuco_sst_standard")
    full_train = load_frame("full_sst_train")
    full_valid = load_frame("full_sst_valid")
    full_test = load_frame("full_sst_test")

    sample = full_train.sample(6, random_state=7)[
        ["sentence_id", "sentence", "sentiment_label", "nFix", "FFD", "GPT", "TRT", "GD"]
    ]

    sample_rows = []
    for row in sample.itertuples(index=False):
        snippet = row.sentence if len(row.sentence) <= 90 else row.sentence[:87] + "..."
        sample_rows.append(
            [
                int(row.sentence_id),
                SENTIMENT_LABELS[int(row.sentiment_label)],
                snippet.replace("|", "/"),
                f"{row.nFix:.3f}",
                f"{row.FFD:.3f}",
            ]
        )

    report = "\n".join(
        [
            "# Full SST + predicted gaze",
            "",
            "`model_full_SST.py` trains on the large Stanford Sentiment Treebank "
            "split, not the 400 ZuCo sentences. Those larger tables store predicted "
            "gaze under the short header `nFix` (an alias of `nFixations`).",
            "",
            "## Split sizes and labels",
            "",
            "| Split | Rows | NEGATIVE | NEUTRAL | POSITIVE |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    split_lines = []
    for name, frame in (
        ("full train", full_train),
        ("full valid", full_valid),
        ("full test", full_test),
        ("ZuCo+SST (all)", zuco),
    ):
        counts = summarize_labels(frame.to_dict("records"))
        split_lines.append(
            f"| {name} | {len(frame)} | {counts[0]} | {counts[1]} | {counts[2]} |"
        )
    report = "\n".join(
        [
            report,
            "\n".join(split_lines),
            "",
            "## Why the two tables are not interchangeable",
            "",
            "- ZuCo+SST gaze is **measured** from 12 readers and then z-scored.",
            "- Full SST gaze is **predicted** (see `gaze_prediction/`) and written as `nFix`.",
            "- The full SST label mix is less balanced: Neutral is the minority class.",
            "- `model_full_SST.py` uses a 5-epoch train/valid/test loop and a larger batch size.",
            "",
            "## Random rows from `train_full_sst.csv`",
            "",
            _md_table(["id", "label", "sentence", "nFix", "FFD"], sample_rows),
            "",
            _feature_table(zuco, "Measured ZuCo+SST fusion features"),
            _feature_table(full_train.sample(800, random_state=1), "Predicted full-SST train sample (n=800)"),
            "ZuCo+SST label mix for reference:",
            "",
            markdown_label_table(summarize_labels(zuco.to_dict("records"))),
            "",
        ]
    )
    path = write_text(repo_root() / "examples" / "output" / "full_sst_sample.md", report)
    print(f"Wrote {path.relative_to(repo_root())}")
    print(f"full SST train/valid/test = {len(full_train)}/{len(full_valid)}/{len(full_test)}")
    print(f"ZuCo+SST rows = {len(zuco)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
