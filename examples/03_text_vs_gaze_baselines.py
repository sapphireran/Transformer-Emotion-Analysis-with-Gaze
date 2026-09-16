#!/usr/bin/env python3
"""Compare text-only, gaze-only, and early-fusion logistic baselines."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tea_gaze.baselines import gaze_only_coefficients, run_suite
from tea_gaze.features import CORE_FUSION_FEATURES, SENTIMENT_LABELS
from tea_gaze.io import load_frame
from tea_gaze.paths import repo_root
from tea_gaze.reports import html_baselines, markdown_baseline_suite, write_text


def main() -> int:
    frame = load_frame("zuco_sst_standard")
    results = run_suite(frame, n_splits=5, random_state=42)
    coef = gaze_only_coefficients(frame)
    out_dir = repo_root() / "examples" / "output"

    notes = [
        "Text model: TF-IDF unigrams+bigrams (min_df=2, 4000 features) + balanced logistic regression.",
        "Gaze model: the five fusion columns (`nFixations`, `FFD`, `GPT`, `TRT`, `GD`) after StandardScaler.",
        "Fusion model: sparse TF-IDF horizontally stacked with the five scaled gaze columns.",
        "Splits match `model_ZuCo_SST.py`: StratifiedKFold(n_splits=5, shuffle=True, random_state=42).",
        "This is a CPU documentation baseline, not the BERT/RoBERTa experiment.",
    ]
    md = markdown_baseline_suite(
        results,
        title="Text vs gaze vs fusion on ZuCo+SST",
        extra_notes=notes,
    )
    coef_lines = [
        "",
        "## Gaze-only logistic coefficients (fit on all 400 rows)",
        "",
        "Each class has one weight per fusion feature. Positive weight means a higher "
        "feature value pushes the model toward that class.",
        "",
        "| Feature | NEGATIVE | NEUTRAL | POSITIVE |",
        "|---|---:|---:|---:|",
    ]
    class_index = {label: i for i, label in enumerate(coef["classes"])}
    for feature_i, name in enumerate(CORE_FUSION_FEATURES):
        cells = [
            f"{coef['coef'][class_index[label]][feature_i]:+.3f}"
            for label in SENTIMENT_LABELS
        ]
        coef_lines.append(f"| `{name}` | " + " | ".join(cells) + " |")
    md = md.rstrip() + "\n" + "\n".join(coef_lines) + "\n"

    md_path = write_text(out_dir / "baseline_comparison.md", md)
    html_path = write_text(out_dir / "baseline_comparison.html", html_baselines(results))
    json_path = write_text(
        out_dir / "baseline_comparison.json",
        json.dumps(
            {
                "models": [
                    {
                        "name": result.name,
                        "mean": result.mean.as_dict(),
                        "folds": [fold.metrics.as_dict() for fold in result.folds],
                    }
                    for result in results
                ],
                "gaze_coefficients": coef,
            },
            indent=2,
        )
        + "\n",
    )

    print(f"Wrote {md_path.relative_to(repo_root())}")
    print(f"Wrote {html_path.relative_to(repo_root())}")
    print(f"Wrote {json_path.relative_to(repo_root())}")
    print()
    for result in results:
        print(result.mean.format_line(result.name))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
