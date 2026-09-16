#!/usr/bin/env python3
"""Write the analysis tables used by the other examples as CSV/JSON."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from examples.lib.gaze import (
    FULL_SST_GAZE,
    ZUCO_ALL_GAZE,
    feature_label_anova,
    inter_subject_cv,
    per_class_means,
    sentence_skip_rate,
    skip_rate_by_word_length,
    subject_feature_panel,
)
from examples.lib.loading import (
    documented_datasets,
    load_dataset,
    load_full_sst_splits,
    load_subject_sentence_tables,
    load_zuco_combined,
    load_zuco_word_average,
)
from examples.lib.metrics import majority_baseline
from examples.lib.paths import resolve_root
from examples.lib.reporting import banner

import pandas as pd


def _write(df, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=True)
    print(f"wrote {path}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=None)
    parser.add_argument(
        "--output-dir",
        default="examples/output",
        help="Directory for CSV/JSON products (gitignored by default).",
    )
    args = parser.parse_args()
    root = resolve_root(args.root)
    out = Path(args.output_dir)
    if not out.is_absolute():
        out = root / out

    sst = load_dataset(
        next(s for s in documented_datasets() if s.key == "sst_combined"), root=root
    )
    zuco = load_zuco_combined(root=root)
    words = load_zuco_word_average(root=root)
    splits = load_full_sst_splits(root=root)
    panel = subject_feature_panel(load_subject_sentence_tables(root=root))

    banner(f"Exporting tables under {out}")
    _write(per_class_means(sst, FULL_SST_GAZE), out / "sst_per_class_gaze_means.csv")
    _write(per_class_means(zuco, ZUCO_ALL_GAZE), out / "zuco_per_class_gaze_means.csv")
    _write(feature_label_anova(sst, FULL_SST_GAZE), out / "sst_gaze_anova.csv")
    _write(feature_label_anova(zuco, ZUCO_ALL_GAZE), out / "zuco_gaze_anova.csv")
    _write(skip_rate_by_word_length(words), out / "zuco_skip_rate_by_wordlen.csv")
    _write(sentence_skip_rate(words), out / "zuco_sentence_skip_rate.csv")

    cv = inter_subject_cv(panel, ZUCO_ALL_GAZE)
    cv_summary = (
        cv.groupby("feature", observed=True)["subject_cv"]
        .agg(mean_cv="mean", median_cv="median")
        .sort_values("mean_cv", ascending=False)
    )
    _write(cv_summary, out / "zuco_inter_subject_cv.csv")

    split_counts = []
    for name, df in {"combined": sst, **splits}.items():
        counts = df["sentiment_label"].value_counts().sort_index()
        split_counts.append(
            {
                "split": name,
                "n": int(len(df)),
                "negative": int(counts.get(0, 0)),
                "neutral": int(counts.get(1, 0)),
                "positive": int(counts.get(2, 0)),
            }
        )
    _write(pd.DataFrame(split_counts).set_index("split"), out / "label_counts.csv")

    summary = {
        "sst_majority_baseline": majority_baseline(sst["sentiment_label"]),
        "zuco_majority_baseline": majority_baseline(zuco["sentiment_label"]),
        "zuco_word_skip_rate": float(words["nFixations"].fillna(0).eq(0).mean()),
        "output_dir": str(out),
    }
    summary_path = out / "export_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"wrote {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
