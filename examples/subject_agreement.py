#!/usr/bin/env python3
"""How much the twelve ZuCo readers disagree on the same sentence.

The training tables average subjects first. That hides cases where one
reader skipped a sentence that others fixated heavily. This script loads
`ZuCo_et_csv_data/{1-12}_SR.csv`, aligns on `id`, and reports:

- per-feature coefficient of variation across subjects
- sentences with the highest TRT disagreement
- the correlation of each subject against the leave-one-out mean

Usage (from repo root):

    python3 examples/subject_agreement.py
    python3 examples/subject_agreement.py --write examples/sample_outputs/subject_agreement.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from paths import ZUCO_SUBJECT_DIR

FEATURES = ["nFixations", "FFD", "GPT", "TRT", "GD", "omissionRate"]


def _load_subjects() -> dict[int, pd.DataFrame]:
    frames = {}
    for path in sorted(ZUCO_SUBJECT_DIR.glob("[0-9]*_SR.csv")):
        subject = int(path.stem.split("_")[0])
        df = pd.read_csv(path)
        if "id" not in df.columns:
            raise ValueError(f"{path} has no id column")
        frames[subject] = df.set_index("id").sort_index()
    return frames


def _stack(frames: dict[int, pd.DataFrame], feature: str) -> pd.DataFrame:
    series = []
    for subject, df in frames.items():
        col = pd.to_numeric(df[feature], errors="coerce")
        col.name = subject
        series.append(col)
    return pd.concat(series, axis=1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", type=Path, default=None)
    parser.add_argument("--top", type=int, default=8)
    args = parser.parse_args()

    frames = _load_subjects()
    subjects = sorted(frames)
    print(f"loaded {len(subjects)} subjects: {subjects}")

    feature_reports = {}
    for feature in FEATURES:
        mat = _stack(frames, feature)
        mean = mat.mean(axis=1)
        std = mat.std(axis=1, ddof=1)
        cv = (std / mean.replace(0, np.nan)).abs()
        feature_reports[feature] = {
            "n_sentences": int(len(mat)),
            "mean_across_sentences_of_subject_mean": float(mean.mean()),
            "mean_subject_std": float(std.mean()),
            "median_cv": float(cv.median(skipna=True)),
            "high_cv_ids": [int(i) for i in cv.nlargest(args.top).index],
        }

        loo_corr = {}
        for subject in subjects:
            others = mat.drop(columns=[subject]).mean(axis=1)
            loo_corr[str(subject)] = float(mat[subject].corr(others))
        feature_reports[feature]["loo_corr"] = loo_corr

    print("\n=== cross-subject spread ===")
    print(f"{'feature':14s}  mean(μ)   mean(σ)  median CV")
    for feature, report in feature_reports.items():
        print(
            f"{feature:14s}  {report['mean_across_sentences_of_subject_mean']:8.3f}  "
            f"{report['mean_subject_std']:8.3f}  {report['median_cv']:8.3f}"
        )

    trt = _stack(frames, "TRT")
    trt_std = trt.std(axis=1, ddof=1)
    print(f"\n=== top {args.top} sentences by TRT subject-std ===")
    for sent_id, value in trt_std.nlargest(args.top).items():
        row = trt.loc[sent_id]
        print(
            f"id={int(sent_id):4d}  std={value:7.2f}  "
            f"min={row.min():7.1f} max={row.max():7.1f} mean={row.mean():7.1f}"
        )

    print("\n=== leave-one-out correlation vs other-subject mean (TRT) ===")
    for subject, corr in feature_reports["TRT"]["loo_corr"].items():
        print(f"  subject {int(subject):2d}: r={corr:.3f}")

    print("\nHigh median CV means the twelve-subject mean is a blurry sentence vector.")
    print("A low-r subject is a candidate to drop before you re-average.")

    payload = {"subjects": subjects, "features": feature_reports}
    if args.write:
        args.write.parent.mkdir(parents=True, exist_ok=True)
        args.write.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"\nwrote {args.write}")


if __name__ == "__main__":
    main()
