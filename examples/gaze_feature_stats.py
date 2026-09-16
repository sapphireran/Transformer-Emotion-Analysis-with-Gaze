#!/usr/bin/env python3
"""Class-conditional gaze statistics for the ZuCo and full-SST tracks.

Prints per-class means, Pearson correlations, and sklearn f_classif scores
so you can see whether any fusion column linearly separates sentiment
before you launch a transformer run.

Usage (from repo root):

    python3 examples/gaze_feature_stats.py
    python3 examples/gaze_feature_stats.py --write examples/sample_outputs/gaze_feature_stats.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_selection import f_classif

from paths import (
    LABEL_NAMES,
    SAMPLE_OUTPUT_DIR,
    SST_COMBINED,
    SST_FUSION_FEATURES,
    ZUCO_COMBINED_STANDARD,
    ZUCO_EXTRA_FEATURES,
    ZUCO_FUSION_FEATURES,
)


def _class_means(df: pd.DataFrame, features: list[str]) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for label, group in df.groupby("sentiment_label"):
        name = LABEL_NAMES.get(int(label), str(label))
        stats = {}
        for col in features:
            series = group[col].astype(float)
            stats[col] = {
                "n": int(series.notna().sum()),
                "mean": float(series.mean()),
                "std": float(series.std(ddof=1)) if len(series) > 1 else 0.0,
            }
        out[name] = stats
    return out


def _corr(df: pd.DataFrame, features: list[str]) -> dict[str, dict[str, float]]:
    corr = df[features].astype(float).corr(method="pearson")
    return {
        row: {col: float(corr.loc[row, col]) for col in features}
        for row in features
    }


def _f_scores(df: pd.DataFrame, features: list[str]) -> dict[str, dict[str, float]]:
    x = df[features].astype(float).to_numpy()
    y = df["sentiment_label"].astype(int).to_numpy()
    scores, pvalues = f_classif(x, y)
    return {
        feat: {"f": float(f), "p": float(p)}
        for feat, f, p in zip(features, scores, pvalues)
    }


def analyze(name: str, path: Path, features: list[str], extra: list[str] | None = None) -> dict:
    df = pd.read_csv(path)
    cols = list(features)
    if extra:
        cols = cols + [c for c in extra if c in df.columns]
    payload = {
        "track": name,
        "path": str(path),
        "n": int(len(df)),
        "label_counts": {
            LABEL_NAMES.get(int(k), str(k)): int(v)
            for k, v in df["sentiment_label"].value_counts().sort_index().items()
        },
        "class_means": _class_means(df, cols),
        "pearson": _corr(df, features),
        "f_classif": _f_scores(df, features),
    }
    if extra:
        present = [c for c in extra if c in df.columns]
        if present:
            payload["extra_f_classif"] = _f_scores(df, present)
            if "omissionRate" in df.columns:
                payload["omission_by_class"] = {
                    LABEL_NAMES.get(int(k), str(k)): float(g["omissionRate"].mean())
                    for k, g in df.groupby("sentiment_label")
                }
    if "sentence" in df.columns:
        lengths = df.assign(n_chars=df["sentence"].astype(str).str.len())
        payload["chars_by_class"] = {
            LABEL_NAMES.get(int(k), str(k)): float(g["n_chars"].mean())
            for k, g in lengths.groupby("sentiment_label")
        }
    return payload


def _print_track(payload: dict) -> None:
    print(f"\n=== {payload['track']} (n={payload['n']}) ===")
    print("labels:", payload["label_counts"])
    print("class means (fusion + extras):")
    for label, feats in payload["class_means"].items():
        parts = [
            f"{col}={vals['mean']:+.3f}±{vals['std']:.3f}"
            for col, vals in feats.items()
        ]
        print(f"  {label}: " + "; ".join(parts))
    print("f_classif (higher => more label-associated, uncorrected p):")
    ranked = sorted(payload["f_classif"].items(), key=lambda kv: kv[1]["f"], reverse=True)
    for feat, stats in ranked:
        print(f"  {feat:12s}  F={stats['f']:.3f}  p={stats['p']:.4g}")
    if "extra_f_classif" in payload:
        print("extra columns:")
        for feat, stats in payload["extra_f_classif"].items():
            print(f"  {feat:12s}  F={stats['f']:.3f}  p={stats['p']:.4g}")
    print("Pearson |r| among fusion features:")
    features = list(payload["pearson"])
    for i, a in enumerate(features):
        for b in features[i + 1 :]:
            r = payload["pearson"][a][b]
            if abs(r) >= 0.5:
                print(f"  {a}–{b}: {r:+.3f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", type=Path, default=None, help="optional JSON dump")
    args = parser.parse_args()

    reports = [
        analyze("zuco_standard", ZUCO_COMBINED_STANDARD, ZUCO_FUSION_FEATURES, ZUCO_EXTRA_FEATURES),
        analyze("full_sst", SST_COMBINED, SST_FUSION_FEATURES),
    ]
    for report in reports:
        _print_track(report)

    print("\n=== reading notes ===")
    print("TRT/GPT correlations are expected (both include re-reading).")
    print("A high F on a small ZuCo class can be a sample-size artifact.")
    print("Full-SST gaze is projected, so class differences may be weaker.")

    if args.write:
        args.write.parent.mkdir(parents=True, exist_ok=True)
        args.write.write_text(json.dumps(reports, indent=2), encoding="utf-8")
        print(f"\nwrote {args.write}")
    else:
        SAMPLE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    np.set_printoptions(precision=3, suppress=True)
    main()
