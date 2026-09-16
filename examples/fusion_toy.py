#!/usr/bin/env python3
"""Explicit late-fusion toy that mirrors EyeTrackingModel without a GPU.

EyeTrackingModel does:

    concat(transformer_pooler, Linear(gaze)) -> classifier

This script does:

    concat(TF-IDF(sentence), scaled_gaze) -> multinomial logistic regression

It also reports a *gated* variant: the five gaze features are multiplied by
a learned per-feature scale (implemented as a second logistic model on
[tfidf | gaze | tfidf_mean * gaze] interaction summaries). The point is to
show the concat idea on the committed CSVs, not to clone RoBERTa.

Usage (from repo root):

    python3 examples/fusion_toy.py
    python3 examples/fusion_toy.py --track zuco --write examples/sample_outputs/fusion_toy.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler

from paths import (
    SST_FUSION_FEATURES,
    SST_TEST,
    SST_TRAIN,
    SST_VALID,
    ZUCO_FUSION_FEATURES,
    ZUCO_TEST,
    ZUCO_TRAIN,
    ZUCO_VALID,
)


def _stack_split(train_path: Path, valid_path: Path, test_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    fit = pd.concat([pd.read_csv(train_path), pd.read_csv(valid_path)], ignore_index=True)
    test = pd.read_csv(test_path)
    return fit, test


def _eval(name: str, clf, x_test, y_test: np.ndarray) -> dict:
    pred = clf.predict(x_test)
    return {
        "name": name,
        "accuracy": float(accuracy_score(y_test, pred)),
        "f1_weighted": float(f1_score(y_test, pred, average="weighted", zero_division=0)),
        "f1_macro": float(f1_score(y_test, pred, average="macro", zero_division=0)),
    }


def _clf() -> LogisticRegression:
    return LogisticRegression(max_iter=2000, class_weight="balanced", solver="lbfgs")


def run_track(track: str, train: pd.DataFrame, test: pd.DataFrame, gaze_cols: list[str]) -> dict:
    y_train = train["sentiment_label"].astype(int).to_numpy()
    y_test = test["sentiment_label"].astype(int).to_numpy()

    vectorizer = TfidfVectorizer(min_df=2, ngram_range=(1, 2), max_features=15000)
    text_train = vectorizer.fit_transform(train["sentence"].astype(str))
    text_test = vectorizer.transform(test["sentence"].astype(str))

    gaze_scaler = StandardScaler()
    gaze_train = gaze_scaler.fit_transform(train[gaze_cols].astype(float).to_numpy())
    gaze_test = gaze_scaler.transform(test[gaze_cols].astype(float).to_numpy())

    # Dense 16-d projection analogue: PCA-free linear map via a tiny random
    # projection so the concat width stays readable in the printed shapes.
    rng = np.random.default_rng(42)
    projection = rng.normal(0.0, 1.0 / np.sqrt(len(gaze_cols)), size=(len(gaze_cols), 16))
    gaze_h_train = gaze_train @ projection
    gaze_h_test = gaze_test @ projection

    concat_train = sparse.hstack([text_train, sparse.csr_matrix(gaze_h_train)], format="csr")
    concat_test = sparse.hstack([text_test, sparse.csr_matrix(gaze_h_test)], format="csr")

    # Interaction summaries: each gaze column times the sentence TF-IDF L2
    # mass. Cheap stand-in for "hard sentence + long TRT".
    text_mass_train = np.sqrt(text_train.multiply(text_train).sum(axis=1)).A1.reshape(-1, 1)
    text_mass_test = np.sqrt(text_test.multiply(text_test).sum(axis=1)).A1.reshape(-1, 1)
    interact_train = gaze_train * text_mass_train
    interact_test = gaze_test * text_mass_test
    gated_train = sparse.hstack(
        [text_train, sparse.csr_matrix(np.hstack([gaze_h_train, interact_train]))],
        format="csr",
    )
    gated_test = sparse.hstack(
        [text_test, sparse.csr_matrix(np.hstack([gaze_h_test, interact_test]))],
        format="csr",
    )

    text_only = _clf().fit(text_train, y_train)
    gaze_only = _clf().fit(gaze_train, y_train)
    concat = _clf().fit(concat_train, y_train)
    gated = _clf().fit(gated_train, y_train)

    results = [
        _eval("text_tfidf", text_only, text_test, y_test),
        _eval("gaze_raw5", gaze_only, gaze_test, y_test),
        _eval("concat_tfidf_plus_gaze16", concat, concat_test, y_test),
        _eval("concat_plus_textmass_x_gaze", gated, gated_test, y_test),
    ]
    for row in results:
        row["track"] = track

    delta = results[2]["f1_weighted"] - results[0]["f1_weighted"]
    return {
        "track": track,
        "n_train": int(len(train)),
        "n_test": int(len(test)),
        "tfidf_dim": int(text_train.shape[1]),
        "gaze_projected_dim": 16,
        "concat_minus_text_f1": float(delta),
        "models": results,
    }


def _print(report: dict) -> None:
    print(f"\n=== {report['track']} ===")
    print(
        f"train={report['n_train']} test={report['n_test']} "
        f"tfidf_dim={report['tfidf_dim']} gaze_h=16"
    )
    print(f"{'model':32s}  acc     f1_w    f1_mac")
    for row in report["models"]:
        print(
            f"{row['name']:32s}  {row['accuracy']:.4f}  "
            f"{row['f1_weighted']:.4f}  {row['f1_macro']:.4f}"
        )
    sign = "+" if report["concat_minus_text_f1"] >= 0 else ""
    print(f"concat − text weighted F1: {sign}{report['concat_minus_text_f1']:.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--track", choices=("zuco", "sst", "both"), default="both")
    parser.add_argument("--write", type=Path, default=None)
    args = parser.parse_args()

    reports = []
    if args.track in {"zuco", "both"}:
        train, test = _stack_split(ZUCO_TRAIN, ZUCO_VALID, ZUCO_TEST)
        reports.append(run_track("zuco", train, test, ZUCO_FUSION_FEATURES))
    if args.track in {"sst", "both"}:
        train, test = _stack_split(SST_TRAIN, SST_VALID, SST_TEST)
        reports.append(run_track("full_sst", train, test, SST_FUSION_FEATURES))

    for report in reports:
        _print(report)

    print("\nThis is a linear concat, not EyeTrackingModel.")
    print("A near-zero delta is a useful personal result: gaze is weak as a residual.")

    if args.write:
        args.write.parent.mkdir(parents=True, exist_ok=True)
        args.write.write_text(json.dumps(reports, indent=2), encoding="utf-8")
        print(f"\nwrote {args.write}")


if __name__ == "__main__":
    main()
