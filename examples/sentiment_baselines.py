#!/usr/bin/env python3
"""CPU sentiment baselines on the committed ZuCo and full-SST tables.

Compares four feature sets with logistic regression:

- gaze: the five fusion columns only
- length: character length (a dummy complexity covariate)
- tfidf: word TF-IDF on the sentence string
- tfidf+gaze: horizontal concat of TF-IDF and the five gaze columns

ZuCo uses the committed train/valid/test split. Full SST uses the committed
full-SST split. Metrics are accuracy and weighted F1 plus a classification
report on the test slice.

This is the floor you want before spending a GPU hour on RoBERTa.

Usage (from repo root):

    python3 examples/sentiment_baselines.py
    python3 examples/sentiment_baselines.py --track zuco
    python3 examples/sentiment_baselines.py --write examples/sample_outputs/baselines.json
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
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import StandardScaler

from paths import (
    LABEL_NAMES,
    SST_FUSION_FEATURES,
    SST_TEST,
    SST_TRAIN,
    SST_VALID,
    ZUCO_FUSION_FEATURES,
    ZUCO_TEST,
    ZUCO_TRAIN,
    ZUCO_VALID,
)

TARGET_NAMES = [LABEL_NAMES[i] for i in range(3)]


def _load_split(train_path: Path, valid_path: Path, test_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    train = pd.read_csv(train_path)
    valid = pd.read_csv(valid_path)
    test = pd.read_csv(test_path)
    # Fit on train+valid so the small ZuCo valid slice is not wasted on a
    # linear model that has no early stopping. Evaluate on test only.
    fit = pd.concat([train, valid], ignore_index=True)
    return fit, test


def _gaze_matrix(df: pd.DataFrame, columns: list[str]) -> np.ndarray:
    return df[columns].astype(float).to_numpy()


def _length_matrix(df: pd.DataFrame) -> np.ndarray:
    return df["sentence"].astype(str).str.len().to_numpy().reshape(-1, 1)


def _fit_eval(
    name: str,
    x_train,
    y_train: np.ndarray,
    x_test,
    y_test: np.ndarray,
) -> dict:
    clf = LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        solver="lbfgs",
    )
    clf.fit(x_train, y_train)
    pred = clf.predict(x_test)
    report = classification_report(
        y_test,
        pred,
        target_names=TARGET_NAMES,
        output_dict=True,
        zero_division=0,
    )
    payload = {
        "name": name,
        "accuracy": float(accuracy_score(y_test, pred)),
        "f1_weighted": float(f1_score(y_test, pred, average="weighted", zero_division=0)),
        "f1_macro": float(f1_score(y_test, pred, average="macro", zero_division=0)),
        "per_class_f1": {label: float(report[label]["f1-score"]) for label in TARGET_NAMES},
        "support": {label: int(report[label]["support"]) for label in TARGET_NAMES},
    }
    return payload


def _run_track(track: str, train: pd.DataFrame, test: pd.DataFrame, gaze_cols: list[str]) -> list[dict]:
    y_train = train["sentiment_label"].astype(int).to_numpy()
    y_test = test["sentiment_label"].astype(int).to_numpy()

    gaze_scaler = StandardScaler()
    gaze_train = gaze_scaler.fit_transform(_gaze_matrix(train, gaze_cols))
    gaze_test = gaze_scaler.transform(_gaze_matrix(test, gaze_cols))

    length_scaler = StandardScaler()
    length_train = length_scaler.fit_transform(_length_matrix(train))
    length_test = length_scaler.transform(_length_matrix(test))

    vectorizer = TfidfVectorizer(min_df=2, ngram_range=(1, 2), max_features=20000)
    tfidf_train = vectorizer.fit_transform(train["sentence"].astype(str))
    tfidf_test = vectorizer.transform(test["sentence"].astype(str))

    fused_train = sparse.hstack([tfidf_train, sparse.csr_matrix(gaze_train)], format="csr")
    fused_test = sparse.hstack([tfidf_test, sparse.csr_matrix(gaze_test)], format="csr")

    results = [
        _fit_eval("gaze", gaze_train, y_train, gaze_test, y_test),
        _fit_eval("length", length_train, y_train, length_test, y_test),
        _fit_eval("tfidf", tfidf_train, y_train, tfidf_test, y_test),
        _fit_eval("tfidf+gaze", fused_train, y_train, fused_test, y_test),
    ]
    for row in results:
        row["track"] = track
        row["n_train"] = int(len(train))
        row["n_test"] = int(len(test))
    return results


def _print_results(rows: list[dict]) -> None:
    current = None
    for row in rows:
        if row["track"] != current:
            current = row["track"]
            print(f"\n=== {current} (train={row['n_train']} test={row['n_test']}) ===")
            print(f"{'model':12s}  acc     f1_w    f1_mac  neg    neu    pos")
        print(
            f"{row['name']:12s}  "
            f"{row['accuracy']:.4f}  {row['f1_weighted']:.4f}  {row['f1_macro']:.4f}  "
            f"{row['per_class_f1']['negative']:.3f}  "
            f"{row['per_class_f1']['neutral']:.3f}  "
            f"{row['per_class_f1']['positive']:.3f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--track", choices=("zuco", "sst", "both"), default="both")
    parser.add_argument("--write", type=Path, default=None)
    args = parser.parse_args()

    rows: list[dict] = []
    if args.track in {"zuco", "both"}:
        train, test = _load_split(ZUCO_TRAIN, ZUCO_VALID, ZUCO_TEST)
        rows.extend(_run_track("zuco", train, test, ZUCO_FUSION_FEATURES))
    if args.track in {"sst", "both"}:
        train, test = _load_split(SST_TRAIN, SST_VALID, SST_TEST)
        rows.extend(_run_track("full_sst", train, test, SST_FUSION_FEATURES))

    _print_results(rows)
    print("\nNotes:")
    print("- gaze-only near chance means fusion should be a residual, not a driver.")
    print("- tfidf+gaze vs tfidf is the linear analogue of EyeTrackingModel.")
    print("- ZuCo test is ~40 rows; treat per-class F1 as noisy.")

    if args.write:
        args.write.parent.mkdir(parents=True, exist_ok=True)
        args.write.write_text(json.dumps(rows, indent=2), encoding="utf-8")
        print(f"\nwrote {args.write}")


if __name__ == "__main__":
    main()
