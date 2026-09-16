#!/usr/bin/env python3
"""Sentence-level ET stats and their linear association with polarity."""

from __future__ import annotations

import sys
from pathlib import Path

_EXAMPLES = Path(__file__).resolve().parent
if str(_EXAMPLES) not in sys.path:
    sys.path.insert(0, str(_EXAMPLES))

import numpy as np
import pandas as pd

from paths import (
    FULL_SST_TRAIN,
    SENTIMENT_NAME,
    SST_ET_COLS,
    ZUCO_ET_COLS,
    ZUCO_STANDARD,
)


def _pearson(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.std() == 0 or y.std() == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def report(name: str, df: pd.DataFrame, feature_cols: list[str], label_col: str) -> None:
    print(f"=== {name}  (n={len(df)}) ===")
    labels = df[label_col].astype(int)
    print("label counts:", ", ".join(
        f"{k} {SENTIMENT_NAME[k]}={int((labels == k).sum())}" for k in (0, 1, 2)
        if k in set(labels)
    ))
    print()
    header = f"{'feature':<14} {'mean':>10} {'std':>10} {'min':>10} {'max':>10} {'r(label)':>10}"
    print(header)
    print("-" * len(header))
    for col in feature_cols:
        series = df[col].astype(float)
        r = _pearson(series.to_numpy(), labels.to_numpy())
        print(
            f"{col:<14} {series.mean():10.4f} {series.std():10.4f} "
            f"{series.min():10.4f} {series.max():10.4f} {r:10.4f}"
        )
    print()
    print("per-class feature means")
    means = df.groupby(label_col)[feature_cols].mean()
    means.index = [f"{int(i)} {SENTIMENT_NAME.get(int(i), str(i))}" for i in means.index]
    print(means.to_string(float_format=lambda v: f"{v:8.4f}"))
    print()

    # Pairwise ET correlations — useful because TRT tracks nFix and GD tracks FFD.
    corr = df[feature_cols].corr()
    print("pairwise feature correlations")
    print(corr.to_string(float_format=lambda v: f"{v:8.3f}"))
    print()


def main() -> int:
    zuco = pd.read_csv(ZUCO_STANDARD)
    report(
        "ZuCo ∩ SST  (combined_sst_et_standard.csv, recorded gaze)",
        zuco,
        ZUCO_ET_COLS,
        "sentiment_label",
    )

    extra = [c for c in ("omissionRate", "meanPupilSize", "SFD") if c in zuco.columns]
    if extra:
        report(
            "ZuCo ∩ SST  extra columns not fed to EyeTrackingModel",
            zuco,
            extra,
            "sentiment_label",
        )

    sst = pd.read_csv(FULL_SST_TRAIN)
    report(
        "full SST train  (train_full_sst.csv, projected gaze)",
        sst,
        SST_ET_COLS,
        "sentiment_label",
    )
    print(
        "r(label) is Pearson correlation against the integer 0/1/2 label. "
        "A value near 0 means the feature has little *linear* association "
        "with polarity; the transformer could still use it non-linearly."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
