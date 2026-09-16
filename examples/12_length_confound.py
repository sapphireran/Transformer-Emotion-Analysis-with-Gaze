#!/usr/bin/env python3
"""Is gaze just sentence length in disguise?

nFix / TRT / GPT all grow with how much there is to read. If polarity
classes differ in length, ANOVA on raw gaze will look like an emotion
effect. This script:

1. Correlates each fused feature with whitespace token count (and ZuCo
   SentLen where present).
2. Residualizes each feature on length and reruns ANOVA.
3. Prints per-class mean length so the confound is visible.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from examples.lib.baselines import residualize_on_length
from examples.lib.gaze import (
    FULL_SST_GAZE,
    ZUCO_FUSION_GAZE,
    feature_label_anova,
    length_feature_correlation,
    per_class_means,
)
from examples.lib.loading import (
    LABEL_NAMES,
    documented_datasets,
    load_dataset,
    load_zuco_combined,
)
from examples.lib.paths import resolve_root
from examples.lib.reporting import banner, print_frame

import pandas as pd


def _length_by_class(df: pd.DataFrame, length: pd.Series, name: str) -> pd.DataFrame:
    tmp = pd.DataFrame({"sentiment_label": df["sentiment_label"], "length": length})
    out = tmp.groupby("sentiment_label", observed=True)["length"].agg(
        mean="mean", median="median", n="size"
    )
    out.index = out.index.map(lambda i: LABEL_NAMES.get(int(i), str(i)))
    out.insert(0, "track", name)
    return out


def _block(title: str, df: pd.DataFrame, features, length: pd.Series) -> None:
    banner(f"{title}: per-class length")
    print_frame(_length_by_class(df, length, title))
    banner(f"{title}: gaze vs length (Pearson r)")
    print_frame(length_feature_correlation(df, features, length))
    banner(f"{title}: ANOVA before residualizing")
    print_frame(feature_label_anova(df, features))
    residual = residualize_on_length(df, features, length)
    banner(f"{title}: ANOVA after removing linear length")
    print_frame(feature_label_anova(residual, features))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=None)
    args = parser.parse_args()
    root = resolve_root(args.root)

    sst = load_dataset(
        next(s for s in documented_datasets() if s.key == "sst_combined"), root=root
    )
    sst_len = sst["sentence"].astype(str).str.split().str.len()
    zuco = load_zuco_combined(root=root)
    zuco_len = zuco["sentence"].astype(str).str.split().str.len()

    _block("Full SST (predicted gaze)", sst, FULL_SST_GAZE, sst_len)
    _block("ZuCo 400 (measured gaze)", zuco, ZUCO_FUSION_GAZE, zuco_len)

    print(
        "\nA feature whose ANOVA F collapses after residualizing was mostly "
        "length. A feature that keeps F is a better candidate for the fusion "
        "vector. Predicted full-SST gaze is almost a length proxy (see the "
        "near-1 correlations among nFix/FFD/GPT/TRT in example 03)."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
