#!/usr/bin/env python3
"""Compare PROVO reference gaze with predicted SST word-level gaze."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from examples.lib.loading import documented_datasets, load_dataset
from examples.lib.paths import resolve_root
from examples.lib.reporting import banner, print_frame

import pandas as pd


SHARED = ("nFix", "FFD", "GPT", "TRT")


def _describe(df: pd.DataFrame, columns: tuple[str, ...], name: str) -> pd.DataFrame:
    desc = df[list(columns)].describe().T
    desc = desc.rename(columns={"50%": "median"})
    desc = desc[["count", "mean", "std", "min", "median", "max"]]
    desc.insert(0, "source", name)
    desc.index.name = "feature"
    return desc.reset_index()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=None)
    args = parser.parse_args()
    root = resolve_root(args.root)

    by_key = {spec.key: spec for spec in documented_datasets()}
    provo = load_dataset(by_key["provo"], root=root)
    pred = load_dataset(by_key["pred_v2"], root=root)
    extract = load_dataset(by_key["pred_extract"], root=root)

    banner("Schema")
    print(f"PROVO columns:           {list(provo.columns)}")
    print(f"prediction_test_v2 cols: {list(pred.columns)}")
    print(f"prediction_test cols:    {list(extract.columns)}")
    print(
        "PROVO has fixProp (percent of readers who fixated the word). "
        "The SST predictions have GD instead. Do not concat these tables."
    )

    banner("Shared-feature distributions")
    table = pd.concat(
        [
            _describe(provo, SHARED, "provo"),
            _describe(pred, SHARED, "sst_predicted_v2"),
        ],
        ignore_index=True,
    )
    print_frame(table)

    banner("Within-source Pearson correlations (nFix, FFD, GPT, TRT)")
    print("PROVO")
    print_frame(provo[list(SHARED)].corr())
    print("\nSST predicted v2")
    print_frame(pred[list(SHARED)].corr())

    banner("Coverage")
    print(f"PROVO words:              {len(provo):7d}  sentences: {provo['sentence_id'].nunique()}")
    print(f"SST predicted v2 words:   {len(pred):7d}  sentences: {pred['sentence_id'].nunique()}")
    print(f"SST predicted extract:    {len(extract):7d}  sentences: {extract['sentence_id'].nunique()}")
    print(f"extract sentence_id min:  {int(extract['sentence_id'].min())}")
    print(
        "\nresult/*.png is the picture version of these correlations: "
        "predicted durations sit on a near-linear ridge with nFix. "
        "That is a property of the predictor, not of ZuCo."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
