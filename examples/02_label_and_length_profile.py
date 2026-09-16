#!/usr/bin/env python3
"""Label balance, sentence length, and a majority-class baseline."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from examples.lib.loading import (
    LABEL_NAMES,
    load_dataset,
    load_full_sst_splits,
    load_zuco_combined,
    documented_datasets,
)
from examples.lib.metrics import majority_baseline
from examples.lib.paths import resolve_root
from examples.lib.reporting import banner, print_frame

import pandas as pd


def _profile(df: pd.DataFrame, name: str) -> pd.DataFrame:
    tokens = df["sentence"].astype(str).str.split().str.len()
    counts = df["sentiment_label"].value_counts().sort_index()
    rows = []
    for label, count in counts.items():
        rows.append(
            {
                "split": name,
                "label": int(label),
                "class": LABEL_NAMES[int(label)],
                "count": int(count),
                "share": float(count / len(df)),
            }
        )
    summary = pd.DataFrame(rows)
    summary.attrs["n"] = len(df)
    summary.attrs["token_min"] = int(tokens.min())
    summary.attrs["token_mean"] = float(tokens.mean())
    summary.attrs["token_max"] = int(tokens.max())
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=None)
    args = parser.parse_args()
    root = resolve_root(args.root)

    splits = load_full_sst_splits(root=root)
    combined_spec = next(s for s in documented_datasets() if s.key == "sst_combined")
    combined = load_dataset(combined_spec, root=root)
    zuco = load_zuco_combined(root=root)

    frames = [
        _profile(combined, "sst_combined"),
        *(_profile(df, f"sst_{name}") for name, df in splits.items()),
        _profile(zuco, "zuco_standard"),
    ]

    banner("Label counts")
    print_frame(pd.concat(frames, ignore_index=True))

    banner("Whitespace-token length")
    length_rows = []
    for frame in frames:
        length_rows.append(
            {
                "split": frame["split"].iloc[0],
                "n": frame.attrs["n"],
                "token_min": frame.attrs["token_min"],
                "token_mean": round(frame.attrs["token_mean"], 2),
                "token_max": frame.attrs["token_max"],
            }
        )
    print_frame(pd.DataFrame(length_rows))

    banner("Majority-class baseline (weighted scores)")
    baselines = []
    for name, df in (("sst_combined", combined), ("zuco_standard", zuco)):
        scores = majority_baseline(df["sentiment_label"])
        scores["split"] = name
        scores["majority_class_name"] = LABEL_NAMES[int(scores["majority_class"])]
        baselines.append(scores)
    print_frame(pd.DataFrame(baselines))
    print(
        "\nA transformer that cannot beat the majority baseline is not "
        "learning polarity. On full SST the majority class is positive; "
        "on ZuCo 400 it is nearly even, so the floor is close to 1/3."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
