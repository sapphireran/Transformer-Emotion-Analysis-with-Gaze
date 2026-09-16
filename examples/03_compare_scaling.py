#!/usr/bin/env python3
"""Rebuild min-max and standard scaling from the raw reader-mean sentence file.

The committed files `min_max_scaled_average_data.csv` and
`standard_scaled_average_data.csv` should match these transforms up to
floating-point noise.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from examples._common import banner
from tea_gaze.features import compare_scaling, summarize_numeric
from tea_gaze.paths import DataPaths
from tea_gaze.reports import markdown_table
from tea_gaze.schema import ZUCO_SENTENCE_ET_COLUMNS


def main() -> None:
    paths = DataPaths.from_cwd()
    raw = pd.read_csv(paths.sentence_averages_raw)
    committed_std = pd.read_csv(paths.zuco_sentence_et / "standard_scaled_average_data.csv")
    committed_mm = pd.read_csv(paths.zuco_sentence_et / "min_max_scaled_average_data.csv")

    cols = [col for col in ZUCO_SENTENCE_ET_COLUMNS if col in raw.columns]
    views = compare_scaling(raw, cols)

    banner("Raw reader-mean sentence ET (average_data.csv)")
    print(markdown_table(summarize_numeric(views["raw"], cols).reset_index().rename(columns={"index": "feature"})))

    banner("Standard scaling rebuilt from the same file")
    print(markdown_table(summarize_numeric(views["standard"], cols).reset_index().rename(columns={"index": "feature"})))

    banner("Min-max scaling rebuilt from the same file")
    print(markdown_table(summarize_numeric(views["minmax"], cols).reset_index().rename(columns={"index": "feature"})))

    banner("Does the rebuild match the committed scaled CSVs?")
    for name, rebuilt, committed in (
        ("standard", views["standard"], committed_std),
        ("minmax", views["minmax"], committed_mm),
    ):
        overlap = [col for col in cols if col in committed.columns]
        delta = np.nanmax(np.abs(rebuilt[overlap].to_numpy() - committed[overlap].to_numpy()))
        print(f"  {name:10} max |rebuilt - committed| on {overlap} = {delta:.6g}")

    print()
    print("model_ZuCo_SST.py reads the already-joined standard table,")
    print("not these average_data files. This script only checks the scaler step.")


if __name__ == "__main__":
    main()
