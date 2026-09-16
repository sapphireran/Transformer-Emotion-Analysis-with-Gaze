#!/usr/bin/env python3
"""5-fold majority vs logistic-regression-on-gaze for both corpora.

This is the personal sanity check I want next to any GPU fusion number.
If gaze-only logistic cannot beat majority, a concat head has to get its
gains from the interaction with the text encoder, not from the 5 floats
alone.
"""

from __future__ import annotations

import pandas as pd

from examples._common import banner
from tea_gaze.baselines import gaze_only_logistic_cv, majority_cv
from tea_gaze.io import load_full_sst_splits, load_zuco_combined
from tea_gaze.reports import markdown_table
from tea_gaze.schema import SST_MODEL_FEATURES, ZUCO_MODEL_FEATURES


def _mean_row(name: str, report) -> dict[str, object]:
    return {"setup": name, **report.mean.as_dict()}


def main() -> None:
    zuco = load_zuco_combined(scaling="standard").frame
    sst = load_full_sst_splits()["combined"].frame

    banner("ZuCo combined — 5-fold stratified, seed=42")
    z_maj = majority_cv(zuco)
    z_gaze = gaze_only_logistic_cv(zuco, ZUCO_MODEL_FEATURES)
    print(z_maj.pretty())
    print()
    print(z_gaze.pretty())
    print()
    print(
        markdown_table(
            pd.DataFrame(
                [
                    _mean_row("ZuCo majority", z_maj),
                    _mean_row("ZuCo gaze-only logistic", z_gaze),
                ]
            )
        )
    )

    banner("Full SST combined — 5-fold stratified, seed=42")
    s_maj = majority_cv(sst)
    s_gaze = gaze_only_logistic_cv(sst, SST_MODEL_FEATURES)
    print(s_maj.pretty())
    print()
    print(s_gaze.pretty())
    print()
    print(
        markdown_table(
            pd.DataFrame(
                [
                    _mean_row("SST majority", s_maj),
                    _mean_row("SST gaze-only logistic", s_gaze),
                ]
            )
        )
    )

    print()
    print("Features used:")
    print("  ZuCo:", ", ".join(ZUCO_MODEL_FEATURES))
    print("  SST: ", ", ".join(SST_MODEL_FEATURES))


if __name__ == "__main__":
    main()
