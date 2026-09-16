#!/usr/bin/env python3
"""Rebuild the text⨝gaze join and the min-max / z-score fingerprints."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from gazebook.csvio import read_dicts, table_to_array
from gazebook.paths import repo_root
from gazebook.reports import write_text
from gazebook.schema import SENTENCE_RAW
from gazebook.stats import minmax, zscore


def main() -> int:
    root = repo_root()
    _, text = read_dicts(root / "ZuCo_SST_data/ssts_ZuCo.csv")
    _, scaled = read_dicts(root / "ZuCo_et_csv_data/standard_scaled_average_data.csv")
    _, combined = read_dicts(root / "ZuCo_SST_data/combined_sst_et_standard.csv")
    _, raw = read_dicts(root / "ZuCo_et_csv_data/average_data.csv")
    _, mm = read_dicts(root / "ZuCo_et_csv_data/min_max_scaled_average_data.csv")
    _, st = read_dicts(root / "ZuCo_et_csv_data/standard_scaled_average_data.csv")

    text_by = {int(r["sentence_id"]): r for r in text}
    scaled_by = {int(r["id"]): r for r in scaled}

    join_err = 0.0
    text_mismatches = 0
    for row in combined:
        sid = int(row["sentence_id"])
        if text_by[sid]["sentence"] != row["sentence"]:
            text_mismatches += 1
        if text_by[sid]["sentiment_label"] != row["sentiment_label"]:
            text_mismatches += 1
        join_err = max(join_err, abs(float(row["nFixations"]) - float(scaled_by[sid]["nFixations"])))

    R = table_to_array(raw, SENTENCE_RAW)
    M = table_to_array(mm, SENTENCE_RAW)
    S = table_to_array(st, SENTENCE_RAW)
    mm_err = float(np.max(np.abs(minmax(R) - M)))
    z0_err = float(np.max(np.abs(zscore(R, ddof=0) - S)))
    z1_err = float(np.max(np.abs(zscore(R, ddof=1) - S)))

    print(f"combined_sst_et_standard text mismatches vs ssts_ZuCo: {text_mismatches}")
    print(f"combined nFixations vs standard_scaled_average_data max|Δ|: {join_err:.3e}")
    print(f"min-max rebuild max|Δ|: {mm_err:.3e}")
    print(f"z-score rebuild max|Δ| (ddof=0 / population): {z0_err:.3e}")
    print(f"z-score rebuild max|Δ| (ddof=1 / sample):     {z1_err:.3e}")
    print("The committed standard table is sklearn-style population scaling (ddof=0).")

    write_text(
        root / "examples/output/06_join_and_scalers.md",
        "\n".join(
            [
                "# Join and scalers",
                "",
                f"text mismatches: {text_mismatches}",
                f"join nFix max|Δ|: {join_err:.3e}",
                f"minmax max|Δ|: {mm_err:.3e}",
                f"z-score ddof0 max|Δ|: {z0_err:.3e}",
                f"z-score ddof1 max|Δ|: {z1_err:.3e}",
                "",
            ]
        ),
    )
    if text_mismatches or join_err > 1e-12 or mm_err > 1e-12 or z0_err > 1e-12:
        return 1
    if z1_err < 1e-6:
        # Sample std should *not* reconstruct the published table.
        print("ddof=1 unexpectedly matched; scaler assumption changed", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
