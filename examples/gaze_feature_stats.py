#!/usr/bin/env python3
"""Summarise measured vs projected gaze columns.

Prints min / mean / max, a Pearson correlation matrix, and a reminder
that full-SST numbers are not milliseconds.

Usage:
    python3 examples/gaze_feature_stats.py
    python3 examples/gaze_feature_stats.py --out examples/output
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from examples.lib.loaders import FUSION_GAZE_COLUMNS, fusion_frame, load_csv, load_zuco_experiment


RAW_COLS = [
    "SentLen",
    "omissionRate",
    "nFixations",
    "meanPupilSize",
    "GD",
    "TRT",
    "FFD",
    "SFD",
    "GPT",
]


def _describe(df: pd.DataFrame, columns: list[str], title: str) -> str:
    lines = [title, "=" * len(title)]
    header = f"{'column':16} {'min':>12} {'mean':>12} {'std':>12} {'max':>12} {'zeros':>8}"
    lines.append(header)
    lines.append("-" * len(header))
    for col in columns:
        if col not in df.columns:
            continue
        s = pd.to_numeric(df[col], errors="coerce")
        lines.append(
            f"{col:16} {s.min():12.4f} {s.mean():12.4f} {s.std():12.4f} "
            f"{s.max():12.4f} {(s == 0).sum():8d}"
        )
    return "\n".join(lines)


def _corr(df: pd.DataFrame, columns: list[str], title: str) -> str:
    present = [c for c in columns if c in df.columns]
    matrix = df[present].apply(pd.to_numeric, errors="coerce").corr()
    lines = [title, "=" * len(title)]
    header = f"{'':16}" + "".join(f"{c:>10}" for c in present)
    lines.append(header)
    for row in present:
        cells = "".join(f"{matrix.loc[row, col]:10.3f}" for col in present)
        lines.append(f"{row:16}{cells}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=None, help="directory for .txt dumps")
    args = parser.parse_args(argv)

    chunks: list[str] = []

    raw = load_csv("zuco_sentence_average")
    chunks.append(
        _describe(raw, RAW_COLS, "ZuCo subject-mean gaze (raw units, 400 sentences)")
    )
    chunks.append(
        _corr(raw, RAW_COLS[1:], "Pearson correlation — measured raw sentence means")
    )

    std = load_zuco_experiment("standard")
    chunks.append(
        _describe(
            std,
            ["omissionRate", *FUSION_GAZE_COLUMNS, "SFD", "meanPupilSize"],
            "ZuCo experiment table (z-scored, what model_ZuCo_SST.py reads)",
        )
    )
    chunks.append(
        _corr(
            fusion_frame(std),
            list(FUSION_GAZE_COLUMNS),
            "Pearson correlation — five fusion features on measured z-scores",
        )
    )

    sst = load_csv("sst_combined")
    chunks.append(
        _describe(
            sst,
            ["nFix", "GD", "TRT", "FFD", "GPT"],
            "Full SST projected gaze (NOT milliseconds, 11853 sentences)",
        )
    )
    chunks.append(
        _corr(
            fusion_frame(sst),
            list(FUSION_GAZE_COLUMNS),
            "Pearson correlation — five fusion features on projected full SST",
        )
    )

    # How similar are the two correlation geometries?
    c_meas = fusion_frame(std).corr().to_numpy()
    c_proj = fusion_frame(sst).corr().to_numpy()
    # Frobenius distance between correlation matrices, ignore diagonal.
    mask = ~np.eye(c_meas.shape[0], dtype=bool)
    dist = float(np.sqrt(((c_meas - c_proj)[mask] ** 2).mean()))
    chunks.append(
        "Geometry check\n==============\n"
        f"RMS off-diagonal difference between measured-ZuCo and projected-SST\n"
        f"correlation matrices: {dist:.4f}\n"
        "A small number means the predictor roughly kept the duration cluster.\n"
        "A large number means Experiment B is looking at a different shape."
    )

    text = "\n\n".join(chunks)
    print(text)

    if args.out:
        args.out.mkdir(parents=True, exist_ok=True)
        dest = args.out / "gaze_feature_stats.txt"
        dest.write_text(text + "\n", encoding="utf-8")
        print(f"\nWrote {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
