#!/usr/bin/env python3
"""Pairwise reader agreement on the clean 150-sentence overlap."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from gazebook.paths import repo_root
from gazebook.remap import CLEAN_SENTENCE_END, load_subject_tables
from gazebook.reports import md_table, write_text
from gazebook.stats import icc1, pairwise_reader_r


FEATURES = ("nFixations", "TRT", "GPT", "GD", "FFD", "omissionRate", "meanPupilSize")


def main() -> int:
    root = repo_root()
    tables = load_subject_tables(root)
    ids = range(CLEAN_SENTENCE_END)
    rows = []
    icc_nfix = None
    for col in FEATURES:
        X = np.array([[float(tables[r][i][col]) for r in range(12)] for i in ids], dtype=np.float64)
        mean_r, min_r, max_r, n_pairs = pairwise_reader_r(X)
        icc = icc1(X)
        rows.append([col, mean_r, min_r, max_r, icc, n_pairs])
        if col == "nFixations":
            icc_nfix = icc
        print(f"{col:16s} mean r={mean_r:.3f}  min={min_r:.3f}  max={max_r:.3f}  ICC(1)={icc:.3f}")

    print()
    print(md_table(["feature", "mean pairwise r", "min r", "max r", "ICC(1)", "pairs"], rows))
    print()
    print(
        "Only original ids 0–149 are used. After that, reader 3's compact ids "
        "no longer refer to the same sentence, so an ICC on 0–399 would be invalid."
    )
    write_text(
        root / "examples/output/05_reader_reliability.md",
        md_table(["feature", "mean pairwise r", "min r", "max r", "ICC(1)", "pairs"], rows) + "\n",
    )
    if icc_nfix is None or not (0.20 < icc_nfix < 0.30):
        print("UNEXPECTED nFixations ICC", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
