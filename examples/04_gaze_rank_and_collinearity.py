#!/usr/bin/env python3
"""Compare recorded ZuCo gaze rank with projected full-SST gaze rank."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from gazebook.csvio import float_col, read_dicts, table_to_array
from gazebook.paths import repo_root
from gazebook.reports import md_table, write_text
from gazebook.schema import SST_GAZE, ZUCO_GAZE
from gazebook.stats import condition_number_corr, corr_matrix, eigenvalues_corr, pearson


def _block(name: str, rows, cols, labels, lengths) -> tuple[str, np.ndarray, float]:
    X = table_to_array(rows, cols)
    C = corr_matrix(X)
    ev = eigenvalues_corr(X)
    cond = condition_number_corr(X)
    header = [" "] + list(cols)
    body = [[cols[i], *C[i].tolist()] for i in range(len(cols))]
    print(f"## {name}")
    print(md_table(header, body))
    print(f"eigenvalues: {np.round(ev, 5).tolist()}")
    print(f"condition number of the correlation matrix: {cond:.1f}")
    print("Pearson r(label, feature):")
    for i, col in enumerate(cols):
        print(f"  {col:12s} {pearson(labels, X[:, i]):+.4f}")
    print("Pearson r(word-count, feature):")
    for i, col in enumerate(cols):
        print(f"  {col:12s} {pearson(lengths, X[:, i]):+.4f}")
    print()
    return name, ev, cond


def main() -> int:
    root = repo_root()
    _, zuco = read_dicts(root / "ZuCo_SST_data/combined_sst_et_standard.csv")
    _, sst = read_dicts(root / "SST_data/combined_full_sst_et.csv")
    z_lab = float_col(zuco, "sentiment_label")
    s_lab = float_col(sst, "sentiment_label")
    z_len = np.array([len(r["sentence"].split()) for r in zuco], dtype=np.float64)
    s_len = np.array([len(r["sentence"].split()) for r in sst], dtype=np.float64)

    _, ev_z, cond_z = _block("ZuCo recorded gaze (400 sentences)", zuco, ZUCO_GAZE, z_lab, z_len)
    _, ev_s, cond_s = _block("Full SST projected gaze (11,853 sentences)", sst, SST_GAZE, s_lab, s_len)

    print(
        "Projected SST gaze is nearly rank-1: four of five channels have pairwise "
        f"r ≥ 0.986 except GD. Correlation condition number {cond_s:.0f} vs "
        f"{cond_z:.0f} on recorded ZuCo gaze."
    )

    write_text(
        root / "examples/output/04_gaze_collinearity.md",
        "\n".join(
            [
                "# Gaze rank",
                "",
                f"ZuCo corr condition number: {cond_z:.1f}",
                f"SST corr condition number: {cond_s:.1f}",
                f"ZuCo eigenvalues: {np.round(ev_z, 6).tolist()}",
                f"SST eigenvalues: {np.round(ev_s, 6).tolist()}",
                "",
            ]
        ),
    )
    # Degeneracy lock: projected SST should be far more collinear than ZuCo.
    if not (cond_s > 1000 and cond_z < 1000):
        print("UNEXPECTED collinearity regime", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
