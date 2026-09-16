#!/usr/bin/env python3
"""Ridge one-vs-rest probes on the same five gaze columns the fusion head uses.

These numbers are a teaching floor, not a RoBERTa result. The question is
whether linear gaze even has room to help hashed text on 400 sentences.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from gazebook.baselines import cv_majority, cv_ridge, hash_bow, permute_rows
from gazebook.csvio import float_col, read_dicts, table_to_array
from gazebook.paths import repo_root
from gazebook.reports import md_table, write_text
from gazebook.schema import SST_GAZE, ZUCO_GAZE
from gazebook.stats import residualize, zscore


def _pack(name, result):
    return [name, result.mean_acc, result.std_acc, result.mean_f1]


def run_track(name: str, texts, y, gaze, seed: int = 42) -> list[list[object]]:
    y = np.asarray(y, dtype=int)
    gaze = np.asarray(gaze, dtype=np.float64)
    text_X = hash_bow(texts, dim=128)
    lengths = np.array([len(t.split()) for t in texts], dtype=np.float64).reshape(-1, 1)
    gaze_resid = np.column_stack([residualize(gaze[:, j], lengths[:, 0]) for j in range(gaze.shape[1])])
    fused = np.concatenate([zscore(text_X), gaze], axis=1)
    shuffled = np.concatenate([zscore(text_X), permute_rows(gaze, seed=7)], axis=1)

    rows = [
        _pack("majority", cv_majority(y, seed=seed)),
        _pack("length only", cv_ridge(zscore(lengths), y, seed=seed)),
        _pack("gaze only", cv_ridge(gaze, y, seed=seed)),
        _pack("gaze ⟂ length", cv_ridge(gaze_resid, y, seed=seed)),
        _pack("hashed text", cv_ridge(text_X, y, seed=seed)),
        _pack("text + gaze", cv_ridge(fused, y, seed=seed)),
        _pack("text + shuffled gaze", cv_ridge(shuffled, y, seed=seed)),
    ]
    print(f"## {name}  (5-fold stratified, seed={seed}, ridge OvR)")
    print(md_table(["model", "mean acc", "acc sd", "mean weighted F1"], rows))
    print()
    return rows


def main() -> int:
    root = repo_root()
    _, zuco = read_dicts(root / "ZuCo_SST_data/combined_sst_et_standard.csv")
    z_rows = run_track(
        "ZuCo ∩ SST (400 recorded-gaze sentences)",
        [r["sentence"] for r in zuco],
        float_col(zuco, "sentiment_label"),
        table_to_array(zuco, ZUCO_GAZE),
    )

    # Full SST is 11.8k rows; hashed 128-d ridge is still cheap.
    _, sst = read_dicts(root / "SST_data/combined_full_sst_et.csv")
    s_rows = run_track(
        "Full SST (11,853 projected-gaze sentences)",
        [r["sentence"] for r in sst],
        float_col(sst, "sentiment_label"),
        table_to_array(sst, SST_GAZE),
    )

    lines = [
        "# CPU baselines",
        "",
        "Ridge one-vs-rest, 5 stratified folds, seed 42. Not RoBERTa.",
        "",
        "## ZuCo",
        md_table(["model", "mean acc", "acc sd", "mean weighted F1"], z_rows),
        "",
        "## Full SST",
        md_table(["model", "mean acc", "acc sd", "mean weighted F1"], s_rows),
        "",
    ]
    write_text(root / "examples/output/08_cpu_baselines.md", "\n".join(lines))

    z_map = {r[0]: r for r in z_rows}
    # Gaze-only should sit near majority on 400 rows; hashed text should beat both.
    if z_map["hashed text"][1] <= z_map["majority"][1]:
        print("hashed text failed to beat majority on ZuCo — unexpected", file=sys.stderr)
        return 1
    if z_map["gaze only"][1] > z_map["hashed text"][1]:
        print("gaze-only beat hashed text — unexpected on this table", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
