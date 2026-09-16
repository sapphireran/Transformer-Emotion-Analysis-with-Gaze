#!/usr/bin/env python3
"""Rebuild the ZuCo text + gaze join and compare it to the committed table.

The original merge that produced `combined_sst_et_standard.csv` is not a
committed script. This example performs the join from the pieces that *are*
in git and checks that sentence ids, labels, and fusion columns still match.

Usage (from repo root):

    python3 examples/join_zuco_sst.py
    python3 examples/join_zuco_sst.py --write examples/sample_outputs/join_zuco_sst.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from paths import (
    ZUCO_AVERAGE_MINMAX,
    ZUCO_AVERAGE_STANDARD,
    ZUCO_COMBINED_MINMAX,
    ZUCO_COMBINED_STANDARD,
    ZUCO_FUSION_FEATURES,
    ZUCO_TEXT,
)

ID_ALIASES = ("id", "Unnamed: 0")


def _with_sentence_id(gaze: pd.DataFrame) -> pd.DataFrame:
    frame = gaze.copy()
    if "id" in frame.columns:
        frame["sentence_id"] = frame["id"].astype(int)
    elif "Unnamed: 0" in frame.columns:
        frame["sentence_id"] = frame["Unnamed: 0"].astype(int)
    else:
        frame["sentence_id"] = frame.index.astype(int)
    return frame


def _merge(text: pd.DataFrame, gaze: pd.DataFrame) -> pd.DataFrame:
    gaze = _with_sentence_id(gaze)
    merged = text.merge(gaze, on="sentence_id", how="inner", suffixes=("", "_gaze"))
    return merged.sort_values("sentence_id").reset_index(drop=True)


def _compare(name: str, rebuilt: pd.DataFrame, committed: pd.DataFrame) -> dict:
    report = {
        "name": name,
        "rebuilt_rows": int(len(rebuilt)),
        "committed_rows": int(len(committed)),
        "id_match": set(rebuilt["sentence_id"]) == set(committed["sentence_id"]),
        "columns_only_in_rebuilt": sorted(set(rebuilt.columns) - set(committed.columns)),
        "columns_only_in_committed": sorted(set(committed.columns) - set(rebuilt.columns)),
        "fusion_max_abs_diff": {},
        "label_mismatch": 0,
        "sentence_mismatch": 0,
    }

    aligned = committed.sort_values("sentence_id").reset_index(drop=True)
    rebuilt_a = rebuilt.sort_values("sentence_id").reset_index(drop=True)
    if len(aligned) != len(rebuilt_a) or not np.array_equal(
        aligned["sentence_id"].to_numpy(), rebuilt_a["sentence_id"].to_numpy()
    ):
        report["aligned"] = False
        return report
    report["aligned"] = True

    report["label_mismatch"] = int(
        (aligned["sentiment_label"].astype(int) != rebuilt_a["sentiment_label"].astype(int)).sum()
    )
    report["sentence_mismatch"] = int(
        (aligned["sentence"].astype(str) != rebuilt_a["sentence"].astype(str)).sum()
    )
    for col in ZUCO_FUSION_FEATURES:
        if col not in rebuilt_a.columns or col not in aligned.columns:
            report["fusion_max_abs_diff"][col] = None
            continue
        diff = (rebuilt_a[col].astype(float) - aligned[col].astype(float)).abs()
        report["fusion_max_abs_diff"][col] = float(diff.max())
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", type=Path, default=None)
    parser.add_argument(
        "--atol",
        type=float,
        default=1e-6,
        help="max allowed |rebuilt - committed| on fusion columns",
    )
    args = parser.parse_args()

    text = pd.read_csv(ZUCO_TEXT)
    standard_gaze = pd.read_csv(ZUCO_AVERAGE_STANDARD)
    minmax_gaze = pd.read_csv(ZUCO_AVERAGE_MINMAX)
    committed_std = pd.read_csv(ZUCO_COMBINED_STANDARD)
    committed_mm = pd.read_csv(ZUCO_COMBINED_MINMAX)

    rebuilt_std = _merge(text, standard_gaze)
    rebuilt_mm = _merge(text, minmax_gaze)

    reports = [
        _compare("standard", rebuilt_std, committed_std),
        _compare("min-max", rebuilt_mm, committed_mm),
    ]

    print(f"text rows={len(text)} gaze_standard={len(standard_gaze)} gaze_minmax={len(minmax_gaze)}")
    print(f"inner-join standard={len(rebuilt_std)} min-max={len(rebuilt_mm)}")
    dropped = set(text["sentence_id"]) - set(rebuilt_std["sentence_id"])
    if dropped:
        print("text ids missing from standard gaze:", sorted(dropped)[:20])
    extra = set(_with_sentence_id(standard_gaze)["sentence_id"]) - set(text["sentence_id"])
    if extra:
        print("gaze ids missing from text:", sorted(extra)[:20])

    ok = True
    for report in reports:
        print(f"\n=== {report['name']} ===")
        print(
            f"rows rebuilt/committed={report['rebuilt_rows']}/{report['committed_rows']} "
            f"ids_equal={report['id_match']} aligned={report['aligned']}"
        )
        print("label mismatches:", report["label_mismatch"])
        print("sentence mismatches:", report["sentence_mismatch"])
        print("fusion max |Δ|:", report["fusion_max_abs_diff"])
        if report["columns_only_in_rebuilt"] or report["columns_only_in_committed"]:
            print("column delta rebuilt-only:", report["columns_only_in_rebuilt"])
            print("column delta committed-only:", report["columns_only_in_committed"])
        if not report["aligned"] or report["label_mismatch"] or report["sentence_mismatch"]:
            ok = False
        for col, value in report["fusion_max_abs_diff"].items():
            if value is None or value > args.atol:
                ok = False

    print("\njoin matches committed tables" if ok else "\njoin differs from committed tables")

    payload = {"ok": ok, "reports": reports}
    if args.write:
        args.write.parent.mkdir(parents=True, exist_ok=True)
        args.write.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"wrote {args.write}")

    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
