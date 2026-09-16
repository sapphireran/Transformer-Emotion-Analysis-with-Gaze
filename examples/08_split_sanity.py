#!/usr/bin/env python3
"""Confirm the committed 80/10/10 splits are disjoint and add up to the combined table."""

from __future__ import annotations

import pandas as pd

from examples._common import banner
from tea_gaze.io import load_full_sst_splits, load_zuco_combined, load_zuco_splits
from tea_gaze.reports import label_counts, markdown_table


def _check_partition(name: str, combined: pd.DataFrame, parts: dict[str, pd.DataFrame], id_col: str) -> None:
    banner(f"{name}: row counts")
    rows = [{"piece": "combined", "rows": len(combined)}]
    rows.extend({"piece": key, "rows": len(frame)} for key, frame in parts.items())
    print(markdown_table(pd.DataFrame(rows), digits=0))

    ids = {key: set(frame[id_col].tolist()) for key, frame in parts.items()}
    overlap_tv = ids["train"] & ids["valid"]
    overlap_tt = ids["train"] & ids["test"]
    overlap_vt = ids["valid"] & ids["test"]
    union = ids["train"] | ids["valid"] | ids["test"]
    combined_ids = set(combined[id_col].tolist())

    banner(f"{name}: id disjointness on {id_col}")
    print(f"  train ∩ valid = {len(overlap_tv)}")
    print(f"  train ∩ test  = {len(overlap_tt)}")
    print(f"  valid ∩ test  = {len(overlap_vt)}")
    print(f"  |union of splits| = {len(union)}")
    print(f"  |combined ids|    = {len(combined_ids)}")
    print(f"  union == combined ids: {union == combined_ids}")
    print(f"  sum of split rows == combined rows: {sum(len(v) for v in parts.values()) == len(combined)}")

    banner(f"{name}: class share by split")
    frames = []
    for key, frame in {"combined": combined, **parts}.items():
        counts = label_counts(frame)
        counts.insert(0, "split", key)
        frames.append(counts)
    print(markdown_table(pd.concat(frames, ignore_index=True)))


def main() -> None:
    zuco_combined = load_zuco_combined(scaling="standard").frame
    zuco_parts = {name: bundle.frame for name, bundle in load_zuco_splits().items()}
    _check_partition("ZuCo standard (spilt.py)", zuco_combined, zuco_parts, "sentence_id")
    print()
    print("Reminder: model_ZuCo_SST.py ignores these three files and re-folds the 400.")

    sst = load_full_sst_splits()
    sst_parts = {name: sst[name].frame for name in ("train", "valid", "test")}
    _check_partition("Full SST (spilt.py)", sst["combined"].frame, sst_parts, "sentence_id")
    print()
    print("model_full_SST.py does use these three files.")


if __name__ == "__main__":
    main()
