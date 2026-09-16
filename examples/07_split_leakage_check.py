#!/usr/bin/env python3
"""Confirm the full-SST holdout splits do not share sentence_id values."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from examples.lib.loading import documented_datasets, load_dataset, load_full_sst_splits
from examples.lib.paths import resolve_root
from examples.lib.reporting import banner, print_frame

import pandas as pd


def check_full_sst(root) -> tuple[pd.DataFrame, dict[str, int]]:
    splits = load_full_sst_splits(root=root)
    combined = load_dataset(
        next(s for s in documented_datasets() if s.key == "sst_combined"), root=root
    )
    ids = {name: set(df["sentence_id"].tolist()) for name, df in splits.items()}
    rows = []
    for name, df in splits.items():
        rows.append(
            {
                "split": name,
                "rows": len(df),
                "unique_ids": len(ids[name]),
                "duplicate_ids_inside": int(len(df) - len(ids[name])),
            }
        )
    union = ids["train"] | ids["valid"] | ids["test"]
    combined_ids = set(combined["sentence_id"].tolist())
    extra = {
        "train_valid_overlap": len(ids["train"] & ids["valid"]),
        "train_test_overlap": len(ids["train"] & ids["test"]),
        "valid_test_overlap": len(ids["valid"] & ids["test"]),
        "union_vs_combined": len(union ^ combined_ids),
        "union_size": len(union),
        "combined_unique": len(combined_ids),
    }
    return pd.DataFrame(rows), extra


def check_zuco_holdout(root) -> dict[str, int]:
    """The unused 320/40/40 files — still worth proving they are disjoint."""
    base = Path(root) / "ZuCo_SST_data"
    pieces = {
        name: set(pd.read_csv(base / f"{name}.csv")["sentence_id"].tolist())
        for name in ("train", "valid", "test")
    }
    return {
        "train_valid_overlap": len(pieces["train"] & pieces["valid"]),
        "train_test_overlap": len(pieces["train"] & pieces["test"]),
        "valid_test_overlap": len(pieces["valid"] & pieces["test"]),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=None)
    args = parser.parse_args()
    root = resolve_root(args.root)

    table, extra = check_full_sst(root)
    banner("Full SST split integrity")
    print_frame(table)
    print()
    for key, value in extra.items():
        print(f"{key}: {value}")

    banner("Unused ZuCo 320/40/40 split integrity")
    zuco_overlap = check_zuco_holdout(root)
    for key, value in zuco_overlap.items():
        print(f"{key}: {value}")

    leaked = any(
        extra[k] for k in ("train_valid_overlap", "train_test_overlap", "valid_test_overlap")
    )
    leaked = leaked or extra["union_vs_combined"] != 0
    leaked = leaked or bool((table["duplicate_ids_inside"] > 0).any())
    leaked = leaked or any(zuco_overlap.values())
    if leaked:
        print("\nLEAK or coverage problem detected.")
        return 1
    print(
        "\nNo sentence_id overlap in the full-SST holdout, and the three "
        "splits cover combined_full_sst_et.csv exactly. The unused ZuCo "
        "320/40/40 files are also disjoint. model_ZuCo_SST.py still uses "
        "5-fold CV on all 400 rows — this check does not change that."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
