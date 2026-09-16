#!/usr/bin/env python3
"""Id leakage and class-drift checks for the convenience 80/10/10 splits."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "examples"))

from gazekit.io import load_sentence_table
from gazekit.paths import default_paths
from gazekit.report import format_split, section
from gazekit.splits import check_splits, max_class_drift


def build_report(root: Path | None = None) -> dict:
    paths = default_paths(root)
    zuco = {
        "train": load_sentence_table(paths.zuco_train),
        "valid": load_sentence_table(paths.zuco_valid),
        "test": load_sentence_table(paths.zuco_test),
        "combined": load_sentence_table(paths.zuco_combined_standard),
    }
    full = {
        "train": load_sentence_table(paths.full_sst_train),
        "valid": load_sentence_table(paths.full_sst_valid),
        "test": load_sentence_table(paths.full_sst_test),
    }
    zuco_report = check_splits(
        zuco["train"], zuco["valid"], zuco["test"], expected_union=zuco["combined"]
    )
    full_report = check_splits(full["train"], full["valid"], full["test"])
    return {
        "zuco": zuco_report,
        "full": full_report,
        "zuco_drift": max_class_drift(zuco_report, versus="combined"),
        "full_drift": max_class_drift(full_report, versus="train"),
    }


def render(bundle: dict) -> str:
    return "\n".join(
        [
            section(
                "ZuCo SST convenience split (train/valid/test CSV, not used by model_ZuCo_SST.py)",
                format_split(bundle["zuco"]) + f"\nmax class drift vs combined: {bundle['zuco_drift']:.4f}",
            ),
            section(
                "Full SST split (the one model_full_SST.py actually trains on)",
                format_split(bundle["full"]) + f"\nmax class drift vs train: {bundle['full_drift']:.4f}",
            ),
        ]
    )


def main() -> None:
    print(render(build_report()))


if __name__ == "__main__":
    main()
