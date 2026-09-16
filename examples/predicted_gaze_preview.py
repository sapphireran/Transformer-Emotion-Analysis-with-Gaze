#!/usr/bin/env python3
"""Preview predicted sentence-level gaze on full SST (Track B).

Human ZuCo milliseconds and predicted full-SST columns are **not** on the
same scale. This script prints distributions on train/valid/test so a
fusion run is not surprised by negatives, values > 1, or split drift.

Usage (repo root):

    python examples/predicted_gaze_preview.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from examples.common import (
    GAZE5_SST,
    GAZE_PRED_DIR,
    SST_DIR,
    format_stats_table,
    floats,
    int_col,
    label_histogram,
    matrix,
    read_dicts,
)


def _split_drift() -> None:
    cols = GAZE5_SST
    parts = {
        "train": read_dicts(SST_DIR / "train_full_sst.csv"),
        "valid": read_dicts(SST_DIR / "valid_full_sst.csv"),
        "test": read_dicts(SST_DIR / "test_full_sst.csv"),
    }
    print("\n===== mean gaze by split (should be close if i.i.d.) =====")
    print(f"{'split':<8} " + " ".join(f"{c:>10}" for c in cols))
    print("-" * (8 + 11 * len(cols)))
    for name, rows in parts.items():
        means = matrix(rows, cols).mean(axis=0)
        print(f"{name:<8} " + " ".join(f"{m:10.4f}" for m in means))


def _word_level_sample() -> None:
    pred = read_dicts(GAZE_PRED_DIR / "prediction_test.csv")
    v2 = read_dicts(GAZE_PRED_DIR / "prediction_test_v2.csv")
    provo = read_dicts(GAZE_PRED_DIR / "provo.csv")
    print("\n===== word-level predicted dumps =====")
    print(f"prediction_test.csv     rows={len(pred):7d}  cols={list(pred[0])}")
    print(f"prediction_test_v2.csv  rows={len(v2):7d}  cols={list(v2[0])}")
    print(f"provo.csv               rows={len(provo):7d}  cols={list(provo[0])}")
    print()
    print("PROVO uses fixProp instead of GD — do not stack with SST word files.")
    nfix = floats(v2, "nFix")
    print(
        f"prediction_test_v2 nFix: min={nfix.min():.3f} max={nfix.max():.3f} "
        f"mean={nfix.mean():.3f}  (already scaled, not ZuCo counts)"
    )
    sent_ids = {int(row["sentence_id"]) for row in v2}
    print(f"unique sentence_id in v2: {min(sent_ids)}–{max(sent_ids)} n={len(sent_ids)}")


def main() -> None:
    print("Track B sentence tables use columns nFix, FFD, GPT, TRT, GD.")
    print("model_full_SST.py reorders CSV nFix,GD,TRT,FFD,GPT into that fusion order.")

    for split in ("train_full_sst.csv", "valid_full_sst.csv", "test_full_sst.csv"):
        rows = read_dicts(SST_DIR / split)
        print(f"\n===== {split} =====")
        print(f"rows={len(rows)}")
        print(label_histogram(int_col(rows, "sentiment_label")))
        print()
        print(format_stats_table(GAZE5_SST, [floats(rows, c) for c in GAZE5_SST]))
        x = matrix(rows, GAZE5_SST)
        print(f"share of rows with any negative gaze value: {(x < 0).any(axis=1).mean():.1%}")
        print(f"share of rows with any value > 1:           {(x > 1).any(axis=1).mean():.1%}")

    _split_drift()
    _word_level_sample()

    print()
    print("Negatives and values > 1 are expected after z-scoring / model output.")
    print("They are a bug only if you thought these were ZuCo milliseconds.")


if __name__ == "__main__":
    main()
