#!/usr/bin/env python3
"""Walk through the five fused gaze channels with real ZuCo sentences."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gaze_emotion.constants import FUSION_GAZE_FEATURES, label_name
from gaze_emotion.datasets import feature_matrix, load_csv_rows

DEFINITIONS = {
    "nFixations": "How many times the eyes landed (sentence mean over fixated words).",
    "FFD": "First Fixation Duration — length of the first landing.",
    "GPT": "Go-Past Time — first entry until the eyes move past, including regressions.",
    "TRT": "Total Reading Time — every look, including later rereads.",
    "GD": "Gaze Duration — first-pass time before the eyes first leave the region.",
}


def clip(text: str, width: int = 88) -> str:
    text = " ".join(text.split())
    return text if len(text) <= width else text[: width - 3] + "..."


def extreme_rows(rows, matrix: np.ndarray, column: int, k: int = 2):
    order = np.argsort(matrix[:, column])
    low = [(rows[i], matrix[i, column]) for i in order[:k]]
    high = [(rows[i], matrix[i, column]) for i in order[-k:][::-1]]
    return low, high


def main() -> int:
    path = "ZuCo_SST_data/combined_sst_et_standard.csv"
    rows = load_csv_rows(path)
    matrix = feature_matrix(rows, FUSION_GAZE_FEATURES)

    print("ZuCo-SST late-fusion channels (standard-scaled, n=400)")
    print("=" * 72)
    print()
    for idx, name in enumerate(FUSION_GAZE_FEATURES):
        col = matrix[:, idx]
        print(f"{name}")
        print(f"  {DEFINITIONS[name]}")
        print(
            f"  mean={col.mean():+.3f}  std={col.std():.3f}  "
            f"min={col.min():+.3f}  max={col.max():+.3f}"
        )
        low, high = extreme_rows(rows, matrix, idx)
        print("  lowest:")
        for row, value in low:
            print(
                f"    {value:+.3f}  [{label_name(row['sentiment_label'])}]  "
                f"{clip(row['sentence'])}"
            )
        print("  highest:")
        for row, value in high:
            print(
                f"    {value:+.3f}  [{label_name(row['sentiment_label'])}]  "
                f"{clip(row['sentence'])}"
            )
        print()

    # A short mixed-polarity review is easy to hold in working memory while
    # looking at all five channels at once.
    demo_id = "3"
    demo = next(row for row in rows if row["sentence_id"] == demo_id)
    values = feature_matrix([demo], FUSION_GAZE_FEATURES)[0]
    print("Worked example: sentence_id=3")
    print(f"  text : {demo['sentence']}")
    print(f"  label: {label_name(demo['sentiment_label'])}")
    for name, value in zip(FUSION_GAZE_FEATURES, values):
        print(f"  {name:<12} {value:+.4f}")
    print()
    print(
        "These five numbers are what EyeTrackingModel projects with "
        "Linear(5, 16) before concatenating onto RoBERTa's pooler output."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
