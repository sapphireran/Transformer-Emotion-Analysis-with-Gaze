#!/usr/bin/env python3
"""Print a catalog of every checked-in table the trainers and docs mention."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gaze_emotion.datasets import KNOWN_TABLES, load_and_summarize


EXTRA_TABLES = {
    "zuco_subject_1_sentence": "ZuCo_et_csv_data/1_SR.csv",
    "zuco_subject_3_sentence": "ZuCo_et_csv_data/3_SR.csv",
    "zuco_subject_1_word": "ZuCo_et_csv_data/word/1_SR.csv",
    "gaze_prediction_v2": "gaze_prediction/data/prediction_test_v2.csv",
    "provo_word": "gaze_prediction/data/provo.csv",
}


def main() -> int:
    print("Checked-in dataset catalog")
    print("=" * 72)
    names = list(KNOWN_TABLES.items()) + list(EXTRA_TABLES.items())
    failures = 0
    for name, relpath in names:
        print()
        print(f"[{name}]")
        try:
            summary = load_and_summarize(relpath)
        except FileNotFoundError as exc:
            print(f"  missing: {exc}")
            failures += 1
            continue
        for line in summary.as_lines():
            print(f"  {line}")
    print()
    print(f"tables inspected: {len(names)}  missing: {failures}")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
