#!/usr/bin/env python3
"""Run the five personal examples in order."""

from __future__ import annotations

import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SCRIPTS = (
    "01_inspect_datasets.py",
    "02_gaze_feature_report.py",
    "03_text_vs_gaze_baselines.py",
    "04_sentence_walkthrough.py",
    "05_full_sst_sample.py",
)


def main() -> int:
    here = Path(__file__).resolve().parent
    for name in SCRIPTS:
        print(f"\n=== {name} ===")
        try:
            runpy.run_path(str(here / name), run_name="__main__")
        except SystemExit as exc:
            if exc.code not in (0, None):
                raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
