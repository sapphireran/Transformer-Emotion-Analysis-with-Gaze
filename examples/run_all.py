#!/usr/bin/env python3
"""Run every numbered example in order, then assemble the HTML atlas."""

from __future__ import annotations

import runpy
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SCRIPTS = [
    "01_inventory.py",
    "02_label_atlas.py",
    "03_sidecar_rank.py",
    "04_subject3_reindex.py",
    "05_split_fingerprint.py",
    "06_last_batch_metric.py",
    "07_length_confound.py",
    "08_fusion_forward.py",
    "09_word_skips.py",
    "10_shuffle_control.py",
    "11_gaze_prediction_track.py",
    "12_reader_means.py",
    "13_write_html_atlas.py",
]


def main() -> int:
    sys.path.insert(0, str(HERE))
    for name in SCRIPTS:
        path = HERE / name
        print(f"\n=== {name} ===", flush=True)
        try:
            runpy.run_path(str(path), run_name="__main__")
        except SystemExit as exc:
            code = 0 if exc.code is None else int(exc.code)
            if code != 0:
                print(f"{name} exited with {code}", file=sys.stderr)
                return code
    print("\nAll atlas examples finished.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
