#!/usr/bin/env python3
"""Run every numbered example and write markdown snapshots."""

from __future__ import annotations

import runpy
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SCRIPTS = [
    "01_repo_inventory.py",
    "02_subject_alignment.py",
    "03_reader_agreement.py",
    "04_label_conditioned_gaze.py",
    "05_word_sentence_bridge.py",
    "06_split_and_schema.py",
    "07_test_loop_bug.py",
    "08_cpu_fusion.py",
    "09_gaze_baselines.py",
    "10_provo_preview.py",
]


def main() -> None:
    for name in SCRIPTS:
        print(f"\n======== {name} ========")
        sys.argv = [str(HERE / name), "--write"]
        runpy.run_path(str(HERE / name), run_name="__main__")
    sys.argv = [str(HERE / "11_write_lab_notebook.py")]
    runpy.run_path(str(HERE / "11_write_lab_notebook.py"), run_name="__main__")
    print("\nall examples wrote snapshots under examples/outputs/")


if __name__ == "__main__":
    main()
