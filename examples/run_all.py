#!/usr/bin/env python3
"""Run the personal example suite in order and refresh examples/output/."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

SCRIPTS = (
    "01_inventory.py",
    "02_reader3_forensics.py",
    "03_word_token_census.py",
    "04_gaze_rank_and_collinearity.py",
    "05_reader_reliability.py",
    "06_join_and_scalers.py",
    "07_split_and_trainer_errata.py",
    "08_cpu_baselines.py",
    "09_sentence_walkthrough.py",
    "10_build_labbook.py",
)


def main() -> int:
    here = Path(__file__).resolve().parent
    root = here.parent
    failed = []
    for name in SCRIPTS:
        print("=" * 72)
        print(name)
        print("=" * 72)
        proc = subprocess.run([sys.executable, str(here / name)], cwd=root)
        if proc.returncode != 0:
            failed.append(name)
        print()
    if failed:
        print("FAILED:", ", ".join(failed))
        return 1
    print(f"All {len(SCRIPTS)} example scripts passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
