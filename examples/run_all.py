#!/usr/bin/env python3
"""Run every CPU example and collect JSON dumps under sample_outputs/.

Usage (from repo root):

    python3 examples/run_all.py
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from paths import REPO_ROOT, SAMPLE_OUTPUT_DIR

COMMANDS = [
    [sys.executable, "examples/inspect_datasets.py"],
    [sys.executable, "examples/join_zuco_sst.py", "--write", "examples/sample_outputs/join_zuco_sst.json"],
    [sys.executable, "examples/subject_agreement.py", "--write", "examples/sample_outputs/subject_agreement.json"],
    [sys.executable, "examples/gaze_feature_stats.py", "--write", "examples/sample_outputs/gaze_feature_stats.json"],
    [sys.executable, "examples/sentiment_baselines.py", "--write", "examples/sample_outputs/baselines.json"],
    [sys.executable, "examples/fusion_toy.py", "--write", "examples/sample_outputs/fusion_toy.json"],
    [sys.executable, "examples/word_level_gaze.py", "--write", "examples/sample_outputs/word_level_gaze.json"],
]


def main() -> int:
    SAMPLE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    failed = []
    for cmd in COMMANDS:
        printable = " ".join(cmd[1:])
        print(f"\n########## {printable} ##########\n")
        proc = subprocess.run(cmd, cwd=REPO_ROOT)
        if proc.returncode != 0:
            failed.append((printable, proc.returncode))
    print("\n========== summary ==========")
    if failed:
        for name, code in failed:
            print(f"FAIL {name} (exit {code})")
        return 1
    print("all example scripts passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
