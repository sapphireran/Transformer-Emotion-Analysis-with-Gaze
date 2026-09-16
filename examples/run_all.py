#!/usr/bin/env python3
"""Run every example script in a fixed order and fail on the first error.

Usage (repo root):

    python examples/run_all.py
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

SCRIPTS = [
    "examples/sst_split_sanity.py",
    "examples/subject_coverage.py",
    "examples/inspect_sentence_gaze.py",
    "examples/inspect_word_gaze.py",
    "examples/label_and_gaze_summary.py",
    "examples/predicted_gaze_preview.py",
    "examples/toy_fusion_forward.py",
    "examples/gaze_only_baseline.py",
    "examples/metrics_example.py",
]


def main() -> int:
    for rel in SCRIPTS:
        print("\n" + "#" * 72)
        print(f"# {rel}")
        print("#" * 72, flush=True)
        proc = subprocess.run([sys.executable, str(ROOT / rel)], cwd=ROOT)
        if proc.returncode != 0:
            print(f"FAILED {rel} with exit {proc.returncode}", file=sys.stderr)
            return proc.returncode
    print("\nAll example scripts exited 0.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
