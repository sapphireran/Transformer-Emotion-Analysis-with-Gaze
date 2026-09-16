#!/usr/bin/env python3
"""Run every docs example and exit non-zero if any check fails."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

EXAMPLES = Path(__file__).resolve().parent
SCRIPTS = [
    "inspect_datasets.py",
    "split_integrity.py",
    "scaling_check.py",
    "fusion_forward.py",
    "gaze_by_sentiment.py",
    "word_to_sentence.py",
]


def main() -> int:
    failures = []
    for name in SCRIPTS:
        command = [sys.executable, str(EXAMPLES / name)]
        print(f"\n>>> {name}")
        result = subprocess.run(command, check=False)
        if result.returncode != 0:
            failures.append(name)

    print()
    if failures:
        print("FAILED:", ", ".join(failures))
        return 1
    print(f"all {len(SCRIPTS)} examples OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
