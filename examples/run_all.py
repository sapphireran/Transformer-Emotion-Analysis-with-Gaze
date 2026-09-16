#!/usr/bin/env python3
"""Run every personal example script in a fixed order.

    python3 examples/run_all.py
"""

from __future__ import annotations

import os
import subprocess
import sys

SCRIPTS = (
    "inspect_datasets.py",
    "label_distribution.py",
    "feature_stats.py",
    "schema_validate.py",
    "sentence_gaze_join.py",
    "sample_rows.py",
    "word_level_preview.py",
    "gaze_fusion_demo.py",
)


def main() -> int:
    here = os.path.dirname(os.path.abspath(__file__))
    repo = os.path.abspath(os.path.join(here, os.pardir))
    python = sys.executable

    def log(msg: str = "") -> None:
        print(msg, flush=True)

    log(f"Running {len(SCRIPTS)} example scripts with {python}")
    log(f"cwd = {repo}")
    log()
    for name in SCRIPTS:
        path = os.path.join(here, name)
        log("=" * 72)
        log(name)
        log("=" * 72)
        proc = subprocess.run([python, path], cwd=repo)
        if proc.returncode != 0:
            print(f"\n{name} exited {proc.returncode}", file=sys.stderr, flush=True)
            return proc.returncode
        log()
    log("All example scripts exited 0.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
