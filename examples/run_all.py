#!/usr/bin/env python3
"""Run the six examples in order and stop on the first non-zero exit.

    python3 examples/run_all.py
"""

from __future__ import annotations

import runpy
import sys
from pathlib import Path

SCRIPTS = [
    "01_inspect_datasets.py",
    "02_schema_check.py",
    "03_gaze_feature_summary.py",
    "04_label_and_length.py",
    "05_word_level_gaze.py",
    "06_toy_late_fusion.py",
]


def main() -> int:
    here = Path(__file__).resolve().parent
    sys.path.insert(0, str(here))
    for name in SCRIPTS:
        path = here / name
        print("=" * 72)
        print(f"running {name}")
        print("=" * 72)
        # Each script is a __main__ module with its own main().
        try:
            runpy.run_path(str(path), run_name="__main__")
        except SystemExit as exc:
            code = exc.code if isinstance(exc.code, int) else (0 if exc.code is None else 1)
            if code != 0:
                print(f"{name} exited with {code}", file=sys.stderr)
                return code
        print()
    print("all examples finished")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
