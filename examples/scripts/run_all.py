"""Run every documentation example script and print a short summary."""

from __future__ import annotations

import _bootstrap  # noqa: F401

import runpy
import sys
from pathlib import Path

SCRIPTS = [
    "summarize_datasets.py",
    "analyze_gaze_features.py",
    "check_splits.py",
    "reconstruct_zuco_combined.py",
    "check_subject_alignment.py",
    "preview_word_gaze.py",
    "demo_fusion_forward.py",
]


def main(argv: list[str] | None = None) -> int:
    del argv
    here = Path(__file__).resolve().parent
    failures = []
    for name in SCRIPTS:
        path = here / name
        sys.stdout.write(f"\n======== {name} ========\n")
        sys.stdout.flush()
        try:
            runpy.run_path(str(path), run_name="__main__")
        except SystemExit as exc:
            code = exc.code if isinstance(exc.code, int) else 1
            if code not in (0, None):
                failures.append((name, code))
        except Exception as exc:  # pragma: no cover - surfaced in the summary
            failures.append((name, exc))
            sys.stdout.write(f"{name} raised {exc!r}\n")
    if failures:
        sys.stderr.write(f"failed: {failures}\n")
        return 1
    sys.stdout.write("\nAll example scripts finished.\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
