#!/usr/bin/env python3
"""Run every documentation example and fail if any child exits non-zero."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

_EXAMPLES = Path(__file__).resolve().parent
if str(_EXAMPLES) not in sys.path:
    sys.path.insert(0, str(_EXAMPLES))

from paths import EXAMPLES_DIR, OUTPUT_DIR, REPO_ROOT

SCRIPTS = [
    "inspect_datasets.py",
    "data_integrity.py",
    "gaze_feature_report.py",
    "gaze_only_baseline.py",
    "late_fusion_demo.py",
    "word_level_preview.py",
    "sentence_walkthrough.py",
]


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    log_path = OUTPUT_DIR / "run_all.txt"
    failed: list[str] = []
    chunks: list[str] = []

    for name in SCRIPTS:
        banner = f"RUNNING examples/{name}"
        print("=" * 72)
        print(banner)
        print("=" * 72)
        proc = subprocess.run(
            [sys.executable, str(EXAMPLES_DIR / name)],
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
        )
        text = proc.stdout
        if proc.stderr:
            text += ("\n[stderr]\n" + proc.stderr) if text else proc.stderr
        print(text, end="" if text.endswith("\n") or not text else "\n")
        chunks.append(f"$ python examples/{name}\n{text}\n")
        if proc.returncode != 0:
            failed.append(name)
            print(f"[exit {proc.returncode}]")

    log_path.write_text("".join(chunks), encoding="utf-8")
    print("=" * 72)
    print(f"Wrote transcript to {log_path.relative_to(REPO_ROOT)}")
    if failed:
        print("FAILED: " + ", ".join(failed))
        return 1
    print(f"All {len(SCRIPTS)} example scripts passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
