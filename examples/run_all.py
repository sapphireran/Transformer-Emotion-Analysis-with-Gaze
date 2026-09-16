#!/usr/bin/env python3
"""Run the numbered examples in order and collect a summary JSON."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from examples.lib.paths import resolve_root
from examples.lib.reporting import banner

SCRIPTS = (
    "01_inspect_datasets.py",
    "02_label_and_length_profile.py",
    "03_gaze_feature_stats.py",
    "04_subject_variability.py",
    "05_word_level_skip_rates.py",
    "06_feature_fusion_walkthrough.py",
    "07_split_leakage_check.py",
    "08_gaze_prediction_compare.py",
    "09_export_analysis_tables.py",
    "10_zuco_index_alignment.py",
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=None)
    parser.add_argument("--output-dir", default="examples/output")
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop on the first non-zero example exit code.",
    )
    args = parser.parse_args()
    root = resolve_root(args.root)
    out = Path(args.output_dir)
    if not out.is_absolute():
        out = root / out
    out.mkdir(parents=True, exist_ok=True)

    python = sys.executable
    results = []
    overall_ok = True
    for name in SCRIPTS:
        script = Path(__file__).resolve().parent / name
        cmd = [python, str(script), "--root", str(root)]
        if name in {"09_export_analysis_tables.py", "10_zuco_index_alignment.py"}:
            cmd.extend(["--output-dir", str(out)])
        banner(f"RUN {name}")
        started = time.time()
        proc = subprocess.run(cmd, cwd=str(root))
        elapsed = time.time() - started
        ok = proc.returncode == 0
        overall_ok = overall_ok and ok
        results.append(
            {
                "script": name,
                "returncode": proc.returncode,
                "seconds": round(elapsed, 3),
                "ok": ok,
            }
        )
        print(f"exit {proc.returncode} in {elapsed:.2f}s")
        if args.fail_fast and not ok:
            break

    summary = {
        "ok": overall_ok,
        "output_dir": str(out),
        "results": results,
    }
    summary_path = out / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    banner("Summary")
    print(json.dumps(summary, indent=2))
    print(f"wrote {summary_path}")
    return 0 if overall_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
