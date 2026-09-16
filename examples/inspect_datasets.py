#!/usr/bin/env python3
"""Print the personal dataset inventory and compare it to documented sizes.

Run from the repository root:

    python3 examples/inspect_datasets.py
"""

from __future__ import annotations

import os
import sys

from common import (
    DATASETS,
    SUBJECT_SENTENCE_FILES,
    SUBJECT_WORD_FILES,
    abs_path,
    count_rows,
    ensure_cwd_message,
)


def main() -> int:
    warn = ensure_cwd_message()
    if warn:
        print(warn, file=sys.stderr)
        return 2

    print("Transformer Emotion Analysis with Gaze — dataset inventory")
    print(f"repo root: {abs_path('')}")
    print()

    worst = 0
    print(f"{'key':<26}{'rows':<14}{'status':<28}path")
    print("-" * 100)
    for spec in DATASETS:
        rel = str(spec["path"])
        expected = int(spec["expected_rows"])
        path = abs_path(rel)
        if not os.path.isfile(path):
            status = "MISSING"
            rows = "-"
            worst = max(worst, 2)
        else:
            cols, n = count_rows(rel)
            rows = str(n)
            if n == expected:
                status = f"ok ({len(cols)} cols)"
            else:
                status = f"expected {expected}"
                worst = max(worst, 1)
        print(f"{spec['key']:<26}{rows:<14}{status:<28}{rel}")

    print()
    print("Per-subject sentence-level CSVs (ZuCo_et_csv_data/{n}_SR.csv)")
    print(f"{'file':<40}{'rows':>8}")
    print("-" * 50)
    for rel in SUBJECT_SENTENCE_FILES:
        if not os.path.isfile(abs_path(rel)):
            print(f"{rel:<40}{'MISSING':>8}")
            worst = max(worst, 2)
            continue
        _cols, n = count_rows(rel)
        flag = "  ← short subject" if n != 400 else ""
        print(f"{os.path.basename(rel):<40}{n:>8}{flag}")

    print()
    print("Per-subject word-level CSVs (ZuCo_et_csv_data/word/{n}_SR.csv)")
    print(f"{'file':<40}{'rows':>8}")
    print("-" * 50)
    for rel in SUBJECT_WORD_FILES:
        if not os.path.isfile(abs_path(rel)):
            print(f"{rel:<40}{'MISSING':>8}")
            worst = max(worst, 2)
            continue
        _cols, n = count_rows(rel)
        flag = "  ← short subject" if n != 7129 else ""
        print(f"{os.path.basename(rel):<40}{n:>8}{flag}")

    print()
    print("Notes")
    print("  • Subject 3 is expected to be short (task1 MATLAB holes).")
    print("  • model_ZuCo_SST.py reads combined_sst_et_standard.csv, not train/valid/test.")
    print("  • model_full_SST.py reads SST_data/*_full_sst.csv.")
    if worst == 0:
        print("  • All documented tables match the expected row counts.")
    return worst


if __name__ == "__main__":
    sys.exit(main())
