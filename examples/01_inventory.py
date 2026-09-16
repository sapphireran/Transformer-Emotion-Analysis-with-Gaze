#!/usr/bin/env python3
"""Print every catalogued table and fail if a contract drifts."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gazebook.reports import md_table, write_text
from gazebook.schema import (
    FULL_SST_LAST_BATCH,
    FULL_SST_TEST_ROWS,
    check_label_mix,
    check_subject_lengths,
    check_tables,
    inventory,
)
from gazebook.paths import repo_root


def main() -> int:
    root = repo_root()
    rows = []
    for item in inventory(root):
        mark = "yes" if item.get("rows") == item["expected_rows"] else "NO"
        cols = item.get("columns") or []
        preview = ", ".join(cols[:6])
        if len(cols) > 6:
            preview += ", …"
        rows.append(
            [
                item["path"],
                item["expected_rows"],
                item.get("rows", "missing"),
                mark,
                preview,
            ]
        )
    table = md_table(["path", "expected", "seen", "match", "columns"], rows)
    print(table)

    errors = []
    for check in (check_tables, check_label_mix, check_subject_lengths):
        result = check(root)
        if not result.ok:
            errors.extend(result.errors)

    print()
    print(
        f"Full-SST test loop: {FULL_SST_TEST_ROWS} rows, batch 256, "
        f"last batch {FULL_SST_LAST_BATCH} "
        f"({FULL_SST_LAST_BATCH / FULL_SST_TEST_ROWS:.1%} of the test set)."
    )
    if errors:
        print("CONTRACT FAILURES:")
        for err in errors:
            print(" -", err)
        return 1
    print("All inventory contracts passed.")
    write_text(root / "examples/output/01_inventory.md", table + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
