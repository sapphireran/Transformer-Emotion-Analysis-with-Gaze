#!/usr/bin/env python3
"""List every personal CSV in this repo and write inventory reports."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tea_gaze.io import inventory, load_and_validate
from tea_gaze.paths import repo_root
from tea_gaze.reports import (
    html_inventory,
    markdown_inventory,
    markdown_label_table,
    write_text,
)
from tea_gaze.schema import summarize_labels


def main() -> int:
    items = inventory()
    out_dir = repo_root() / "examples" / "output"
    md_path = write_text(out_dir / "dataset_inventory.md", markdown_inventory(items))
    html_path = write_text(out_dir / "dataset_inventory.html", html_inventory(items))

    print(f"Wrote {md_path.relative_to(repo_root())}")
    print(f"Wrote {html_path.relative_to(repo_root())}")
    print()
    print(f"{'key':<24} {'rows':>6} {'gaze':<10} path")
    print("-" * 80)
    for item in items:
        rows = "—" if item.rows is None else str(item.rows)
        print(f"{item.key:<24} {rows:>6} {item.gaze_source:<10} {item.path}")

    print()
    rows, report = load_and_validate("zuco_sst_standard")
    report.raise_if_invalid()
    print("ZuCo + SST label mix (measured gaze, 400 sentences)")
    print(markdown_label_table(summarize_labels(rows)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
