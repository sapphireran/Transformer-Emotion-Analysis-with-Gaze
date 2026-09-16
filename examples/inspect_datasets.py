#!/usr/bin/env python3
"""Print row counts, columns, and a sample sentence for every named table."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from examples.lib import cli, io_csv, paths, report


def inspect(name: str) -> dict:
    path = paths.table_path(name)
    header = io_csv.read_header(path)
    rows = io_csv.read_rows(path)
    sample = ""
    if rows:
        row = rows[0]
        sample = row.get("sentence") or row.get("Word") or row.get("word") or ""
        sample = " ".join(sample.split())
        if len(sample) > 96:
            sample = sample[:93] + "..."
    return {
        "name": name,
        "role": paths.table_role(name),
        "path": str(path.relative_to(paths.REPO_ROOT)),
        "rows": len(rows),
        "cols": len(header),
        "header": header,
        "sample": sample,
    }


def render(records: list[dict]) -> str:
    table = report.ascii_table(
        ("name", "rows", "cols", "role"),
        [(r["name"], r["rows"], r["cols"], r["role"]) for r in records],
    )
    details = []
    for rec in records:
        details.append(
            f"### {rec['name']}\n\n"
            f"- file: `{rec['path']}`\n"
            f"- columns: {', '.join(rec['header'])}\n"
            f"- sample: {rec['sample'] or '(none)'}"
        )
    return report.join_sections(
        [
            "# Dataset inspection",
            table,
            "\n\n".join(details),
        ]
    )


def main() -> int:
    args = cli.parser("Inspect checked-in CSVs.").parse_args()
    records = [inspect(name) for name in paths.TABLES]
    body = render(records)
    print(body)
    written = cli.maybe_write(args, "dataset_inspection.md", body)
    if written:
        print(f"\nwrote {written}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
