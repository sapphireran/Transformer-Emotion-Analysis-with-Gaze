#!/usr/bin/env python3
"""Validate required columns and numeric gaze fields on every named table."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from examples.lib import cli, io_csv, paths, report, schema


def check_one(name: str) -> schema.SchemaReport:
    path = paths.table_path(name)
    header = io_csv.read_header(path)
    rows = io_csv.read_rows(path)
    return schema.validate_rows(name, header, rows)


def render(reports: list[schema.SchemaReport]) -> str:
    rows = []
    details = []
    for item in reports:
        status = "ok" if item.ok else "FAIL"
        n_err = sum(1 for i in item.issues if i.level == "error")
        n_warn = sum(1 for i in item.issues if i.level == "warning")
        rows.append((item.name, status, n_err, n_warn))
        if item.issues:
            lines = "\n".join(f"  - [{iss.level}] {iss.message}" for iss in item.issues)
            details.append(f"### {item.name}\n\n{lines}")
    body = [
        "# Schema check",
        report.ascii_table(("table", "status", "errors", "warnings"), rows),
    ]
    if details:
        body.append("\n\n".join(details))
    else:
        body.append("No extra columns and no type errors in the sampled rows.")
    return report.join_sections(body)


def main() -> int:
    args = cli.parser("Validate CSV schemas.").parse_args()
    reports = [check_one(name) for name in paths.TABLES]
    body = render(reports)
    print(body)
    written = cli.maybe_write(args, "schema_check.md", body)
    if written:
        print(f"\nwrote {written}")
    return 0 if all(item.ok for item in reports) else 1


if __name__ == "__main__":
    raise SystemExit(main())
