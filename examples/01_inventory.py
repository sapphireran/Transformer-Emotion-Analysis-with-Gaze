#!/usr/bin/env python3
"""Inventory every committed table the atlas knows about."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from sidecar.paths import ROOT, expected_tables  # noqa: E402
from sidecar.reports import markdown_table, write_text  # noqa: E402


def main() -> int:
    rows = []
    missing: list[str] = []
    for name, path in expected_tables().items():
        exists = path.is_file()
        rows.append(
            {
                "name": name,
                "path": str(path.relative_to(ROOT)),
                "exists": exists,
                "bytes": path.stat().st_size if exists else 0,
            }
        )
        if not exists:
            missing.append(name)
    df = pd.DataFrame(rows)
    text = "\n".join(
        [
            "# Dataset inventory",
            "",
            "Personal checkout map. Every path is relative to the repository root.",
            "",
            markdown_table(df, floatfmt="{:.0f}"),
            "",
            f"Missing tables: {missing if missing else 'none'}.",
            "",
        ]
    )
    out = write_text("inventory.md", text)
    print(f"wrote {out} ({len(df)} rows, missing={len(missing)})")
    return 0 if not missing else 1


if __name__ == "__main__":
    raise SystemExit(main())
