#!/usr/bin/env python3
"""Print schema and row counts for every documented CSV."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from examples.lib.loading import documented_datasets, load_dataset, load_sst_raw
from examples.lib.paths import resolve_root
from examples.lib.reporting import banner, print_frame

import pandas as pd


def inspect(root: Path) -> pd.DataFrame:
    rows = []
    for spec in documented_datasets():
        df = load_dataset(spec, root=root)
        rows.append(
            {
                "key": spec.key,
                "path": spec.relative_path,
                "rows": len(df),
                "expected_rows": spec.expected_rows,
                "rows_match": spec.expected_rows is None or len(df) == spec.expected_rows,
                "n_columns": df.shape[1],
                "columns": ", ".join(df.columns.astype(str)),
                "notes": spec.notes,
            }
        )
    raw = load_sst_raw(root=root)
    rows.append(
        {
            "key": "sst_raw",
            "path": "SST_data/stts_all_sentence_level.csv",
            "rows": len(raw),
            "expected_rows": 11853,
            "rows_match": len(raw) == 11853,
            "n_columns": raw.shape[1],
            "columns": ", ".join(raw.columns.astype(str)),
            "notes": "Headerless SST dump (sentence, POSITIVE|NEGATIVE|NEUTRAL).",
        }
    )
    return pd.DataFrame(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=None, help="Repository root (default: auto).")
    args = parser.parse_args()
    root = resolve_root(args.root)

    banner("Documented datasets")
    table = inspect(root)
    print_frame(table[["key", "rows", "expected_rows", "rows_match", "n_columns", "path"]])
    print()
    for _, row in table.iterrows():
        print(f"- {row['key']}: {row['notes']}")
        print(f"    columns: {row['columns']}")

    mismatches = table.loc[~table["rows_match"]]
    if not mismatches.empty:
        banner("Row-count mismatches")
        print_frame(mismatches[["key", "rows", "expected_rows", "path"]])
        return 1
    print("\nAll documented row counts match this checkout.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
