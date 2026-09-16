#!/usr/bin/env python3
"""Class histograms and majority-class baselines for every sentiment table."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from examples.lib import cli, features, io_csv, paths, report, schema, stats


def summarize(name: str) -> dict:
    rows = io_csv.read_rows(paths.table_path(name))
    labels = features.labels_of(rows)
    tallies = {i: 0 for i in (0, 1, 2)}
    for lab in labels:
        tallies[lab] = tallies.get(lab, 0) + 1
    winner, baseline = stats.majority_baseline(labels)
    return {
        "name": name,
        "n": len(labels),
        "neg": tallies[0],
        "neu": tallies[1],
        "pos": tallies[2],
        "majority": f"{schema.LABEL_NAMES[winner]} ({winner})",
        "baseline": baseline,
    }


def render(records: list[dict]) -> str:
    table = report.ascii_table(
        ("table", "n", "neg", "neu", "pos", "majority", "majority acc"),
        [
            (
                r["name"],
                r["n"],
                r["neg"],
                r["neu"],
                r["pos"],
                r["majority"],
                r["baseline"],
            )
            for r in records
        ],
    )
    notes = [
        "Majority accuracy is the number a classifier must beat.",
        "ZuCo combined is nearly balanced; the 40-row valid/test slices are not.",
        "Full SST is short on neutrals (~19%), so prefer macro F1 over weighted F1.",
    ]
    return report.join_sections(["# Label balance", table, report.bullet(notes)])


def main() -> int:
    args = cli.parser("Print sentiment label histograms.").parse_args()
    records = [summarize(name) for name in paths.SENTIMENT_TABLES]
    body = render(records)
    print(body)
    written = cli.maybe_write(args, "label_balance.md", body)
    if written:
        print(f"\nwrote {written}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
