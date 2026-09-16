#!/usr/bin/env python3
"""Check that train/valid/test partitions cover the combined table without overlap."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from examples.lib import cli, io_csv, paths, report


def ids_of(name: str) -> list[str]:
    rows = io_csv.read_rows(paths.table_path(name))
    return [row["sentence_id"] for row in rows]


def audit(group: str, combined: str, train: str, valid: str, test: str) -> dict:
    c, tr, va, te = map(ids_of, (combined, train, valid, test))
    sc, str_, sva, ste = map(set, (c, tr, va, te))
    issues = []
    if len(c) != len(sc):
        issues.append(f"combined has {len(c) - len(sc)} duplicate sentence_id values")
    if len(tr) + len(va) + len(te) != len(c):
        issues.append(
            f"row counts {len(tr)}+{len(va)}+{len(te)} != combined {len(c)}"
        )
    overlap_tv = str_ & sva
    overlap_tt = str_ & ste
    overlap_vt = sva & ste
    if overlap_tv:
        issues.append(f"train ∩ valid = {len(overlap_tv)} ids")
    if overlap_tt:
        issues.append(f"train ∩ test = {len(overlap_tt)} ids")
    if overlap_vt:
        issues.append(f"valid ∩ test = {len(overlap_vt)} ids")
    missing = sc - (str_ | sva | ste)
    extra = (str_ | sva | ste) - sc
    if missing:
        issues.append(f"{len(missing)} combined ids absent from the splits")
    if extra:
        issues.append(f"{len(extra)} split ids absent from combined")
    return {
        "group": group,
        "combined": len(c),
        "train": len(tr),
        "valid": len(va),
        "test": len(te),
        "ok": not issues,
        "issues": issues,
    }


def render(results: list[dict]) -> str:
    table = report.ascii_table(
        ("group", "combined", "train", "valid", "test", "status"),
        [
            (
                r["group"],
                r["combined"],
                r["train"],
                r["valid"],
                r["test"],
                "ok" if r["ok"] else "FAIL",
            )
            for r in results
        ],
    )
    details = []
    for r in results:
        if r["issues"]:
            details.append(f"### {r['group']}\n\n" + report.bullet(r["issues"]))
        else:
            details.append(
                f"### {r['group']}\n\n"
                "Ids are unique, splits are disjoint, and they partition the combined table."
            )
    return report.join_sections(["# Split audit", table, "\n\n".join(details)])


def main() -> int:
    args = cli.parser("Audit train/valid/test partitions.").parse_args()
    results = [
        audit("zuco", *paths.SPLIT_GROUPS["zuco"]),
        audit("sst", *paths.SPLIT_GROUPS["sst"]),
    ]
    body = render(results)
    print(body)
    written = cli.maybe_write(args, "split_audit.md", body)
    if written:
        print(f"\nwrote {written}")
    return 0 if all(r["ok"] for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
