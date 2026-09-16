"""Verify train/valid/test ids are disjoint and cover the combined tables."""

from __future__ import annotations

import _bootstrap  # noqa: F401

import sys

from teag_examples.io import (
    load_full_sst_combined,
    load_full_sst_split,
    load_zuco_combined,
    load_zuco_split,
)
from teag_examples.paths import examples_output
from teag_examples.reports import write_json, write_markdown
from teag_examples.schema import FULL_SST_SPLIT_ROWS, ZUCO_SPLIT_ROWS
from teag_examples.splits import split_inventory


def _fmt(inv: dict, expected_rows: dict) -> str:
    lines = [
        f"- train/valid/test rows: {inv['n_train']} / {inv['n_valid']} / {inv['n_test']} "
        f"(expected {expected_rows['train']} / {expected_rows['valid']} / {expected_rows['test']})",
        f"- combined rows: {inv['n_combined']}",
        f"- disjoint sentence_id: {inv['disjoint']}",
        f"- union covers combined: {inv['covers_combined']}",
        f"- overlaps (truncated): {inv['overlaps'] or '{}'}",
        f"- train labels: {inv['train_labels']}",
        f"- valid labels: {inv['valid_labels']}",
        f"- test labels: {inv['test_labels']}",
    ]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    del argv
    zuco = split_inventory(
        load_zuco_split("train"),
        load_zuco_split("valid"),
        load_zuco_split("test"),
        load_zuco_combined("standard"),
    )
    sst = split_inventory(
        load_full_sst_split("train"),
        load_full_sst_split("valid"),
        load_full_sst_split("test"),
        load_full_sst_combined(),
    )
    md = (
        "# Split sanity check\n\n"
        "## ZuCo-SST (`spilt.py` 80/10/10, unused by the CV trainer)\n\n"
        + _fmt(zuco, dict(ZUCO_SPLIT_ROWS))
        + "\n## Full SST (used by `model_full_SST.py`)\n\n"
        + _fmt(sst, dict(FULL_SST_SPLIT_ROWS))
    )
    write_markdown(md, examples_output() / "split_check.md")
    write_json({"zuco": zuco, "sst": sst}, examples_output() / "split_check.json")
    sys.stdout.write(md + "\n")
    ok = (
        zuco["disjoint"]
        and sst["disjoint"]
        and zuco["covers_combined"]
        and sst["covers_combined"]
        and zuco["n_train"] == ZUCO_SPLIT_ROWS["train"]
        and sst["n_train"] == FULL_SST_SPLIT_ROWS["train"]
    )
    if not ok:
        sys.stderr.write("split check failed\n")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
