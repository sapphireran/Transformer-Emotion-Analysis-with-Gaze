#!/usr/bin/env python3
"""List the consumer CSVs that are actually in this clone."""

from __future__ import annotations

from _util import maybe_write, parser
from zuco_lab.catalogs import FILE_NOTES, TRACKS
from zuco_lab.inventory import inventory
from zuco_lab.reports import join_sections, md_heading, md_table


def build() -> str:
    items = inventory()
    rows = [
        (item.relpath, item.rows, "yes" if item.schema_ok else "no", item.note)
        for item in items
        if not item.key.startswith("subject_")
    ]
    subject_rows = [
        (item.relpath, item.rows, "yes" if item.schema_ok else "no", item.note)
        for item in items
        if item.key.startswith("subject_")
    ]
    track_rows = [
        (t.title, t.n_sentences, t.script, t.gaze_source)
        for t in TRACKS
    ]
    note_rows = [(n.path, n.kind, n.rows, n.note) for n in FILE_NOTES]
    return join_sections(
        [
            md_heading("Repository inventory", 1),
            "Personal checklist of tables this clone can consume without MATLAB.",
            md_heading("Experiment tracks"),
            md_table(("track", "sentences", "script", "gaze"), track_rows),
            md_heading("Consumer tables"),
            md_table(("path", "rows", "schema", "note"), rows),
            md_heading("Twelve readers"),
            md_table(("path", "rows", "schema", "note"), subject_rows),
            md_heading("File notes"),
            md_table(("path", "kind", "rows", "note"), note_rows),
        ]
    )


def main() -> None:
    args = parser("Inventory the checked-in gaze/sentiment tables.").parse_args()
    text = build()
    print(text)
    maybe_write(args, "01_repo_inventory.md", text)


if __name__ == "__main__":
    main()
