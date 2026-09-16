#!/usr/bin/env python3
"""Reproduce the full-SST test-loop overwrite with a fake loader."""

from __future__ import annotations

from _util import maybe_write, parser
from zuco_lab.bugs import KNOWN_BUGS, simulate_test_loop
from zuco_lab.reports import join_sections, md_heading, md_table


def build() -> str:
    demo = simulate_test_loop()
    bug_rows = [(b["id"], b["file"], b["severity"], b["summary"]) for b in KNOWN_BUGS]
    return join_sections(
        [
            md_heading("Full-SST test-loop overwrite", 1),
            (
                "In ``model_full_SST.py`` the validation loop extends prediction "
                "lists. The test loop assigns ``all_preds = preds.cpu().numpy()``, "
                "so only the last batch survives. Default ``batch_size=256`` and "
                f"{demo.n_items} test rows → last batch is {demo.overwrite_covers} examples."
            ),
            md_heading("Simulated numbers"),
            md_table(
                ("quantity", "value"),
                [
                    ("items", demo.n_items),
                    ("batch size", demo.batch_size),
                    ("batches", demo.n_batches),
                    ("overwrite coverage", demo.overwrite_covers),
                    ("full-set accuracy (extend)", demo.extend_accuracy),
                    ("printed-style accuracy (overwrite)", demo.overwrite_accuracy),
                ],
            ),
            demo.note,
            (
                "The dummy model is correct everywhere except the last batch, so "
                "the two aggregation rules disagree on purpose. The real network "
                "will not look like this; the point is coverage, not the dummy "
                "accuracy values."
            ),
            md_heading("Other known issues (not patched here)"),
            md_table(("id", "file", "severity", "summary"), bug_rows),
        ]
    )


def main() -> None:
    args = parser("Demonstrate the test-loop overwrite in model_full_SST.py.").parse_args()
    text = build()
    print(text)
    maybe_write(args, "07_test_loop_bug.md", text)


if __name__ == "__main__":
    main()
