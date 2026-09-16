#!/usr/bin/env python3
"""Partition integrity for both experiment tracks."""

from __future__ import annotations

from _util import maybe_write, parser
from zuco_lab.reports import join_sections, md_heading, md_table
from zuco_lab.splits import audit_sst, audit_zuco, text_overlap


def _audit_rows(audit) -> list[tuple]:
    return [
        ("train / valid / test", f"{audit.n_train} / {audit.n_valid} / {audit.n_test}"),
        ("combined", audit.n_combined),
        ("train∩valid", audit.train_valid_overlap),
        ("train∩test", audit.train_test_overlap),
        ("valid∩test", audit.valid_test_overlap),
        ("missing from splits", audit.missing_from_splits),
        ("extra in splits", audit.extra_in_splits),
        ("combined labels 0/1/2", f"{audit.combined_labels[0]} / {audit.combined_labels[1]} / {audit.combined_labels[2]}"),
        ("train labels 0/1/2", f"{audit.train_labels[0]} / {audit.train_labels[1]} / {audit.train_labels[2]}"),
        ("valid labels 0/1/2", f"{audit.valid_labels[0]} / {audit.valid_labels[1]} / {audit.valid_labels[2]}"),
        ("test labels 0/1/2", f"{audit.test_labels[0]} / {audit.test_labels[1]} / {audit.test_labels[2]}"),
    ]


def build() -> str:
    zuco = audit_zuco()
    sst = audit_sst()
    shared = text_overlap()
    overlap_rows = [
        (text[:72] + ("…" if len(text) > 72 else ""), z, s, "yes" if z == s else "no")
        for text, z, s in shared
    ]
    return join_sections(
        [
            md_heading("Split and schema audit", 1),
            md_heading("ZuCo–SST (400)"),
            md_table(("check", "value"), _audit_rows(zuco)),
            (
                "``model_ZuCo_SST.py`` ignores these 320/40/40 files and runs "
                "stratified 5-fold on the combined table. The 40-row valid split "
                "is not stratified (7 / 14 / 19). A 5-point accuracy swing there "
                "is two sentences."
            ),
            md_heading("Full SST (11,853)"),
            md_table(("check", "value"), _audit_rows(sst)),
            (
                "These files *are* what ``model_full_SST.py`` trains on. Neutral "
                "is under-represented. See example 07 before quoting the printed "
                "test score."
            ),
            md_heading("Normalized text overlap between tracks"),
            md_table(("sentence", "ZuCo label", "SST label", "agree"), overlap_rows),
            (
                "Only three strings match after light normalization. The 400-row "
                "ZuCo overlap is not a subset of ``combined_full_sst_et.csv`` in "
                "any useful sense — different tokenization and a different SST cut."
            ),
        ]
    )


def main() -> None:
    args = parser("Audit train/valid/test partitions and cross-track overlap.").parse_args()
    text = build()
    print(text)
    maybe_write(args, "06_split_and_schema.md", text)


if __name__ == "__main__":
    main()
