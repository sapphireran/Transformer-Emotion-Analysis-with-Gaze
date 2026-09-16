#!/usr/bin/env python3
"""Show that reader 3 is remapped after sentence 149.

This is the main finding in the personal lab notes: ``3_SR.csv`` has 299
rows whose ``id`` is compacted, so an index-wise average of the twelve
readers mixes the wrong sentence into ids 150–298.
"""

from __future__ import annotations

from _util import maybe_write, parser
from zuco_lab.numbers import mean
from zuco_lab.reports import join_sections, md_heading, md_table
from zuco_lab.subjects import (
    alignment_regions,
    compact_to_original_subject3,
    contamination_table,
    first_mismatch_id,
    remapped_sentlen_matches,
)


def _region_summary(rows, lo: int, hi: int) -> tuple[int, float, float]:
    slice_ = [r for r in rows if lo <= r.sentence_id <= hi]
    if not slice_:
        return 0, 0.0, 0.0
    return len(slice_), mean([abs(r.delta) for r in slice_]), max(abs(r.delta) for r in slice_)


def build() -> str:
    mismatch = first_mismatch_id()
    remap_ok = remapped_sentlen_matches()
    nfix = contamination_table("nFixations")
    slen = contamination_table("SentLen")
    dirty = [r for r in nfix if abs(r.delta) > 1e-12]

    region_rows = []
    for name, (lo, hi) in alignment_regions().items():
        n, avg, peak = _region_summary(nfix, lo, hi)
        region_rows.append((name, f"{lo}–{hi}", n, avg, peak))

    sample_ids = [0, 149, 150, 250, 298, 398, 399]
    sample_rows = []
    slen_map = {r.sentence_id: r for r in slen}
    nfix_map = {r.sentence_id: r for r in nfix}
    for sid in sample_ids:
        s = slen_map.get(sid)
        n = nfix_map.get(sid)
        if not s or not n:
            sample_rows.append((sid, "—", "—", "—", "—"))
            continue
        sample_rows.append(
            (
                sid,
                s.index_mean,
                s.aligned_mean,
                n.index_mean,
                n.aligned_mean,
            )
        )

    compact_rows = [(c, compact_to_original_subject3(c)) for c in (0, 149, 150, 298)]

    return join_sections(
        [
            md_heading("Subject-3 alignment audit", 1),
            (
                "``utils_ZuCo.DataTransformer`` skips original sentences 150–249 and 399 "
                "for task-1 subject index 2 (file ``3_SR.csv``), then writes the remaining "
                "299 rows with compacted ids 0..298. ``get_average_sentence_level.py`` "
                "averages the twelve CSVs by row index."
            ),
            md_heading("Sanity checks"),
            md_table(
                ("check", "value"),
                [
                    ("first SentLen mismatch vs reader 1", mismatch),
                    ("remapped SentLen matches reader 1", "yes" if remap_ok else "no"),
                    ("compact 150 maps to original", compact_to_original_subject3(150)),
                    ("sentences with nFixations contamination", len(dirty)),
                ],
            ),
            md_heading("Compact id → original id"),
            md_table(("compact id in 3_SR.csv", "original sentence id"), compact_rows),
            md_heading("nFixations contamination by region"),
            md_table(("region", "ids", "n", "mean abs delta", "max abs delta"), region_rows),
            md_heading("Worked ids"),
            md_table(
                ("id", "SentLen index-mean", "SentLen aligned", "nFix index-mean", "nFix aligned"),
                sample_rows,
            ),
            md_heading("How to read this"),
            (
                "Ids 0–149 are safe: all twelve readers still share the same sentence. "
                "From 150 onward, reader 3's compact id is a later sentence. The "
                "published ``average_data.csv`` follows the index mean, so it is "
                "slightly wrong on 246 sentences. The shift is small on nFixations "
                "(mean |delta| ≈ 0.03) because one reader out of twelve cannot move "
                "the mean far, but it is a real alignment bug, not noise."
            ),
        ]
    )


def main() -> None:
    args = parser("Audit subject-3 remapping against the other 11 readers.").parse_args()
    text = build()
    print(text)
    maybe_write(args, "02_subject_alignment.md", text)


if __name__ == "__main__":
    main()
