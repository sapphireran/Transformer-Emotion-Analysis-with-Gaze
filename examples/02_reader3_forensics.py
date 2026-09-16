#!/usr/bin/env python3
"""Show how compacted reader-3 ids leak into the published sentence means."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gazebook.csvio import read_dicts
from gazebook.paths import repo_root
from gazebook.remap import (
    CLEAN_SENTENCE_END,
    NFIX_CONTAMINATED,
    READER3_COMPACT_ROWS,
    SENTLEN_CONTAMINATED,
    WORD_ALIGN_ROWS,
    WORST_NFIX_ID,
    compact_to_original,
    contamination_report,
    load_subject_tables,
    original_to_compact,
    positional_average,
    published_column,
    word_align_row_count,
    word_mismatch_count,
)
from gazebook.reports import md_table, write_text


def main() -> int:
    root = repo_root()
    tables = load_subject_tables(root)
    assert len(tables[2]) == READER3_COMPACT_ROWS

    pub_nfix = published_column(root, "nFixations")
    pub_len = published_column(root, "SentLen")
    pos_nfix = positional_average(tables, "nFixations")
    rebuild_err = float(abs(pos_nfix - pub_nfix).max())

    nfix = contamination_report(tables, pub_nfix, "nFixations")
    slen = contamination_report(tables, pub_len, "SentLen")

    # Smoking gun: sentence length is not a gaze feature. If the published
    # SentLen at id 150 is 9.5, reader 3 contributed a 15-word sentence.
    s1_len = float(tables[0][150]["SentLen"])
    s3_len = float(tables[2][150]["SentLen"])
    pub150 = float(pub_len[150])
    mapped = compact_to_original(150)

    w_div = word_align_row_count(root)
    w_mis = word_mismatch_count(root)

    print("Reader 3 compact rows:", len(tables[2]))
    print("compact 150 → original", mapped, "(MATLAB sentence 250)")
    print(f"SentLen @150: reader1={s1_len:.1f} reader3_compact={s3_len:.1f} published={pub150:.1f}")
    print(f"Expected mix (11×{s1_len:.0f} + 1×{s3_len:.0f})/12 = {(11 * s1_len + s3_len) / 12:.1f}")
    print(f"Positional 0→NaN rebuild vs average_data.csv max|Δ| nFixations = {rebuild_err:.3e}")
    print(
        f"Remapped vs published nFixations: {nfix.n_changed} sentences, "
        f"mean|Δ|={nfix.mean_abs:.4f}, max|Δ|={nfix.max_abs:.4f} at id {nfix.worst_id}"
    )
    print(f"Remapped vs published SentLen: {slen.n_changed} sentences (length is the tell)")
    print(f"Word streams first diverge at row {w_div} (expected {WORD_ALIGN_ROWS})")
    print(f"Reader1 vs reader3 word mismatches in the 5293-row overlap: {w_mis}")

    # A few compact → original examples around the cut.
    demo = []
    for compact in (0, 149, 150, 151, 298):
        orig = compact_to_original(compact)
        demo.append(
            [
                compact,
                orig,
                original_to_compact(orig),
                float(tables[0][orig]["SentLen"]) if orig < len(tables[0]) else "",
                float(tables[2][compact]["SentLen"]),
            ]
        )
    print()
    print(md_table(["compact id", "original id", "round-trip", "reader1 SentLen", "reader3 SentLen"], demo))

    lines = [
        "# Reader-3 forensics",
        "",
        f"Clean 12-reader overlap: original ids 0–{CLEAN_SENTENCE_END - 1}.",
        f"Published `average_data.csv` matches the positional 0→NaN mean (max|Δ|={rebuild_err:.3e}).",
        f"Remapping reader 3 changes nFixations on **{nfix.n_changed}** sentences "
        f"(locked target {NFIX_CONTAMINATED}); worst id **{nfix.worst_id}** "
        f"(locked {WORST_NFIX_ID}), max|Δ|={nfix.max_abs:.4f}.",
        f"SentLen disagrees on **{slen.n_changed}** sentences (locked {SENTLEN_CONTAMINATED}).",
        f"Id 150 published SentLen = {pub150:.1f} because compact row 150 is original {mapped}.",
        f"Word-level index average is aligned for the first **{w_div}** tokens, then mixes words.",
        "",
    ]
    write_text(root / "examples/output/02_reader3_forensics.md", "\n".join(lines))

    # Soft lock so a silent data edit fails the example, not only the unit tests.
    if nfix.n_changed != NFIX_CONTAMINATED:
        print(f"UNEXPECTED nFix contaminated count {nfix.n_changed}", file=sys.stderr)
        return 1
    if slen.n_changed != SENTLEN_CONTAMINATED:
        print(f"UNEXPECTED SentLen contaminated count {slen.n_changed}", file=sys.stderr)
        return 1
    if w_div != WORD_ALIGN_ROWS:
        print(f"UNEXPECTED word diverge row {w_div}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
