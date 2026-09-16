"""Document Task 1 subject-3 compacted ids vs the other 11 subjects."""

from __future__ import annotations

import _bootstrap  # noqa: F401

import sys

from teag_examples.alignment import (
    SUBJECT_3_SKIP_SET,
    compacted_to_original_subject3,
    original_to_compacted_subject3,
    skip_list_matches_transformer,
    subject3_alignment_report,
)
from teag_examples.io import load_subject_sentence_gaze
from teag_examples.paths import examples_output
from teag_examples.reports import write_json, write_markdown
from teag_examples.schema import SUBJECT_3_SKIP_ORIGINAL, ZUCO_N_SENTENCES


def main(argv: list[str] | None = None) -> int:
    del argv
    report = subject3_alignment_report()
    s1 = load_subject_sentence_gaze(1)
    s3 = load_subject_sentence_gaze(3)

    roundtrip_ok = True
    for orig in range(ZUCO_N_SENTENCES):
        compact = original_to_compacted_subject3(orig)
        if orig in SUBJECT_3_SKIP_SET:
            if compact is not None:
                roundtrip_ok = False
            continue
        back = compacted_to_original_subject3(compact)
        if back != orig:
            roundtrip_ok = False
            break

    md = f"""# Subject 3 row alignment

`utils_ZuCo.DataTransformer` skips original Task 1 sentences
{SUBJECT_3_SKIP_ORIGINAL[0]}–{SUBJECT_3_SKIP_ORIGINAL[99]} and {SUBJECT_3_SKIP_ORIGINAL[-1]}
for subject index 2 (file `3_SR.csv`), then reindexes from 0.

- rows in `3_SR.csv`: {report['n_subject3']}
- rows in `1_SR.csv`: {report['n_reference']}
- first `SentLen` mismatch vs subject 1: row {report['first_mismatch_row']}
- aligned prefix: {report['aligned_prefix_rows']} rows
- compacted row 150 matches subject-1 original sentence 250: {report['compacted_150_matches_original_250']}
- skip list length 101 ⇒ 299 remaining: {skip_list_matches_transformer()}
- compacted ↔ original round-trip: {roundtrip_ok}

`get_average_sentence_level.py` averages by row number, so after row 149
subject 3 contributes a **different sentence** than the other eleven subjects.

Subject 1 `SentLen[150:155]` = {s1['SentLen'].iloc[150:155].tolist()}
Subject 3 `SentLen[150:155]` = {s3['SentLen'].iloc[150:155].tolist()}
Subject 1 `SentLen[250:255]` = {s1['SentLen'].iloc[250:255].tolist()}
"""
    write_markdown(md, examples_output() / "subject3_alignment.md")
    write_json(
        {**report, "roundtrip_ok": roundtrip_ok, "skip_list_ok": skip_list_matches_transformer()},
        examples_output() / "subject3_alignment.json",
    )
    sys.stdout.write(md + "\n")
    if not (
        report["first_mismatch_row"] == 150
        and report["compacted_150_matches_original_250"]
        and roundtrip_ok
    ):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
