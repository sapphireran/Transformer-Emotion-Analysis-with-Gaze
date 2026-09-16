#!/usr/bin/env python3
"""Show that ZuCo subject 3 is reindexed, so row-wise averages drift.

DataTransformer (task 1, subject index 2, filename 3_SR.csv) drops
original sentences 150-249 and 399, then writes a fresh 0..298 index.
``get_average_sentence_level.py`` then ``groupby(level=0).mean()``s the
twelve files. Sentences 0-149 are fine. From id 150 onward, subject 3's
row is a *different* sentence than everyone else's.

This script does not rewrite the training CSVs. It only measures the
shift (SentLen is a perfect alignment key) and writes an optional remapped
copy under --output-dir.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from examples.lib.loading import (
    SUBJECT_3_DROPPED_ORIGINAL_IDS,
    SUBJECT_SENTENCE_ROWS,
    load_subject_sentence_tables,
    remap_subject3_original_ids,
)
from examples.lib.paths import resolve_root
from examples.lib.reporting import banner, print_frame

import pandas as pd


def sentlen_match_rate(left: pd.DataFrame, right: pd.DataFrame) -> float:
    merged = left.merge(right, on="id", suffixes=("_a", "_b"))
    if merged.empty:
        return float("nan")
    return float((merged["SentLen_a"] == merged["SentLen_b"]).mean())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=None)
    parser.add_argument(
        "--output-dir",
        default=None,
        help="If set, write remapped subject-3 CSV and a JSON note.",
    )
    args = parser.parse_args()
    root = resolve_root(args.root)

    tables = load_subject_sentence_tables(root=root)
    counts = {subject: len(df) for subject, df in tables.items()}

    banner("Per-subject sentence-table lengths")
    count_df = pd.DataFrame(
        {"subject": list(counts), "rows": list(counts.values()), "expected": [SUBJECT_SENTENCE_ROWS[s] for s in counts]}
    )
    print_frame(count_df)
    print(
        f"\nSubject 3 dropped original ids {SUBJECT_3_DROPPED_ORIGINAL_IDS[0]}-"
        f"{SUBJECT_3_DROPPED_ORIGINAL_IDS[99]} and 399 "
        f"({len(SUBJECT_3_DROPPED_ORIGINAL_IDS)} sentences)."
    )

    ref = tables[1]
    naive = tables[3]
    remapped = remap_subject3_original_ids(naive)

    before = ref.merge(naive, on="id", suffixes=("_s1", "_s3"))
    after = ref.merge(remapped, on="id", suffixes=("_s1", "_s3"))
    naive_early = float(
        (before.loc[before["id"] <= 149, "SentLen_s1"] == before.loc[before["id"] <= 149, "SentLen_s3"]).mean()
    )
    naive_late = float(
        (before.loc[before["id"] >= 150, "SentLen_s1"] == before.loc[before["id"] >= 150, "SentLen_s3"]).mean()
    )
    remapped_rate = float((after["SentLen_s1"] == after["SentLen_s3"]).mean())

    banner("SentLen alignment vs subject 1 (same id)")
    report = pd.DataFrame(
        [
            {
                "join": "naive id (as stored)",
                "n": len(before),
                "sentlen_match_all": sentlen_match_rate(ref, naive),
                "sentlen_match_id_0_149": naive_early,
                "sentlen_match_id_150_plus": naive_late,
            },
            {
                "join": "remapped original id",
                "n": len(after),
                "sentlen_match_all": remapped_rate,
                "sentlen_match_id_0_149": remapped_rate,
                "sentlen_match_id_150_plus": remapped_rate,
            },
        ]
    )
    print_frame(report)

    banner("Example mismatch at stored id=150")
    row = before.loc[before["id"] == 150, ["id", "SentLen_s1", "SentLen_s3"]].iloc[0]
    print(
        f"subject 1 SentLen={row['SentLen_s1']:.0f}  "
        f"subject 3 SentLen={row['SentLen_s3']:.0f}  "
        f"(subject 3's row is original sentence 250)"
    )

    print(
        "\nget_average_sentence_level.py averages on the default RangeIndex, "
        "so sentences 150-399 in average_data.csv mix eleven aligned readers "
        "with subject 3's shifted rows. Training on "
        "combined_sst_et_standard.csv inherits that. Remap before any new "
        "average; do not treat the checked-in average as subject-clean "
        "from id 150 on."
    )

    if args.output_dir:
        out = Path(args.output_dir)
        if not out.is_absolute():
            out = root / out
        out.mkdir(parents=True, exist_ok=True)
        remap_path = out / "subject3_remapped_original_ids.csv"
        remapped.to_csv(remap_path, index=False)
        note = {
            "subject3_rows": counts[3],
            "dropped_original_ids": list(SUBJECT_3_DROPPED_ORIGINAL_IDS),
            "naive_sentlen_match_id_0_149": naive_early,
            "naive_sentlen_match_id_150_plus": naive_late,
            "remapped_sentlen_match": remapped_rate,
            "remapped_csv": str(remap_path),
        }
        note_path = out / "subject3_alignment.json"
        note_path.write_text(json.dumps(note, indent=2), encoding="utf-8")
        print(f"\nwrote {remap_path}")
        print(f"wrote {note_path}")

    if naive_early != 1.0 or remapped_rate != 1.0:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
