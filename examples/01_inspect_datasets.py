#!/usr/bin/env python3
"""Print schemas, row counts, and label balance for every committed CSV.

Run from anywhere; paths are resolved from the repository root.

    python3 examples/01_inspect_datasets.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import (
    format_counts,
    label_counts,
    read_csv,
    repo_root,
    table,
    whitespace_token_count,
    write_text,
)

DATASETS = [
    ("ZuCo ∩ SST labels only", "ZuCo_SST_data/ssts_ZuCo.csv", True),
    ("ZuCo ∩ SST standard gaze", "ZuCo_SST_data/combined_sst_et_standard.csv", True),
    ("ZuCo ∩ SST min-max gaze", "ZuCo_SST_data/combined_sst_et_min_max.csv", True),
    ("ZuCo leftover train split", "ZuCo_SST_data/train.csv", True),
    ("ZuCo leftover valid split", "ZuCo_SST_data/valid.csv", True),
    ("ZuCo leftover test split", "ZuCo_SST_data/test.csv", True),
    ("Full SST + transferred gaze", "SST_data/combined_full_sst_et.csv", True),
    ("Full SST train", "SST_data/train_full_sst.csv", True),
    ("Full SST valid", "SST_data/valid_full_sst.csv", True),
    ("Full SST test", "SST_data/test_full_sst.csv", True),
    ("SST sentences (no header)", "SST_data/stts_all_sentence_level.csv", False),
    ("ZuCo subject 1 raw", "ZuCo_et_csv_data/1_SR.csv", True),
    ("ZuCo subject 3 raw (truncated)", "ZuCo_et_csv_data/3_SR.csv", True),
    ("ZuCo 12-reader average", "ZuCo_et_csv_data/average_data.csv", True),
    ("Word-level averages v2", "ZuCo_et_csv_data/word/word_averages_v2.csv", True),
    ("Gaze prediction test", "gaze_prediction/data/prediction_test.csv", True),
    ("PROVO helper", "gaze_prediction/data/provo.csv", True),
]


def inspect(title: str, relative: str, has_header: bool) -> list[str]:
    header, rows = read_csv(relative, has_header=has_header)
    lines = [
        f"## {title}",
        f"path: {relative}",
        f"rows: {len(rows)}",
    ]
    if has_header:
        lines.append(f"cols ({len(header)}): {', '.join(header)}")
    else:
        width = len(rows[0]) if rows else 0
        lines.append(f"cols: {width} (file has no header; first row is data)")

    if has_header and "sentiment_label" in header:
        idx = header.index("sentiment_label")
        labels = [int(float(row[idx])) for row in rows]
        lines.append(f"labels: {format_counts(label_counts(labels))}")
    elif not has_header and rows and len(rows[0]) >= 2:
        raw = [row[-1].strip().strip('"') for row in rows]
        # stts file uses POSITIVE / NEUTRAL / NEGATIVE
        mapped = []
        mapping = {"NEGATIVE": 0, "NEUTRAL": 1, "POSITIVE": 2}
        unknown = 0
        for value in raw:
            if value in mapping:
                mapped.append(mapping[value])
            else:
                unknown += 1
        if mapped:
            lines.append(f"labels (mapped): {format_counts(label_counts(mapped))}")
        if unknown:
            lines.append(f"unmapped label strings: {unknown}")

    sentence_idx = None
    if has_header and "sentence" in header:
        sentence_idx = header.index("sentence")
    elif not has_header:
        sentence_idx = 0
    if sentence_idx is not None and rows:
        lengths = [whitespace_token_count(row[sentence_idx]) for row in rows]
        lines.append(
            "tokens (whitespace): "
            f"min={min(lengths)}  mean={sum(lengths) / len(lengths):.2f}  "
            f"max={max(lengths)}"
        )

    if has_header and "Word" in header:
        idx = header.index("Word")
        empty = sum(1 for row in rows if not row[idx].strip())
        lines.append(f"empty Word cells: {empty}")

    lines.append("")
    return lines


def main() -> int:
    root = repo_root()
    banner = [
        "# Dataset inspection",
        f"repo: {root}",
        "",
    ]
    summary_rows = []
    body: list[str] = []
    for title, relative, has_header in DATASETS:
        path = root / relative
        if not path.exists():
            body.extend([f"## {title}", f"MISSING: {relative}", ""])
            summary_rows.append((title, relative, "MISSING", "-", "-"))
            continue
        section = inspect(title, relative, has_header)
        body.extend(section)
        header, rows = read_csv(relative, has_header=has_header)
        n_cols = len(header) if has_header else (len(rows[0]) if rows else 0)
        labels = "-"
        if has_header and "sentiment_label" in header:
            idx = header.index("sentiment_label")
            labels = format_counts(label_counts(int(float(r[idx])) for r in rows))
        summary_rows.append((title, relative, str(len(rows)), str(n_cols), labels))

    summary = table(
        ["dataset", "path", "rows", "cols", "labels"],
        summary_rows,
    )
    text = "\n".join(banner + [summary, "", *body])
    print(text)
    out = write_text("01_inspect_datasets.txt", text)
    print(f"\nwrote {out.relative_to(root)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
