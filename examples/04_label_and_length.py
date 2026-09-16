#!/usr/bin/env python3
"""Length buckets vs sentiment, plus a few qualitative sentence examples.

Useful when you want to know whether a gaze–label correlation is just
"longer reviews are more negative" in disguise.

    python3 examples/04_label_and_length.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np

from common import (
    LABEL_NAMES,
    ZUCO_GAZE_COLS,
    as_int_column,
    format_counts,
    label_counts,
    matrix,
    read_csv,
    table,
    whitespace_token_count,
    write_text,
)

BUCKETS = [
    ("1-8 tokens", 1, 8),
    ("9-16 tokens", 9, 16),
    ("17-24 tokens", 17, 24),
    ("25-32 tokens", 25, 32),
    ("33+ tokens", 33, 10_000),
]


def bucket_name(n: int) -> str:
    for name, lo, hi in BUCKETS:
        if lo <= n <= hi:
            return name
    return "other"


def crosstab(lengths: list[int], labels: np.ndarray) -> str:
    names = [b[0] for b in BUCKETS]
    counts = {name: {0: 0, 1: 0, 2: 0} for name in names}
    for n, y in zip(lengths, labels):
        counts[bucket_name(n)][int(y)] += 1
    rows = []
    for name in names:
        total = sum(counts[name].values())
        cells = [name, str(total)]
        for lab in (0, 1, 2):
            c = counts[name][lab]
            pct = f"{c} ({(100 * c / total):.1f}%)" if total else "0"
            cells.append(pct)
        rows.append(cells)
    return table(["bucket", "n", "neg", "neu", "pos"], rows)


def length_by_class(lengths: list[int], labels: np.ndarray) -> str:
    arr = np.asarray(lengths, dtype=np.float64)
    rows = []
    for lab in (0, 1, 2):
        part = arr[labels == lab]
        rows.append(
            (
                f"{lab} {LABEL_NAMES[lab]}",
                str(int(part.size)),
                f"{part.mean():.2f}",
                f"{np.median(part):.1f}",
                f"{part.min():.0f}",
                f"{part.max():.0f}",
            )
        )
    return table(["class", "n", "mean", "median", "min", "max"], rows)


def gaze_vs_length(x: np.ndarray, lengths: list[int], names: list[str]) -> str:
    arr = np.asarray(lengths, dtype=np.float64)
    rows = []
    for j, name in enumerate(names):
        if arr.std() == 0 or np.nanstd(x[:, j]) == 0:
            corr = float("nan")
        else:
            corr = float(np.corrcoef(arr, x[:, j])[0, 1])
        rows.append((name, f"{corr:.3f}"))
    return table(["fusion feature", "corr with token count"], rows)


def pick_examples(header: list[str], rows: list[list[str]], labels: np.ndarray) -> str:
    sent_idx = header.index("sentence")
    # one short and one long sentence per class, deterministic: first match
    picked = []
    for lab in (0, 1, 2):
        candidates = [(i, whitespace_token_count(rows[i][sent_idx])) for i in range(len(rows)) if labels[i] == lab]
        candidates.sort(key=lambda t: t[1])
        short_i = candidates[0][0]
        long_i = candidates[-1][0]
        picked.append((lab, "shortest", rows[short_i][sent_idx], candidates[0][1]))
        picked.append((lab, "longest", rows[long_i][sent_idx], candidates[-1][1]))
    lines = []
    for lab, kind, sent, n in picked:
        lines.append(f"- [{lab} {LABEL_NAMES[lab]} / {kind} / {n} tokens] {sent}")
    return "\n".join(lines)


def analyze(title: str, relative: str, gaze_cols: list[str]) -> str:
    header, rows = read_csv(relative)
    labels = as_int_column(rows, header, "sentiment_label")
    sent_idx = header.index("sentence")
    lengths = [whitespace_token_count(row[sent_idx]) for row in rows]
    x = matrix(rows, header, gaze_cols)
    return "\n".join(
        [
            f"## {title}",
            f"path: {relative}",
            f"labels: {format_counts(label_counts(labels))}",
            "",
            "### tokens by class",
            length_by_class(lengths, labels),
            "",
            "### length bucket × sentiment",
            crosstab(lengths, labels),
            "",
            "### correlation of each fusion feature with token count",
            gaze_vs_length(x, lengths, gaze_cols),
            "",
            "### example sentences (shortest / longest per class)",
            pick_examples(header, rows, labels),
            "",
        ]
    )


def main() -> int:
    text = "\n".join(
        [
            "# Labels vs sentence length",
            "",
            analyze(
                "ZuCo ∩ SST (standard)",
                "ZuCo_SST_data/combined_sst_et_standard.csv",
                ZUCO_GAZE_COLS,
            ),
            analyze(
                "Full SST train (transferred gaze)",
                "SST_data/train_full_sst.csv",
                ["nFix", "FFD", "GPT", "TRT", "GD"],
            ),
        ]
    )
    print(text)
    write_text("04_label_and_length.txt", text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
