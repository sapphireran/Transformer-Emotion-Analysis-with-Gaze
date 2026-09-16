#!/usr/bin/env python3
"""Show that ZuCo subject 3 (index 2) is missing 100 sentences.

DataTransformer skips MATLAB sentences 150–249 and 399 for task 1 subject 2.
The remaining rows are re-indexed from 0, so you cannot treat `id` as a
stable sentence key across subjects. This script prints the coverage hole
and how many subjects contribute to each averaged row index.

Usage (repo root):

    python examples/subject_coverage.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from examples.common import ZUCO_SENT_DIR, ZUCO_WORD_DIR, floats, read_dicts


def main() -> None:
    sent_counts = []
    word_counts = []
    print("=== row counts ===")
    print(f"{'file':<10} {'sentences':>10} {'words':>10}")
    for i in range(1, 13):
        sent = read_dicts(ZUCO_SENT_DIR / f"{i}_SR.csv")
        word = read_dicts(ZUCO_WORD_DIR / f"{i}_SR.csv")
        sent_counts.append(len(sent))
        word_counts.append(len(word))
        print(f"{i}_SR.csv   {len(sent):10d} {len(word):10d}")

    expected_sent = max(sent_counts)
    expected_word = max(word_counts)
    print()
    for i, n in enumerate(sent_counts, start=1):
        if n != expected_sent:
            missing = expected_sent - n
            print(
                f"Subject {i} is short {missing} sentence rows "
                f"({n} vs {expected_sent}). This matches task1 / subject "
                f"index {i - 1} skip rules in utils_ZuCo.py."
            )
    for i, n in enumerate(word_counts, start=1):
        if n != expected_word:
            print(
                f"Subject {i} is short {expected_word - n} word rows "
                f"({n} vs {expected_word})."
            )

    # How many subjects have a row at each index?
    tables = [read_dicts(ZUCO_SENT_DIR / f"{i}_SR.csv") for i in range(1, 13)]
    max_id = max(int(row["id"]) for table in tables for row in table)
    coverage = np.zeros(max_id + 1, dtype=int)
    nfix_stack = [[] for _ in range(max_id + 1)]
    for table in tables:
        for row in table:
            idx = int(row["id"])
            coverage[idx] += 1
            nfix_stack[idx].append(float(row["nFixations"]))

    print()
    print("=== subjects contributing to each averaged sentence index ===")
    unique, counts = np.unique(coverage, return_counts=True)
    for k, n in zip(unique, counts):
        print(f"{int(k):2d} subjects: {int(n):4d} sentence indices")

    short = np.where(coverage < 12)[0]
    if short.size:
        print()
        print(
            f"Indices with <12 subjects: {int(short.min())}–{int(short.max())} "
            f"({short.size} indices). Averaged CSVs still have a row there;"
        )
        print("the mean is over whoever is present at that *table index*,")
        print("which is not the same MATLAB sentence after subject 3's hole.")

    # Compare mean nFixations using only full-coverage indices vs all indices.
    full_idx = np.where(coverage == 12)[0]
    all_means = np.array([float(np.mean(v)) for v in nfix_stack if v])
    full_means = np.array([float(np.mean(nfix_stack[i])) for i in full_idx])
    print()
    print(
        f"mean nFixations over all averaged indices: {all_means.mean():.4f} "
        f"(n={all_means.size})"
    )
    print(
        f"mean nFixations over 12-subject indices:   {full_means.mean():.4f} "
        f"(n={full_means.size})"
    )

    avg_rows = read_dicts(ZUCO_SENT_DIR / "average_data.csv")
    print()
    print(
        f"average_data.csv rows={len(avg_rows)} "
        f"(matches max subject length {expected_sent}, not the 12-subject subset)."
    )


if __name__ == "__main__":
    main()
