#!/usr/bin/env python3
"""Rebuild subject-1 sentence nFixations from the word table and compare.

DataTransformer averages word features over words that have any non-zero
gaze value. This script repeats that rule for subject 1 and checks it
against ZuCo_et_csv_data/1_SR.csv.
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from csvutil import read_dicts
from paths import ZUCO_SUBJECT_1, ZUCO_WORD_SUBJECT_1

WORD_GAZE = ("nFixations", "meanPupilSize", "GD", "TRT", "FFD", "SFD", "GPT")


def sent_index(sent_id: str) -> int:
    # "0_NR" → 0
    return int(sent_id.split("_", 1)[0])


def word_has_gaze(row: dict[str, str]) -> bool:
    return any(float(row[name]) != 0.0 for name in WORD_GAZE)


def rebuild_sentence_nfix(word_rows: list[dict[str, str]]) -> dict[int, float]:
    buckets: dict[int, list[float]] = defaultdict(list)
    for row in word_rows:
        if not word_has_gaze(row):
            continue
        buckets[sent_index(row["Sent_ID"])].append(float(row["nFixations"]))
    return {sid: sum(values) / len(values) for sid, values in buckets.items() if values}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--atol", type=float, default=1e-6)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    _, words = read_dicts(ZUCO_WORD_SUBJECT_1)
    _, sentences = read_dicts(ZUCO_SUBJECT_1)
    gold = {int(row["id"]): float(row["nFixations"]) for row in sentences}
    rebuilt = rebuild_sentence_nfix(words)

    missing = sorted(set(gold) - set(rebuilt))
    extra = sorted(set(rebuilt) - set(gold))
    deltas = []
    for sid, gold_value in gold.items():
        if sid not in rebuilt:
            continue
        deltas.append(abs(rebuilt[sid] - gold_value))

    max_delta = max(deltas) if deltas else float("inf")
    mean_delta = sum(deltas) / len(deltas) if deltas else float("inf")
    n_close = sum(1 for item in deltas if item <= args.atol)

    if not args.quiet:
        print("Subject 1 word → sentence nFixations")
        print(f"  word rows: {len(words)}")
        print(f"  sentence rows: {len(sentences)}")
        print(f"  rebuilt sentences: {len(rebuilt)}")
        print(f"  compared: {len(deltas)}")
        print(f"  max |delta|: {max_delta:.6f}")
        print(f"  mean |delta|: {mean_delta:.6f}")
        print(f"  within atol {args.atol}: {n_close}/{len(deltas)}")
        if missing:
            print(f"  gold ids with no fixated words: {missing[:10]}")
        if extra:
            print(f"  rebuilt ids missing from gold: {extra[:10]}")

    problems = []
    if extra:
        problems.append(f"{len(extra)} rebuilt ids are not in 1_SR.csv")
    if len(deltas) < 390:
        problems.append(f"only compared {len(deltas)} sentences")
    if max_delta > args.atol:
        problems.append(f"max delta {max_delta} exceeds {args.atol}")

    if problems:
        print("WORD-TO-SENTENCE FAILED:")
        for item in problems:
            print(" -", item)
        return 1

    print("word-to-sentence OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
