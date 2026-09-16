#!/usr/bin/env python3
"""Walk one ZuCo sentence at word level and comment on gaze peaks.

Default sentence is id 0 — the mixed review used throughout docs/06.
Other useful ids from that page: 4 (max GPT; word table has the
`emp11111ty` IA label), 80 (max nFix; `murderoncampus` after hyphen
stripping), 135 (three-word negative).

    python examples/walk_sentence_gaze.py
    python examples/walk_sentence_gaze.py --sentence-id 4
    python examples/walk_sentence_gaze.py --sentence-id 80 --top 5

Word-level numbers come from
`ZuCo_et_csv_data/word/word_averages_v2.csv` (12-reader mean, raw-ish
units). Sentence-level raw and z-scored rows are printed first so the
late-fusion vector and the word table can be compared on one screen.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys


WORD_PATH = "ZuCo_et_csv_data/word/word_averages_v2.csv"
RAW_SENT_PATH = "ZuCo_et_csv_data/average_data.csv"
Z_SENT_PATH = "ZuCo_SST_data/combined_sst_et_standard.csv"
LABELS = {"0": "NEGATIVE", "1": "NEUTRAL", "2": "POSITIVE"}


def _load_csv(path: str) -> list[dict[str, str]]:
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _f(row: dict[str, str], key: str) -> float:
    return float(row[key])


def _find_sentence(rows: list[dict[str, str]], sentence_id: int) -> dict[str, str]:
    sid = str(sentence_id)
    for row in rows:
        if row.get("sentence_id") == sid or row.get("id") == sid:
            return row
    raise KeyError(f"sentence_id {sentence_id} not in table")


def _word_rows(rows: list[dict[str, str]], sentence_id: int) -> list[dict[str, str]]:
    """ZuCo word files tag task-1 sentences as '{id}_NR'."""
    wanted = {f"{sentence_id}_NR", f"{sentence_id}_TSR", str(sentence_id)}
    matched = [row for row in rows if row["Sent_ID"] in wanted]
    if not matched:
        # Some dumps store the bare id in Sent_ID already.
        matched = [row for row in rows if row["Sent_ID"].split("_")[0] == str(sentence_id)]
    matched.sort(key=lambda row: int(float(row["Word_ID"])))
    return matched


def _comment(words: list[dict[str, str]]) -> list[str]:
    """A few deterministic observations so the script is not just a table dump."""
    if not words:
        return ["No word-level rows matched this sentence_id."]

    notes = []
    by_nfix = max(words, key=lambda r: _f(r, "nFixations"))
    by_trt = max(words, key=lambda r: _f(r, "TRT"))
    by_gpt = max(words, key=lambda r: _f(r, "GPT"))
    skipped = [r for r in words if _f(r, "nFixations") < 0.25]

    notes.append(
        f"Highest nFixations: '{by_nfix['Word']}' "
        f"({_f(by_nfix, 'nFixations'):.2f} looks). "
        "Content words that flip or carry sentiment often win this."
    )
    notes.append(
        f"Highest TRT: '{by_trt['Word']}' "
        f"({_f(by_trt, 'TRT'):.1f} ms summed across visits)."
    )
    notes.append(
        f"Highest GPT: '{by_gpt['Word']}' "
        f"({_f(by_gpt, 'GPT'):.1f} ms until the eyes moved past). "
        "If this is the last word, treat it as wrap-up, not as a 1s lexical look."
    )
    if skipped:
        names = ", ".join(f"{r['Word']}" for r in skipped[:8])
        extra = "" if len(skipped) <= 8 else f" (+{len(skipped) - 8} more)"
        notes.append(
            f"Near-skipped (nFix < 0.25, most readers never landed): {names}{extra}."
        )

    last = words[-1]
    if by_gpt["Word"] == last["Word"] and _f(last, "GPT") > 2 * max(_f(last, "TRT"), 1.0):
        notes.append(
            f"GPT on final token '{last['Word']}' is much larger than TRT "
            f"({_f(last, 'GPT'):.0f} vs {_f(last, 'TRT'):.0f}). "
            "Classic sentence-final wrap-up: the go-past clock never gets a "
            "rightward saccade off the last region."
        )
    return notes


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sentence-id", type=int, default=0)
    parser.add_argument(
        "--top",
        type=int,
        default=3,
        help="Also list the top-N words by TRT (default 3).",
    )
    args = parser.parse_args()

    for path in (WORD_PATH, RAW_SENT_PATH, Z_SENT_PATH):
        if not os.path.isfile(path):
            print(f"Missing {path}. Run from the repository root.", file=sys.stderr)
            return 2

    raw = _find_sentence(_load_csv(RAW_SENT_PATH), args.sentence_id)
    zrow = _find_sentence(_load_csv(Z_SENT_PATH), args.sentence_id)
    words = _word_rows(_load_csv(WORD_PATH), args.sentence_id)

    label = zrow["sentiment_label"]
    print(f"sentence_id {args.sentence_id}   label {label} ({LABELS.get(label, '?')})")
    print(zrow["sentence"])
    print()
    print("Sentence-level RAW (12-reader mean)")
    print(
        f"  SentLen={raw['SentLen']}  omissionRate={float(raw['omissionRate']):.3f}  "
        f"nFix={float(raw['nFixations']):.3f}  "
        f"FFD={float(raw['FFD']):.1f}  GD={float(raw['GD']):.1f}  "
        f"TRT={float(raw['TRT']):.1f}  GPT={float(raw['GPT']):.1f}  "
        f"SFD={float(raw['SFD']):.1f}  pupil={float(raw['meanPupilSize']):.1f}"
    )
    print("Sentence-level Z (what EyeTrackingModel consumes)")
    print(
        f"  nFixations={float(zrow['nFixations']):+.3f}  "
        f"FFD={float(zrow['FFD']):+.3f}  "
        f"GPT={float(zrow['GPT']):+.3f}  "
        f"TRT={float(zrow['TRT']):+.3f}  "
        f"GD={float(zrow['GD']):+.3f}"
    )
    print()
    print(f"{'wid':>4}  {'word':<16} {'nFix':>6} {'FFD':>7} {'GD':>7} {'TRT':>7} {'GPT':>8} {'len':>4}")
    for row in words:
        print(
            f"{int(float(row['Word_ID'])):4d}  "
            f"{row['Word']:<16} "
            f"{_f(row, 'nFixations'):6.2f} "
            f"{_f(row, 'FFD'):7.1f} "
            f"{_f(row, 'GD'):7.1f} "
            f"{_f(row, 'TRT'):7.1f} "
            f"{_f(row, 'GPT'):8.1f} "
            f"{int(float(row['WordLen'])):4d}"
        )

    print()
    print("Commentary")
    for note in _comment(words):
        print(f"  - {note}")

    ranked = sorted(words, key=lambda r: _f(r, "TRT"), reverse=True)
    print()
    print(f"Top {args.top} TRT words")
    for row in ranked[: args.top]:
        print(f"  {row['Word']:<16} TRT={_f(row, 'TRT'):.1f} ms  nFix={_f(row, 'nFixations'):.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
