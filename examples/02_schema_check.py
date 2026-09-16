#!/usr/bin/env python3
"""Fail if committed CSVs drift from the contracts the docs describe.

This is the cheap regression test for the personal data tables: headers,
approximate row counts, label alphabets, and "can these gaze cells parse
as floats?"  It does not checksum values, so a silent rescaling still
passes.

    python3 examples/02_schema_check.py
    echo $?   # 0 = ok
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import read_csv, repo_root, write_text

# (relative path, expected header or None, min_rows, max_rows, label_col or None)
CONTRACTS = [
    (
        "ZuCo_SST_data/ssts_ZuCo.csv",
        ["sentence_id", "sentence", "sentiment_label"],
        400,
        400,
        "sentiment_label",
    ),
    (
        "ZuCo_SST_data/combined_sst_et_standard.csv",
        [
            "sentence_id",
            "sentence",
            "sentiment_label",
            "omissionRate",
            "nFixations",
            "meanPupilSize",
            "GD",
            "TRT",
            "FFD",
            "SFD",
            "GPT",
        ],
        400,
        400,
        "sentiment_label",
    ),
    (
        "ZuCo_SST_data/combined_sst_et_min_max.csv",
        [
            "sentence_id",
            "sentence",
            "sentiment_label",
            "omissionRate",
            "nFixations",
            "meanPupilSize",
            "GD",
            "TRT",
            "FFD",
            "SFD",
            "GPT",
        ],
        400,
        400,
        "sentiment_label",
    ),
    (
        "ZuCo_SST_data/train.csv",
        None,  # same header as combined; checked separately
        320,
        320,
        "sentiment_label",
    ),
    (
        "ZuCo_SST_data/valid.csv",
        None,
        40,
        40,
        "sentiment_label",
    ),
    (
        "ZuCo_SST_data/test.csv",
        None,
        40,
        40,
        "sentiment_label",
    ),
    (
        "SST_data/combined_full_sst_et.csv",
        ["sentence_id", "sentence", "sentiment_label", "nFix", "GD", "TRT", "FFD", "GPT"],
        11853,
        11853,
        "sentiment_label",
    ),
    (
        "SST_data/train_full_sst.csv",
        ["sentence_id", "sentence", "sentiment_label", "nFix", "GD", "TRT", "FFD", "GPT"],
        9482,
        9482,
        "sentiment_label",
    ),
    (
        "SST_data/valid_full_sst.csv",
        ["sentence_id", "sentence", "sentiment_label", "nFix", "GD", "TRT", "FFD", "GPT"],
        1185,
        1185,
        "sentiment_label",
    ),
    (
        "SST_data/test_full_sst.csv",
        ["sentence_id", "sentence", "sentiment_label", "nFix", "GD", "TRT", "FFD", "GPT"],
        1186,
        1186,
        "sentiment_label",
    ),
    (
        "ZuCo_et_csv_data/1_SR.csv",
        [
            "id",
            "SentLen",
            "omissionRate",
            "nFixations",
            "meanPupilSize",
            "GD",
            "TRT",
            "FFD",
            "SFD",
            "GPT",
        ],
        400,
        400,
        None,
    ),
    (
        "ZuCo_et_csv_data/3_SR.csv",
        [
            "id",
            "SentLen",
            "omissionRate",
            "nFixations",
            "meanPupilSize",
            "GD",
            "TRT",
            "FFD",
            "SFD",
            "GPT",
        ],
        299,
        299,
        None,
    ),
    (
        "ZuCo_et_csv_data/word/word_averages_v2.csv",
        [
            "id",
            "Sent_ID",
            "Word_ID",
            "Word",
            "nFixations",
            "meanPupilSize",
            "GD",
            "TRT",
            "FFD",
            "SFD",
            "GPT",
            "WordLen",
        ],
        7129,
        7129,
        None,
    ),
    (
        "gaze_prediction/data/prediction_test.csv",
        ["sentence_id", "word_id", "word", "nFix", "FFD", "GPT", "TRT", "GD"],
        1751,
        1751,
        None,
    ),
    (
        "gaze_prediction/data/provo.csv",
        ["sentence_id", "word_id", "word", "nFix", "FFD", "GPT", "TRT", "fixProp"],
        2659,
        2659,
        None,
    ),
]

FUSION_FLOAT_COLS = {
    "ZuCo_SST_data/combined_sst_et_standard.csv": [
        "nFixations",
        "FFD",
        "GPT",
        "TRT",
        "GD",
        "omissionRate",
        "meanPupilSize",
        "SFD",
    ],
    "SST_data/train_full_sst.csv": ["nFix", "FFD", "GPT", "TRT", "GD"],
    "SST_data/valid_full_sst.csv": ["nFix", "FFD", "GPT", "TRT", "GD"],
    "SST_data/test_full_sst.csv": ["nFix", "FFD", "GPT", "TRT", "GD"],
}

COMBINED_HEADER = [
    "sentence_id",
    "sentence",
    "sentiment_label",
    "omissionRate",
    "nFixations",
    "meanPupilSize",
    "GD",
    "TRT",
    "FFD",
    "SFD",
    "GPT",
]


def check_floats(header: list[str], rows: list[list[str]], names: list[str], relative: str) -> list[str]:
    errors = []
    for name in names:
        if name not in header:
            errors.append(f"{relative}: missing float column {name}")
            continue
        idx = header.index(name)
        bad = 0
        for i, row in enumerate(rows, start=2):
            try:
                float(row[idx])
            except (ValueError, IndexError):
                bad += 1
                if bad <= 3:
                    errors.append(f"{relative}:{i} column {name} is not a float: {row[idx]!r}")
        if bad > 3:
            errors.append(f"{relative}: {bad} additional non-float values in {name}")
    return errors


def check_labels(header: list[str], rows: list[list[str]], col: str, relative: str) -> list[str]:
    idx = header.index(col)
    errors = []
    seen = set()
    for i, row in enumerate(rows, start=2):
        raw = row[idx]
        try:
            value = int(float(raw))
        except ValueError:
            errors.append(f"{relative}:{i} label {raw!r} is not an int")
            continue
        if value != float(raw) and "." in raw:
            # allow "0.0" but not "0.5"
            if abs(float(raw) - value) > 1e-9:
                errors.append(f"{relative}:{i} label {raw!r} is not an integer value")
        if value not in {0, 1, 2}:
            errors.append(f"{relative}:{i} label {value} not in {{0,1,2}}")
        seen.add(value)
    if seen != {0, 1, 2}:
        errors.append(f"{relative}: expected labels {{0,1,2}}, saw {sorted(seen)}")
    return errors


def main() -> int:
    errors: list[str] = []
    ok: list[str] = []
    root = repo_root()

    headerless = root / "SST_data" / "stts_all_sentence_level.csv"
    if headerless.exists():
        _, rows = read_csv("SST_data/stts_all_sentence_level.csv", has_header=False)
        if len(rows) != 11853:
            errors.append(
                f"SST_data/stts_all_sentence_level.csv: expected 11853 data rows, got {len(rows)}"
            )
        else:
            ok.append("SST_data/stts_all_sentence_level.csv: 11853 headerless rows")
        # first column looks like a sentence, last like a string label
        mapping = {"NEGATIVE", "NEUTRAL", "POSITIVE"}
        bad_lab = sum(1 for row in rows if row[-1].strip().strip('"') not in mapping)
        if bad_lab:
            errors.append(f"stts_all_sentence_level.csv: {bad_lab} rows with unknown string labels")
    else:
        errors.append("missing SST_data/stts_all_sentence_level.csv")

    for relative, expected_header, min_rows, max_rows, label_col in CONTRACTS:
        path = root / relative
        if not path.exists():
            errors.append(f"missing {relative}")
            continue
        header, rows = read_csv(relative, has_header=True)
        if expected_header is None:
            expected_header = COMBINED_HEADER
        if header != expected_header:
            errors.append(f"{relative}: header mismatch\n  got {header}\n  exp {expected_header}")
        if not (min_rows <= len(rows) <= max_rows):
            errors.append(f"{relative}: row count {len(rows)} not in [{min_rows}, {max_rows}]")
        else:
            ok.append(f"{relative}: {len(rows)} rows, header ok")
        if label_col:
            errors.extend(check_labels(header, rows, label_col, relative))
        if relative in FUSION_FLOAT_COLS:
            errors.extend(check_floats(header, rows, FUSION_FLOAT_COLS[relative], relative))

    # Subject files 1-12 exist; 3 is the truncated one (already checked).
    for i in range(1, 13):
        rel = f"ZuCo_et_csv_data/{i}_SR.csv"
        if not (root / rel).exists():
            errors.append(f"missing {rel}")

    lines = ["# Schema check", ""]
    lines.extend(f"OK  {line}" for line in ok)
    lines.append("")
    if errors:
        lines.append(f"FAILURES ({len(errors)})")
        lines.extend(f"ERR {line}" for line in errors)
        code = 1
    else:
        lines.append("ALL CONTRACTS PASSED")
        code = 0

    text = "\n".join(lines)
    print(text)
    write_text("02_schema_check.txt", text)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
