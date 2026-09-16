"""Shared helpers for the personal example scripts.

Stdlib only. Import from other files in this directory after adding
the repo root to sys.path, or run every script from the repository
root: ``python3 examples/<script>.py``.
"""

from __future__ import annotations

import csv
import math
import os
import statistics
from collections import Counter
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

# examples/ → repo root
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))

LABEL_NAME = {0: "negative", 1: "neutral", 2: "positive"}

# Columns the two training scripts actually feed the fusion layer.
ZUCO_GAZE_COLS = ("nFixations", "FFD", "GPT", "TRT", "GD")
SST_GAZE_COLS = ("nFix", "FFD", "GPT", "TRT", "GD")
ZUCO_SENTENCE_GAZE_ALL = (
    "SentLen",
    "omissionRate",
    "nFixations",
    "meanPupilSize",
    "GD",
    "TRT",
    "FFD",
    "SFD",
    "GPT",
)
ZUCO_JOINED_GAZE_ALL = (
    "omissionRate",
    "nFixations",
    "meanPupilSize",
    "GD",
    "TRT",
    "FFD",
    "SFD",
    "GPT",
)

# Inventory used by inspect_datasets.py and schema_validate.py.
# kind is a coarse tag for reports, not a formal schema language.
DATASETS: Tuple[Dict[str, object], ...] = (
    {
        "key": "zuco_text",
        "path": "ZuCo_SST_data/ssts_ZuCo.csv",
        "kind": "zuco-text",
        "expected_rows": 400,
        "required": ("sentence_id", "sentence", "sentiment_label"),
    },
    {
        "key": "zuco_combined_standard",
        "path": "ZuCo_SST_data/combined_sst_et_standard.csv",
        "kind": "zuco-joined",
        "expected_rows": 400,
        "required": ("sentence_id", "sentence", "sentiment_label") + ZUCO_JOINED_GAZE_ALL,
    },
    {
        "key": "zuco_combined_minmax",
        "path": "ZuCo_SST_data/combined_sst_et_min_max.csv",
        "kind": "zuco-joined",
        "expected_rows": 400,
        "required": ("sentence_id", "sentence", "sentiment_label") + ZUCO_JOINED_GAZE_ALL,
    },
    {
        "key": "zuco_train",
        "path": "ZuCo_SST_data/train.csv",
        "kind": "zuco-joined",
        "expected_rows": 320,
        "required": ("sentence_id", "sentence", "sentiment_label") + ZUCO_JOINED_GAZE_ALL,
    },
    {
        "key": "zuco_valid",
        "path": "ZuCo_SST_data/valid.csv",
        "kind": "zuco-joined",
        "expected_rows": 40,
        "required": ("sentence_id", "sentence", "sentiment_label") + ZUCO_JOINED_GAZE_ALL,
    },
    {
        "key": "zuco_test",
        "path": "ZuCo_SST_data/test.csv",
        "kind": "zuco-joined",
        "expected_rows": 40,
        "required": ("sentence_id", "sentence", "sentiment_label") + ZUCO_JOINED_GAZE_ALL,
    },
    {
        "key": "zuco_average_raw",
        "path": "ZuCo_et_csv_data/average_data.csv",
        "kind": "zuco-sentence-gaze",
        "expected_rows": 400,
        "required": ("id",) + ZUCO_SENTENCE_GAZE_ALL,
    },
    {
        "key": "zuco_average_standard",
        "path": "ZuCo_et_csv_data/standard_scaled_average_data.csv",
        "kind": "zuco-sentence-gaze",
        "expected_rows": 400,
        "required": ("id",) + ZUCO_SENTENCE_GAZE_ALL,
    },
    {
        "key": "zuco_average_minmax",
        "path": "ZuCo_et_csv_data/min_max_scaled_average_data.csv",
        "kind": "zuco-sentence-gaze",
        "expected_rows": 400,
        "required": ("id",) + ZUCO_SENTENCE_GAZE_ALL,
    },
    {
        "key": "sst_combined",
        "path": "SST_data/combined_full_sst_et.csv",
        "kind": "sst-joined",
        "expected_rows": 11853,
        "required": ("sentence_id", "sentence", "sentiment_label") + SST_GAZE_COLS,
    },
    {
        "key": "sst_train",
        "path": "SST_data/train_full_sst.csv",
        "kind": "sst-joined",
        "expected_rows": 9482,
        "required": ("sentence_id", "sentence", "sentiment_label") + SST_GAZE_COLS,
    },
    {
        "key": "sst_valid",
        "path": "SST_data/valid_full_sst.csv",
        "kind": "sst-joined",
        "expected_rows": 1185,
        "required": ("sentence_id", "sentence", "sentiment_label") + SST_GAZE_COLS,
    },
    {
        "key": "sst_test",
        "path": "SST_data/test_full_sst.csv",
        "kind": "sst-joined",
        "expected_rows": 1186,
        "required": ("sentence_id", "sentence", "sentiment_label") + SST_GAZE_COLS,
    },
    {
        "key": "sst_word_placeholder",
        "path": "SST_data/sst_et_test.csv",
        "kind": "sst-word",
        "expected_rows": 191971,
        "required": ("sentence_id", "word_id", "word") + SST_GAZE_COLS,
    },
    {
        "key": "word_averages_v2",
        "path": "ZuCo_et_csv_data/word/word_averages_v2.csv",
        "kind": "zuco-word",
        "expected_rows": 7129,
        "required": (
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
        ),
    },
    {
        "key": "provo",
        "path": "gaze_prediction/data/provo.csv",
        "kind": "provo-word",
        "expected_rows": 2659,
        "required": ("sentence_id", "word_id", "word", "nFix", "FFD", "GPT", "TRT", "fixProp"),
    },
    {
        "key": "prediction_test",
        "path": "gaze_prediction/data/prediction_test.csv",
        "kind": "sst-word",
        "expected_rows": 1751,
        "required": ("sentence_id", "word_id", "word") + SST_GAZE_COLS,
    },
    {
        "key": "prediction_test_v2",
        "path": "gaze_prediction/data/prediction_test_v2.csv",
        "kind": "sst-word",
        "expected_rows": 191971,
        "required": ("sentence_id", "word_id", "word") + SST_GAZE_COLS,
    },
)

SUBJECT_SENTENCE_FILES = tuple(
    f"ZuCo_et_csv_data/{i}_SR.csv" for i in range(1, 13)
)
SUBJECT_WORD_FILES = tuple(
    f"ZuCo_et_csv_data/word/{i}_SR.csv" for i in range(1, 13)
)


def abs_path(rel: str) -> str:
    return os.path.join(REPO_ROOT, rel)


def dataset_by_key(key: str) -> Dict[str, object]:
    for spec in DATASETS:
        if spec["key"] == key:
            return spec
    raise KeyError(f"unknown dataset key: {key}")


def read_rows(rel: str) -> Tuple[List[str], List[Dict[str, str]]]:
    path = abs_path(rel)
    with open(path, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"{rel} has no header")
        cols = list(reader.fieldnames)
        rows = list(reader)
    return cols, rows


def count_rows(rel: str) -> Tuple[List[str], int]:
    path = abs_path(rel)
    with open(path, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        cols = list(reader.fieldnames or [])
        n = sum(1 for _ in reader)
    return cols, n


def as_int(value: str) -> int:
    return int(float(value))


def as_float(value: str) -> float:
    return float(value)


def is_finite_number(value: str) -> bool:
    if value is None or value == "":
        return False
    try:
        return math.isfinite(float(value))
    except ValueError:
        return False


def label_counts(rows: Iterable[Dict[str, str]], column: str = "sentiment_label") -> Counter:
    counts: Counter = Counter()
    for row in rows:
        counts[as_int(row[column])] += 1
    return counts


def format_label_counts(counts: Counter) -> str:
    parts = []
    total = sum(counts.values()) or 1
    for lab in (0, 1, 2):
        n = counts.get(lab, 0)
        parts.append(f"{LABEL_NAME[lab]}={n} ({100.0 * n / total:.1f}%)")
    return ", ".join(parts)


def column_stats(rows: Sequence[Dict[str, str]], columns: Sequence[str]) -> List[Dict[str, object]]:
    out: List[Dict[str, object]] = []
    for col in columns:
        vals: List[float] = []
        zeros = 0
        missing = 0
        for row in rows:
            raw = row.get(col, "")
            if raw is None or raw == "":
                missing += 1
                continue
            try:
                x = float(raw)
            except ValueError:
                missing += 1
                continue
            vals.append(x)
            if x == 0.0:
                zeros += 1
        n = len(vals)
        out.append(
            {
                "column": col,
                "n": n,
                "missing": missing,
                "zeros": zeros,
                "zero_rate": (zeros / n) if n else float("nan"),
                "min": min(vals) if vals else float("nan"),
                "max": max(vals) if vals else float("nan"),
                "mean": statistics.fmean(vals) if vals else float("nan"),
                "stdev": statistics.pstdev(vals) if n > 1 else float("nan"),
            }
        )
    return out


def fmt_stat_table(stats: Sequence[Dict[str, object]]) -> str:
    header = f"{'column':<16}{'n':>8}{'miss':>8}{'zeros':>8}{'min':>12}{'mean':>12}{'max':>12}{'std':>12}"
    lines = [header, "-" * len(header)]
    for row in stats:
        lines.append(
            f"{row['column']:<16}{row['n']:>8}{row['missing']:>8}{row['zeros']:>8}"
            f"{row['min']:>12.4f}{row['mean']:>12.4f}{row['max']:>12.4f}{row['stdev']:>12.4f}"
        )
    return "\n".join(lines)


def truncate(text: str, width: int = 72) -> str:
    text = " ".join(text.split())
    if len(text) <= width:
        return text
    return text[: width - 1] + "…"


def ensure_cwd_message() -> Optional[str]:
    """Return a warning if the obvious data folders are missing from CWD."""
    if os.path.isdir(abs_path("ZuCo_SST_data")):
        return None
    return (
        "Could not find ZuCo_SST_data/ next to examples/. "
        "Run the scripts from the repository root or keep this file in examples/."
    )
