"""Contracts taken from the committed CSVs on this clone.

Counts are locked by ``tests/test_schema.py``. If a table is re-exported,
those tests fail before the lab notes drift.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from .csvio import read_dicts, read_headerless
from .paths import repo_root

# Sentiment: 0 = negative, 1 = neutral, 2 = positive.
# Combined ZuCo ∩ SST (400 sentences) mix, measured on this checkout.
ZUCO_LABEL_COUNTS = {0: 123, 1: 137, 2: 140}
# Full SST-3 mix, measured with a headered combined_full_sst_et.csv.
SST_LABEL_COUNTS = {0: 4649, 1: 2241, 2: 4963}

ZUCO_GAZE = ("nFixations", "FFD", "GPT", "TRT", "GD")
SST_GAZE = ("nFix", "FFD", "GPT", "TRT", "GD")
SENTENCE_RAW = (
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

# Subject 3 (MATLAB task-1 subject index 2) is the short file.
SUBJECT_ROW_COUNTS = {i: 400 for i in range(1, 13)}
SUBJECT_ROW_COUNTS[3] = 299
WORD_ROW_COUNTS = {i: 7129 for i in range(1, 13)}
WORD_ROW_COUNTS[3] = 5293

# model_full_SST.py uses batch_size=256 on 1186 test rows.
FULL_SST_TEST_ROWS = 1186
FULL_SST_BATCH = 256
FULL_SST_LAST_BATCH = FULL_SST_TEST_ROWS % FULL_SST_BATCH  # 162


@dataclass(frozen=True)
class TableSpec:
    relpath: str
    rows: int
    required: tuple[str, ...]
    headerless: bool = False
    notes: str = ""


TABLES: tuple[TableSpec, ...] = (
    TableSpec(
        "ZuCo_SST_data/combined_sst_et_standard.csv",
        400,
        ("sentence_id", "sentence", "sentiment_label", "nFixations", "FFD", "GPT", "TRT", "GD"),
        notes="z-scored sentence gaze joined to ZuCo ∩ SST text",
    ),
    TableSpec(
        "ZuCo_SST_data/combined_sst_et_min_max.csv",
        400,
        ("sentence_id", "sentence", "sentiment_label", "nFixations"),
        notes="same join, min-max gaze",
    ),
    TableSpec(
        "ZuCo_SST_data/ssts_ZuCo.csv",
        400,
        ("sentence_id", "sentence", "sentiment_label"),
        notes="text + labels only",
    ),
    TableSpec("ZuCo_SST_data/train.csv", 320, ("sentence_id", "sentiment_label")),
    TableSpec("ZuCo_SST_data/valid.csv", 40, ("sentence_id", "sentiment_label")),
    TableSpec("ZuCo_SST_data/test.csv", 40, ("sentence_id", "sentiment_label")),
    TableSpec(
        "SST_data/combined_full_sst_et.csv",
        11853,
        ("sentence_id", "sentence", "sentiment_label", "nFix", "FFD", "GPT", "TRT", "GD"),
        notes="projected sentence gaze; nFix not nFixations",
    ),
    TableSpec("SST_data/train_full_sst.csv", 9482, ("sentence_id", "nFix")),
    TableSpec("SST_data/valid_full_sst.csv", 1185, ("sentence_id", "nFix")),
    TableSpec("SST_data/test_full_sst.csv", 1186, ("sentence_id", "nFix")),
    TableSpec(
        "SST_data/stts_all_sentence_level.csv",
        11853,
        (),
        headerless=True,
        notes="headerless SST dump; first column is text, second is POSITIVE/NEUTRAL/NEGATIVE",
    ),
    TableSpec("ZuCo_et_csv_data/average_data.csv", 400, SENTENCE_RAW),
    TableSpec("ZuCo_et_csv_data/min_max_scaled_average_data.csv", 400, SENTENCE_RAW),
    TableSpec("ZuCo_et_csv_data/standard_scaled_average_data.csv", 400, SENTENCE_RAW),
    TableSpec("ZuCo_et_csv_data/3_SR.csv", 299, ("id", "SentLen", "nFixations")),
    TableSpec("ZuCo_et_csv_data/1_SR.csv", 400, ("id", "SentLen", "nFixations")),
    TableSpec(
        "ZuCo_et_csv_data/word/word_averages_v2.csv",
        7129,
        ("id", "Sent_ID", "Word_ID", "Word", "nFixations", "WordLen"),
    ),
    TableSpec(
        "gaze_prediction/data/prediction_test.csv",
        1751,
        ("sentence_id", "word_id", "word", "nFix", "FFD", "GPT", "TRT", "GD"),
        notes="predicted word gaze for sentence_ids 300-399",
    ),
    TableSpec(
        "gaze_prediction/data/prediction_test_v2.csv",
        191971,
        ("sentence_id", "word_id", "word", "nFix"),
        notes="predicted word gaze for all 11853 SST sentences",
    ),
    TableSpec(
        "gaze_prediction/data/provo.csv",
        2659,
        ("sentence_id", "word_id", "word", "nFix", "fixProp"),
        notes="PROVO-format table (fixProp instead of GD)",
    ),
)


@dataclass
class CheckResult:
    ok: bool
    errors: list[str] = field(default_factory=list)


def _label_counts(rows: list[dict[str, str]], key: str = "sentiment_label") -> dict[int, int]:
    counts = {0: 0, 1: 0, 2: 0}
    for row in rows:
        counts[int(row[key])] += 1
    return counts


def check_tables(root: Path | None = None) -> CheckResult:
    root = root or repo_root()
    errors: list[str] = []
    for spec in TABLES:
        path = root / spec.relpath
        if not path.exists():
            errors.append(f"missing {spec.relpath}")
            continue
        if spec.headerless:
            rows = read_headerless(path)
            if len(rows) != spec.rows:
                errors.append(f"{spec.relpath}: {len(rows)} rows, expected {spec.rows}")
            continue
        fields, rows = read_dicts(path)
        if len(rows) != spec.rows:
            errors.append(f"{spec.relpath}: {len(rows)} rows, expected {spec.rows}")
        missing = [c for c in spec.required if c not in fields]
        if missing:
            errors.append(f"{spec.relpath}: missing columns {missing}")
    return CheckResult(ok=not errors, errors=errors)


def check_label_mix(root: Path | None = None) -> CheckResult:
    root = root or repo_root()
    errors: list[str] = []
    _, zuco = read_dicts(root / "ZuCo_SST_data/combined_sst_et_standard.csv")
    got = _label_counts(zuco)
    if got != ZUCO_LABEL_COUNTS:
        errors.append(f"ZuCo labels {got} != {ZUCO_LABEL_COUNTS}")
    _, sst = read_dicts(root / "SST_data/combined_full_sst_et.csv")
    got = _label_counts(sst)
    if got != SST_LABEL_COUNTS:
        errors.append(f"SST labels {got} != {SST_LABEL_COUNTS}")
    return CheckResult(ok=not errors, errors=errors)


def check_subject_lengths(root: Path | None = None) -> CheckResult:
    root = root or repo_root()
    errors: list[str] = []
    for i, expected in SUBJECT_ROW_COUNTS.items():
        _, rows = read_dicts(root / f"ZuCo_et_csv_data/{i}_SR.csv")
        if len(rows) != expected:
            errors.append(f"subject {i} sentence rows {len(rows)} != {expected}")
    for i, expected in WORD_ROW_COUNTS.items():
        _, rows = read_dicts(root / f"ZuCo_et_csv_data/word/{i}_SR.csv")
        if len(rows) != expected:
            errors.append(f"subject {i} word rows {len(rows)} != {expected}")
    return CheckResult(ok=not errors, errors=errors)


def inventory(root: Path | None = None) -> list[dict[str, object]]:
    """Return one dict per catalogued table (exists / rows / columns)."""
    root = root or repo_root()
    out: list[dict[str, object]] = []
    for spec in TABLES:
        path = root / spec.relpath
        item: dict[str, object] = {
            "path": spec.relpath,
            "expected_rows": spec.rows,
            "exists": path.exists(),
            "notes": spec.notes,
        }
        if path.exists():
            if spec.headerless:
                rows = read_headerless(path)
                item["rows"] = len(rows)
                item["columns"] = ["<headerless>", f"{len(rows[0]) if rows else 0} fields"]
            else:
                fields, rows = read_dicts(path)
                item["rows"] = len(rows)
                item["columns"] = fields
        out.append(item)
    return out
