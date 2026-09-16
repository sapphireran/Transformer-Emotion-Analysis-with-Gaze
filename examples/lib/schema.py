"""Expected columns for the checked-in tables."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Sequence, Tuple

from .io_csv import Row, to_float, to_int


GAZE_ALIASES = {
    "nFix": "nFixations",
    "nFixations": "nFix",
}

SST5 = ("nFix", "GD", "TRT", "FFD", "GPT")
ZUCO5 = ("nFixations", "FFD", "GPT", "TRT", "GD")
ZUCO_FULL = (
    "omissionRate",
    "nFixations",
    "meanPupilSize",
    "GD",
    "TRT",
    "FFD",
    "SFD",
    "GPT",
)

LABEL_NAMES = {0: "negative", 1: "neutral", 2: "positive"}


@dataclass(frozen=True)
class TableSchema:
    required: Tuple[str, ...]
    numeric: Tuple[str, ...] = ()
    integer: Tuple[str, ...] = ()
    labels: bool = False


SCHEMAS: Dict[str, TableSchema] = {
    "zuco_text": TableSchema(
        required=("sentence_id", "sentence", "sentiment_label"),
        integer=("sentence_id", "sentiment_label"),
        labels=True,
    ),
    "zuco_standard": TableSchema(
        required=("sentence_id", "sentence", "sentiment_label") + ZUCO_FULL,
        numeric=ZUCO_FULL,
        integer=("sentence_id", "sentiment_label"),
        labels=True,
    ),
    "zuco_minmax": TableSchema(
        required=("sentence_id", "sentence", "sentiment_label") + ZUCO_FULL,
        numeric=ZUCO_FULL,
        integer=("sentence_id", "sentiment_label"),
        labels=True,
    ),
    "zuco_train": TableSchema(
        required=("sentence_id", "sentence", "sentiment_label") + ZUCO_FULL,
        numeric=ZUCO_FULL,
        integer=("sentence_id", "sentiment_label"),
        labels=True,
    ),
    "zuco_valid": TableSchema(
        required=("sentence_id", "sentence", "sentiment_label") + ZUCO_FULL,
        numeric=ZUCO_FULL,
        integer=("sentence_id", "sentiment_label"),
        labels=True,
    ),
    "zuco_test": TableSchema(
        required=("sentence_id", "sentence", "sentiment_label") + ZUCO_FULL,
        numeric=ZUCO_FULL,
        integer=("sentence_id", "sentiment_label"),
        labels=True,
    ),
    "zuco_subject_1": TableSchema(
        required=("id", "SentLen") + ZUCO_FULL,
        numeric=("SentLen",) + ZUCO_FULL,
        integer=("id",),
    ),
    "zuco_average": TableSchema(
        required=("id", "SentLen") + ZUCO_FULL,
        numeric=("SentLen",) + ZUCO_FULL,
        integer=("id",),
    ),
    "zuco_word_avg": TableSchema(
        required=(
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
        numeric=(
            "nFixations",
            "meanPupilSize",
            "GD",
            "TRT",
            "FFD",
            "SFD",
            "GPT",
            "WordLen",
        ),
        integer=("id", "Word_ID"),
    ),
    "sst_combined": TableSchema(
        required=("sentence_id", "sentence", "sentiment_label") + SST5,
        numeric=SST5,
        integer=("sentence_id", "sentiment_label"),
        labels=True,
    ),
    "sst_train": TableSchema(
        required=("sentence_id", "sentence", "sentiment_label") + SST5,
        numeric=SST5,
        integer=("sentence_id", "sentiment_label"),
        labels=True,
    ),
    "sst_valid": TableSchema(
        required=("sentence_id", "sentence", "sentiment_label") + SST5,
        numeric=SST5,
        integer=("sentence_id", "sentiment_label"),
        labels=True,
    ),
    "sst_test": TableSchema(
        required=("sentence_id", "sentence", "sentiment_label") + SST5,
        numeric=SST5,
        integer=("sentence_id", "sentiment_label"),
        labels=True,
    ),
    "pred_word_small": TableSchema(
        required=("sentence_id", "word_id", "word", "nFix", "FFD", "GPT", "TRT", "GD"),
        numeric=("nFix", "FFD", "GPT", "TRT", "GD"),
        integer=("sentence_id", "word_id"),
    ),
}


@dataclass
class SchemaIssue:
    level: str
    message: str


@dataclass
class SchemaReport:
    name: str
    ok: bool
    issues: List[SchemaIssue] = field(default_factory=list)

    def add(self, level: str, message: str) -> None:
        self.issues.append(SchemaIssue(level, message))
        if level == "error":
            self.ok = False


def validate_rows(name: str, header: Sequence[str], rows: Sequence[Row]) -> SchemaReport:
    report = SchemaReport(name=name, ok=True)
    schema = SCHEMAS[name]
    header_set = set(header)

    missing = [col for col in schema.required if col not in header_set]
    extra = [col for col in header if col not in schema.required]
    if missing:
        report.add("error", f"missing columns: {missing}")
    if extra:
        report.add("warning", f"extra columns: {extra}")
    if not rows:
        report.add("error", "table has a header but no data rows")
        return report

    sample_n = min(len(rows), 64)
    for index in range(sample_n):
        row = rows[index]
        for col in schema.numeric:
            if col not in row:
                continue
            try:
                to_float(row[col])
            except ValueError:
                report.add("error", f"row {index} {col!r} is not numeric: {row[col]!r}")
                break
        for col in schema.integer:
            if col not in row:
                continue
            try:
                to_int(row[col])
            except ValueError:
                report.add("error", f"row {index} {col!r} is not an integer: {row[col]!r}")
                break
        if schema.labels and "sentiment_label" in row:
            try:
                label = to_int(row["sentiment_label"])
            except ValueError:
                label = None
            if label not in LABEL_NAMES:
                report.add(
                    "error",
                    f"row {index} sentiment_label={row.get('sentiment_label')!r} "
                    "is not in {0,1,2}",
                )
                break
    return report
