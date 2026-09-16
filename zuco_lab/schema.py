"""Column contracts for the consumer CSVs.

The training scripts pick columns by name. A renamed ``nFixations`` / ``nFix``
field is enough to break a run, so examples validate headers before they
compute anything.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence


class SchemaError(ValueError):
    pass


@dataclass(frozen=True)
class TableSchema:
    name: str
    required: tuple[str, ...]
    numeric: tuple[str, ...] = ()
    label_column: str | None = None

    def missing(self, header: Sequence[str]) -> list[str]:
        have = set(header)
        return [name for name in self.required if name not in have]


ZUCO_SENTENCE_GAZE = TableSchema(
    name="zuco_sentence_gaze",
    required=(
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
    ),
    numeric=(
        "sentence_id",
        "sentiment_label",
        "omissionRate",
        "nFixations",
        "meanPupilSize",
        "GD",
        "TRT",
        "FFD",
        "SFD",
        "GPT",
    ),
    label_column="sentiment_label",
)

ZUCO_TEXT_ONLY = TableSchema(
    name="zuco_text_only",
    required=("sentence_id", "sentence", "sentiment_label"),
    numeric=("sentence_id", "sentiment_label"),
    label_column="sentiment_label",
)

SST_SENTENCE_GAZE = TableSchema(
    name="sst_sentence_gaze",
    required=("sentence_id", "sentence", "sentiment_label", "nFix", "GD", "TRT", "FFD", "GPT"),
    numeric=("sentence_id", "sentiment_label", "nFix", "GD", "TRT", "FFD", "GPT"),
    label_column="sentiment_label",
)

SUBJECT_SENTENCE = TableSchema(
    name="subject_sentence",
    required=(
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
    ),
    numeric=(
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
    ),
)

SUBJECT_WORD = TableSchema(
    name="subject_word",
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
    numeric=("id", "Word_ID", "nFixations", "meanPupilSize", "GD", "TRT", "FFD", "SFD", "GPT", "WordLen"),
)

PROVO_WORD = TableSchema(
    name="provo_word",
    required=("sentence_id", "word_id", "word", "nFix", "FFD", "GPT", "TRT", "fixProp"),
    numeric=("sentence_id", "word_id", "nFix", "FFD", "GPT", "TRT", "fixProp"),
)

PRED_WORD = TableSchema(
    name="pred_word",
    required=("sentence_id", "word_id", "word", "nFix", "FFD", "GPT", "TRT", "GD"),
    numeric=("sentence_id", "word_id", "nFix", "FFD", "GPT", "TRT", "GD"),
)

SCHEMAS = {
    schema.name: schema
    for schema in (
        ZUCO_SENTENCE_GAZE,
        ZUCO_TEXT_ONLY,
        SST_SENTENCE_GAZE,
        SUBJECT_SENTENCE,
        SUBJECT_WORD,
        PROVO_WORD,
        PRED_WORD,
    )
}


def validate_rows(
    rows: Sequence[Mapping[str, str]],
    schema: TableSchema,
    *,
    sample: int = 8,
) -> None:
    if not rows:
        raise SchemaError(f"{schema.name}: table is empty")
    missing = schema.missing(list(rows[0].keys()))
    if missing:
        raise SchemaError(f"{schema.name}: missing columns {missing}")
    if schema.label_column:
        allowed = {"0", "1", "2"}
        for row in rows:
            if row[schema.label_column] not in allowed:
                raise SchemaError(
                    f"{schema.name}: unexpected label {row[schema.label_column]!r}"
                )
    for row in rows[:sample]:
        for name in schema.numeric:
            try:
                float(row[name])
            except (TypeError, ValueError) as exc:
                raise SchemaError(f"{schema.name}: {name}={row[name]!r} is not numeric") from exc


def header_of(rows: Sequence[Mapping[str, str]]) -> list[str]:
    if not rows:
        return []
    return list(rows[0].keys())


def unique_values(rows: Iterable[Mapping[str, str]], column: str) -> set[str]:
    return {row[column] for row in rows}
