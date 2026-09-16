"""Walk the checked-in consumer tables and report row counts + headers."""

from __future__ import annotations

from dataclasses import dataclass

from . import csvio, paths
from .schema import (
    PRED_WORD,
    PROVO_WORD,
    SST_SENTENCE_GAZE,
    SUBJECT_SENTENCE,
    SUBJECT_WORD,
    ZUCO_SENTENCE_GAZE,
    ZUCO_TEXT_ONLY,
    TableSchema,
    validate_rows,
)


@dataclass(frozen=True)
class FileInventory:
    key: str
    relpath: str
    rows: int
    columns: tuple[str, ...]
    schema_ok: bool
    note: str


def _rel(path) -> str:
    return str(path.relative_to(paths.ROOT))


def _one(key: str, path, schema: TableSchema | None, note: str) -> FileInventory:
    rows = csvio.read_dicts(path)
    ok = True
    if schema is not None:
        try:
            validate_rows(rows, schema)
        except Exception:
            ok = False
    return FileInventory(
        key=key,
        relpath=_rel(path),
        rows=len(rows),
        columns=tuple(rows[0].keys()) if rows else tuple(),
        schema_ok=ok,
        note=note,
    )


def inventory() -> list[FileInventory]:
    items = [
        _one("zuco_combined_standard", paths.ZUCO_COMBINED_STANDARD, ZUCO_SENTENCE_GAZE, "5-fold training table"),
        _one("zuco_combined_minmax", paths.ZUCO_COMBINED_MINMAX, ZUCO_SENTENCE_GAZE, "same 400 sentences, min-max gaze"),
        _one("zuco_sentences", paths.ZUCO_SENTENCES, ZUCO_TEXT_ONLY, "text + label only"),
        _one("zuco_train", paths.ZUCO_TRAIN, ZUCO_SENTENCE_GAZE, "320-row leftover split"),
        _one("zuco_valid", paths.ZUCO_VALID, ZUCO_SENTENCE_GAZE, "40-row leftover split"),
        _one("zuco_test", paths.ZUCO_TEST, ZUCO_SENTENCE_GAZE, "40-row leftover split"),
        _one("sst_combined", paths.SST_COMBINED, SST_SENTENCE_GAZE, "projected-gaze SST"),
        _one("sst_train", paths.SST_TRAIN, SST_SENTENCE_GAZE, "80%"),
        _one("sst_valid", paths.SST_VALID, SST_SENTENCE_GAZE, "10%"),
        _one("sst_test", paths.SST_TEST, SST_SENTENCE_GAZE, "10%"),
        _one("et_average", paths.ET_AVERAGE, None, "index-mean of 12 readers"),
        _one("et_average_standard", paths.ET_AVERAGE_STANDARD, None, "z-scored average_data"),
        _one("word_averages", paths.WORD_AVERAGES, SUBJECT_WORD, "word-level means"),
        _one("provo", paths.PROVO, PROVO_WORD, "PROVO, fixProp not GD"),
        _one("pred_test", paths.PRED_TEST, PRED_WORD, "predicted word gaze"),
    ]
    for subject in range(1, 13):
        items.append(
            _one(
                f"subject_{subject}",
                paths.subject_sentence_csv(subject),
                SUBJECT_SENTENCE,
                "299 rows and remapped ids" if subject == 3 else "400 raw-ish sentence rows",
            )
        )
    return items
