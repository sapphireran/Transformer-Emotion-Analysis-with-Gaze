"""Train / valid / test integrity checks."""

from __future__ import annotations

from dataclasses import dataclass

from . import csvio, labels, paths
from .schema import SST_SENTENCE_GAZE, ZUCO_SENTENCE_GAZE, TableSchema, validate_rows


@dataclass(frozen=True)
class SplitAudit:
    name: str
    n_train: int
    n_valid: int
    n_test: int
    n_combined: int
    train_valid_overlap: int
    train_test_overlap: int
    valid_test_overlap: int
    union_vs_combined: int
    missing_from_splits: int
    extra_in_splits: int
    train_labels: dict[int, int]
    valid_labels: dict[int, int]
    test_labels: dict[int, int]
    combined_labels: dict[int, int]


def _ids(rows: list[dict[str, str]], column: str = "sentence_id") -> set[str]:
    return {row[column] for row in rows}


def audit_split(
    name: str,
    combined_path,
    train_path,
    valid_path,
    test_path,
    schema: TableSchema,
) -> SplitAudit:
    combined = csvio.read_dicts(combined_path)
    train = csvio.read_dicts(train_path)
    valid = csvio.read_dicts(valid_path)
    test = csvio.read_dicts(test_path)
    for rows in (combined, train, valid, test):
        validate_rows(rows, schema)

    c_ids, tr, va, te = _ids(combined), _ids(train), _ids(valid), _ids(test)
    union = tr | va | te
    return SplitAudit(
        name=name,
        n_train=len(train),
        n_valid=len(valid),
        n_test=len(test),
        n_combined=len(combined),
        train_valid_overlap=len(tr & va),
        train_test_overlap=len(tr & te),
        valid_test_overlap=len(va & te),
        union_vs_combined=len(union) - len(c_ids),
        missing_from_splits=len(c_ids - union),
        extra_in_splits=len(union - c_ids),
        train_labels=labels.label_counts(train),
        valid_labels=labels.label_counts(valid),
        test_labels=labels.label_counts(test),
        combined_labels=labels.label_counts(combined),
    )


def audit_zuco() -> SplitAudit:
    return audit_split(
        "zuco_sst",
        paths.ZUCO_COMBINED_STANDARD,
        paths.ZUCO_TRAIN,
        paths.ZUCO_VALID,
        paths.ZUCO_TEST,
        ZUCO_SENTENCE_GAZE,
    )


def audit_sst() -> SplitAudit:
    return audit_split(
        "full_sst",
        paths.SST_COMBINED,
        paths.SST_TRAIN,
        paths.SST_VALID,
        paths.SST_TEST,
        SST_SENTENCE_GAZE,
    )


def normalize_text(text: str) -> str:
    cleaned = text.lower().replace("\\", "").replace("/", " ")
    return " ".join(cleaned.split())


def text_overlap() -> list[tuple[str, int, int]]:
    """Normalized sentence strings that appear in both tracks."""
    zuco = csvio.read_dicts(paths.ZUCO_COMBINED_STANDARD)
    sst = csvio.read_dicts(paths.SST_COMBINED)
    z_map = {normalize_text(row["sentence"]): int(row["sentiment_label"]) for row in zuco}
    s_map = {normalize_text(row["sentence"]): int(row["sentiment_label"]) for row in sst}
    shared = sorted(set(z_map) & set(s_map))
    return [(text, z_map[text], s_map[text]) for text in shared]
