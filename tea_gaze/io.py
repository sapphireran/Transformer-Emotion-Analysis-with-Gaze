"""Load the personal CSVs already checked into this repository."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Mapping, Sequence

from tea_gaze.features import canonicalize_gaze_columns
from tea_gaze.paths import DATASETS, DatasetRef, require_dataset, subject_et_paths
from tea_gaze.schema import (
    ValidationReport,
    validate_sentence_rows,
    validate_word_rows,
)


def _open_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def load_rows(key: str) -> list[dict[str, str]]:
    ref = require_dataset(key)
    return _open_csv(ref.path)


def load_and_validate(key: str) -> tuple[list[dict[str, str]], ValidationReport]:
    ref = require_dataset(key)
    rows = _open_csv(ref.path)
    if ref.kind == "word":
        report = validate_word_rows(key, rows)
    else:
        require_text = "sentence" in (rows[0] if rows else {})
        require_fusion = ref.gaze_source in {"measured", "predicted"} and (
            "nFixations" in (rows[0] if rows else {}) or "nFix" in (rows[0] if rows else {})
        )
        report = validate_sentence_rows(
            key,
            rows,
            require_fusion_features=require_fusion,
            require_text=require_text,
        )
    return rows, report


def load_frame(key: str):
    """Load a named dataset as a pandas DataFrame."""
    import pandas as pd

    ref = require_dataset(key)
    return pd.read_csv(ref.path)


def iter_subject_et() -> Iterator[tuple[int, list[dict[str, str]]]]:
    """Yield `(subject_index_1_based, rows)` for each ZuCo reader CSV."""
    for index, path in enumerate(subject_et_paths(), start=1):
        if not path.is_file():
            raise FileNotFoundError(path)
        yield index, _open_csv(path)


def load_subject_et_frame(subject: int):
    import pandas as pd

    path = subject_et_paths()[subject - 1]
    frame = pd.read_csv(path)
    frame["subject"] = subject
    return frame


def load_all_subject_et_frame():
    import pandas as pd

    frames = [load_subject_et_frame(subject) for subject in range(1, 13)]
    return pd.concat(frames, ignore_index=True)


@dataclass(frozen=True)
class DatasetInventoryItem:
    key: str
    path: str
    exists: bool
    kind: str
    gaze_source: str
    rows: int | None
    columns: tuple[str, ...]
    description: str


def inventory(keys: Sequence[str] | None = None) -> list[DatasetInventoryItem]:
    selected = [DATASETS[key] for key in keys] if keys else list(DATASETS.values())
    items: list[DatasetInventoryItem] = []
    for ref in selected:
        if not ref.exists():
            items.append(
                DatasetInventoryItem(
                    key=ref.key,
                    path=str(ref.relative_path),
                    exists=False,
                    kind=ref.kind,
                    gaze_source=ref.gaze_source,
                    rows=None,
                    columns=(),
                    description=ref.description,
                )
            )
            continue
        rows = _open_csv(ref.path)
        columns = tuple(rows[0].keys()) if rows else ()
        items.append(
            DatasetInventoryItem(
                key=ref.key,
                path=str(ref.relative_path),
                exists=True,
                kind=ref.kind,
                gaze_source=ref.gaze_source,
                rows=len(rows),
                columns=columns,
                description=ref.description,
            )
        )
    return items


def rename_gaze_columns(frame):
    """Return a copy whose gaze headers use canonical names."""
    mapping = canonicalize_gaze_columns(frame.columns)
    return frame.rename(columns=mapping)


def dataset_ref(key: str) -> DatasetRef:
    return require_dataset(key)


def peek_columns(path: Path) -> tuple[str, ...]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        header = next(reader)
    return tuple(header)


def row_count(path: Path) -> int:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        next(reader, None)
        return sum(1 for _ in reader)


def as_records(frame) -> list[Mapping[str, object]]:
    return frame.to_dict(orient="records")
