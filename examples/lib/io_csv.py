"""Tiny CSV helpers. Intentionally stdlib-only."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence


Row = Dict[str, str]


def read_rows(path: Path) -> List[Row]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def read_header(path: Path) -> List[str]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        header = next(reader, None)
    if header is None:
        raise ValueError(f"{path} is empty")
    return header


def to_float(value: str) -> float:
    text = (value or "").strip()
    if text == "" or text.lower() in {"nan", "none", "null"}:
        raise ValueError("missing numeric value")
    return float(text)


def to_int(value: str) -> int:
    return int(float((value or "").strip()))


def column(rows: Sequence[Row], name: str) -> List[str]:
    return [row[name] for row in rows]


def float_column(rows: Sequence[Row], name: str) -> List[float]:
    out: List[float] = []
    for index, row in enumerate(rows):
        try:
            out.append(to_float(row[name]))
        except (KeyError, ValueError) as exc:
            raise ValueError(f"row {index} column {name!r}: {exc}") from exc
    return out


def optional_float_column(rows: Sequence[Row], name: str) -> List[Optional[float]]:
    out: List[Optional[float]] = []
    for row in rows:
        try:
            out.append(to_float(row[name]))
        except (KeyError, ValueError):
            out.append(None)
    return out


def select_columns(rows: Sequence[Row], names: Iterable[str]) -> List[List[float]]:
    name_list = list(names)
    matrix: List[List[float]] = []
    for index, row in enumerate(rows):
        values: List[float] = []
        for name in name_list:
            try:
                values.append(to_float(row[name]))
            except (KeyError, ValueError) as exc:
                raise ValueError(f"row {index} column {name!r}: {exc}") from exc
        matrix.append(values)
    return matrix
