"""Small CSV helpers so examples do not depend on pandas."""

from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path
from typing import Iterable, Sequence


def read_dicts(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"{path} has no header")
        rows = list(reader)
        return list(reader.fieldnames), rows


def read_rows(path: Path) -> tuple[list[str] | None, list[list[str]]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        rows = list(reader)
    if not rows:
        return None, []
    return rows[0], rows[1:]


def float_column(rows: Sequence[dict[str, str]], name: str) -> list[float]:
    return [float(row[name]) for row in rows]


def int_column(rows: Sequence[dict[str, str]], name: str) -> list[int]:
    return [int(float(row[name])) for row in rows]


def label_counts(rows: Sequence[dict[str, str]], column: str = "sentiment_label") -> dict[str, int]:
    return dict(sorted(Counter(row[column] for row in rows).items()))


def sentence_ids(rows: Sequence[dict[str, str]], column: str = "sentence_id") -> list[str]:
    return [row[column] for row in rows]


def mean(values: Iterable[float]) -> float:
    data = list(values)
    if not data:
        raise ValueError("mean of empty sequence")
    return sum(data) / len(data)


def pstdev(values: Sequence[float]) -> float:
    if not values:
        raise ValueError("pstdev of empty sequence")
    mu = mean(values)
    return (sum((x - mu) ** 2 for x in values) / len(values)) ** 0.5
