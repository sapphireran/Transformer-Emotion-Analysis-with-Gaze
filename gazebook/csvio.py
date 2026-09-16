"""Small CSV helpers. Pandas is not required (and is not installed here)."""

from __future__ import annotations

import csv
from collections.abc import Iterable, Sequence
from pathlib import Path

import numpy as np


def read_dicts(path: str | Path) -> tuple[list[str], list[dict[str, str]]]:
    path = Path(path)
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"{path} has no header row")
        rows = list(reader)
        fields = list(reader.fieldnames)
    return fields, rows


def read_headerless(path: str | Path) -> list[list[str]]:
    path = Path(path)
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.reader(handle))


def float_col(rows: Sequence[dict[str, str]], name: str) -> np.ndarray:
    return np.array([float(row[name]) for row in rows], dtype=np.float64)


def int_col(rows: Sequence[dict[str, str]], name: str) -> np.ndarray:
    return np.array([int(row[name]) for row in rows], dtype=np.int64)


def table_to_array(rows: Sequence[dict[str, str]], cols: Iterable[str]) -> np.ndarray:
    cols = list(cols)
    return np.array([[float(row[c]) for c in cols] for row in rows], dtype=np.float64)
