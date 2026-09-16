"""Small CSV helpers. Kept on the standard library so examples run cold."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable, Mapping, Sequence


def read_dicts(path: Path | str) -> list[dict[str, str]]:
    path = Path(path)
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def read_rows(path: Path | str) -> tuple[list[str], list[list[str]]]:
    path = Path(path)
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        try:
            header = next(reader)
        except StopIteration:
            return [], []
        return header, list(reader)


def write_dicts(
    path: Path | str,
    rows: Sequence[Mapping[str, object]],
    fieldnames: Sequence[str] | None = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows and fieldnames is None:
        path.write_text("", encoding="utf-8")
        return
    names = list(fieldnames or rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=names, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in names})


def as_float(value: object, default: float | None = None) -> float:
    if value is None or value == "":
        if default is None:
            raise ValueError("empty numeric field")
        return default
    return float(value)


def column_floats(rows: Iterable[Mapping[str, str]], name: str) -> list[float]:
    return [as_float(row[name]) for row in rows]


def count_lines(path: Path | str) -> int:
    path = Path(path)
    with path.open(encoding="utf-8") as handle:
        return sum(1 for _ in handle)
