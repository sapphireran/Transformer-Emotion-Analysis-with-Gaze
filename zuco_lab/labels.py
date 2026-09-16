"""Sentiment label conventions used by every checked-in table."""

from __future__ import annotations

from collections import Counter
from typing import Iterable, Mapping

LABEL_NAMES = {
    0: "negative",
    1: "neutral",
    2: "positive",
}

NAME_TO_ID = {name: idx for idx, name in LABEL_NAMES.items()}
NAME_TO_ID.update({"NEGATIVE": 0, "NEUTRAL": 1, "POSITIVE": 2})


def label_name(label: int | str) -> str:
    return LABEL_NAMES[int(label)]


def parse_label(value: object) -> int:
    if isinstance(value, str):
        stripped = value.strip()
        if stripped in NAME_TO_ID:
            return NAME_TO_ID[stripped]
        return int(stripped)
    return int(value)


def label_counts(rows: Iterable[Mapping[str, str]], column: str = "sentiment_label") -> dict[int, int]:
    counts: Counter[int] = Counter()
    for row in rows:
        counts[parse_label(row[column])] += 1
    return {idx: counts.get(idx, 0) for idx in (0, 1, 2)}


def label_rates(counts: Mapping[int, int]) -> dict[int, float]:
    total = sum(counts.values()) or 1
    return {idx: counts.get(idx, 0) / total for idx in (0, 1, 2)}
