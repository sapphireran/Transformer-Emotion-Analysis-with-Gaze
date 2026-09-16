"""Descriptive stats for measured ZuCo gaze, used by the personal examples."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Sequence


def _mean(values: Sequence[float]) -> float:
    if not values:
        raise ValueError("mean of empty sequence")
    return sum(values) / len(values)


def _stdev(values: Sequence[float]) -> float:
    if len(values) < 2:
        return 0.0
    mu = _mean(values)
    variance = sum((value - mu) ** 2 for value in values) / (len(values) - 1)
    return math.sqrt(variance)


def pearson(xs: Sequence[float], ys: Sequence[float]) -> float:
    if len(xs) != len(ys):
        raise ValueError("pearson expects equal-length series")
    if len(xs) < 2:
        return 0.0
    mx = _mean(xs)
    my = _mean(ys)
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    den_x = math.sqrt(sum((x - mx) ** 2 for x in xs))
    den_y = math.sqrt(sum((y - my) ** 2 for y in ys))
    if den_x == 0 or den_y == 0:
        return 0.0
    return num / (den_x * den_y)


def correlation_matrix(columns: dict[str, Sequence[float]]) -> list[list[float]]:
    names = list(columns)
    return [[pearson(columns[a], columns[b]) for b in names] for a in names]


@dataclass(frozen=True)
class ColumnSummary:
    name: str
    count: int
    mean: float
    stdev: float
    minimum: float
    maximum: float


def summarize_column(name: str, values: Iterable[float]) -> ColumnSummary:
    data = [float(v) for v in values]
    return ColumnSummary(
        name=name,
        count=len(data),
        mean=_mean(data) if data else float("nan"),
        stdev=_stdev(data) if data else float("nan"),
        minimum=min(data) if data else float("nan"),
        maximum=max(data) if data else float("nan"),
    )


def pairwise_subject_correlation(
    subject_series: dict[int, dict[int, float]],
) -> list[tuple[int, int, float, int]]:
    """Pearson r between every pair of readers, aligned on sentence id.

    `subject_series` maps subject -> {sentence_id: value}. Subject 3 has fewer
    sentences than the others; pairs use the intersection of ids.
    """
    subjects = sorted(subject_series)
    pairs: list[tuple[int, int, float, int]] = []
    for i, left in enumerate(subjects):
        for right in subjects[i + 1 :]:
            shared = sorted(set(subject_series[left]) & set(subject_series[right]))
            xs = [subject_series[left][sid] for sid in shared]
            ys = [subject_series[right][sid] for sid in shared]
            pairs.append((left, right, pearson(xs, ys), len(shared)))
    return pairs


def mean_pairwise(pairs: Sequence[tuple[int, int, float, int]]) -> float:
    if not pairs:
        return 0.0
    return sum(item[2] for item in pairs) / len(pairs)


def group_means(
    labels: Sequence[int],
    values: Sequence[float],
) -> dict[int, float]:
    buckets: dict[int, list[float]] = {}
    for label, value in zip(labels, values, strict=True):
        buckets.setdefault(int(label), []).append(float(value))
    return {label: _mean(vals) for label, vals in sorted(buckets.items())}
