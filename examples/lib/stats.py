"""Descriptive statistics without numpy."""

from __future__ import annotations

import math
from collections import Counter
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


def _finite(values: Iterable[Optional[float]]) -> List[float]:
    return [float(v) for v in values if v is not None and math.isfinite(float(v))]


def mean(values: Sequence[Optional[float]]) -> float:
    data = _finite(values)
    if not data:
        raise ValueError("mean() of empty sequence")
    return sum(data) / len(data)


def variance(values: Sequence[Optional[float]], *, sample: bool = True) -> float:
    data = _finite(values)
    if len(data) < 2:
        return 0.0
    mu = sum(data) / len(data)
    denom = (len(data) - 1) if sample else len(data)
    return sum((x - mu) ** 2 for x in data) / denom


def std(values: Sequence[Optional[float]], *, sample: bool = True) -> float:
    return math.sqrt(variance(values, sample=sample))


def quantile(values: Sequence[Optional[float]], q: float) -> float:
    if not 0.0 <= q <= 1.0:
        raise ValueError("q must be in [0, 1]")
    data = sorted(_finite(values))
    if not data:
        raise ValueError("quantile() of empty sequence")
    if len(data) == 1:
        return data[0]
    pos = (len(data) - 1) * q
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return data[lo]
    weight = pos - lo
    return data[lo] * (1.0 - weight) + data[hi] * weight


def summarize(values: Sequence[Optional[float]]) -> Dict[str, float]:
    data = _finite(values)
    if not data:
        return {"n": 0.0}
    return {
        "n": float(len(data)),
        "mean": mean(data),
        "std": std(data),
        "min": min(data),
        "q25": quantile(data, 0.25),
        "median": quantile(data, 0.50),
        "q75": quantile(data, 0.75),
        "max": max(data),
    }


def counts(values: Iterable[object]) -> Dict[object, int]:
    return dict(Counter(values))


def majority_baseline(labels: Sequence[int]) -> Tuple[int, float]:
    if not labels:
        raise ValueError("majority_baseline() of empty sequence")
    tallies = Counter(labels)
    winner, count = max(tallies.items(), key=lambda item: (item[1], -item[0]))
    return winner, count / len(labels)


def pearson(xs: Sequence[float], ys: Sequence[float]) -> float:
    if len(xs) != len(ys):
        raise ValueError("pearson() length mismatch")
    pairs = [
        (float(x), float(y))
        for x, y in zip(xs, ys)
        if math.isfinite(float(x)) and math.isfinite(float(y))
    ]
    if len(pairs) < 2:
        return 0.0
    mx = sum(p[0] for p in pairs) / len(pairs)
    my = sum(p[1] for p in pairs) / len(pairs)
    num = sum((x - mx) * (y - my) for x, y in pairs)
    dx = math.sqrt(sum((x - mx) ** 2 for x, _ in pairs))
    dy = math.sqrt(sum((y - my) ** 2 for _, y in pairs))
    if dx == 0.0 or dy == 0.0:
        return 0.0
    return num / (dx * dy)


def zscore_columns(matrix: Sequence[Sequence[float]]) -> List[List[float]]:
    if not matrix:
        return []
    width = len(matrix[0])
    cols: List[List[float]] = []
    for j in range(width):
        values = [row[j] for row in matrix]
        mu = mean(values)
        sigma = std(values, sample=False)
        if sigma == 0.0:
            cols.append([0.0 for _ in values])
        else:
            cols.append([(v - mu) / sigma for v in values])
    return [[cols[j][i] for j in range(width)] for i in range(len(matrix))]
