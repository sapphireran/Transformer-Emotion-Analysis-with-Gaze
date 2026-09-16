"""Numeric helpers used by audits and the CPU fusion demo.

No NumPy. The functions are small on purpose: they are easy to test and they
keep the examples honest about what they compute.
"""

from __future__ import annotations

import math
import random
from typing import Iterable, Sequence


def mean(values: Sequence[float]) -> float:
    if not values:
        raise ValueError("mean of empty sequence")
    return sum(values) / len(values)


def pstdev(values: Sequence[float]) -> float:
    if not values:
        raise ValueError("pstdev of empty sequence")
    mu = mean(values)
    return math.sqrt(sum((value - mu) ** 2 for value in values) / len(values))


def variance(values: Sequence[float]) -> float:
    sigma = pstdev(values)
    return sigma * sigma


def median(values: Sequence[float]) -> float:
    if not values:
        raise ValueError("median of empty sequence")
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return float(ordered[mid])
    return (ordered[mid - 1] + ordered[mid]) / 2.0


def quantile(values: Sequence[float], q: float) -> float:
    if not 0.0 <= q <= 1.0:
        raise ValueError("q must be in [0, 1]")
    if not values:
        raise ValueError("quantile of empty sequence")
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    pos = q * (len(ordered) - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return float(ordered[lo])
    weight = pos - lo
    return ordered[lo] * (1.0 - weight) + ordered[hi] * weight


def pearson(xs: Sequence[float], ys: Sequence[float]) -> float:
    if len(xs) != len(ys):
        raise ValueError("pearson length mismatch")
    if len(xs) < 2:
        return 0.0
    mx, my = mean(xs), mean(ys)
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    dx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    dy = math.sqrt(sum((y - my) ** 2 for y in ys))
    if dx == 0.0 or dy == 0.0:
        return 0.0
    return num / (dx * dy)


def spearman(xs: Sequence[float], ys: Sequence[float]) -> float:
    return pearson(_ranks(xs), _ranks(ys))


def _ranks(values: Sequence[float]) -> list[float]:
    indexed = sorted(enumerate(values), key=lambda item: item[1])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(indexed):
        j = i
        while j + 1 < len(indexed) and indexed[j + 1][1] == indexed[i][1]:
            j += 1
        rank = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[indexed[k][0]] = rank
        i = j + 1
    return ranks


def coeff_of_variation(values: Sequence[float]) -> float:
    mu = mean(values)
    if mu == 0.0:
        return 0.0
    return pstdev(values) / abs(mu)


def softmax(logits: Sequence[float]) -> list[float]:
    peak = max(logits)
    exps = [math.exp(value - peak) for value in logits]
    total = sum(exps)
    return [value / total for value in exps]


def argmax(values: Sequence[float]) -> int:
    if not values:
        raise ValueError("argmax of empty sequence")
    best_i = 0
    best_v = values[0]
    for i, value in enumerate(values):
        if value > best_v:
            best_i, best_v = i, value
    return best_i


def matvec(matrix: Sequence[Sequence[float]], vector: Sequence[float]) -> list[float]:
    if matrix and len(matrix[0]) != len(vector):
        raise ValueError("matvec shape mismatch")
    return [sum(weight * value for weight, value in zip(row, vector)) for row in matrix]


def add(a: Sequence[float], b: Sequence[float]) -> list[float]:
    if len(a) != len(b):
        raise ValueError("add length mismatch")
    return [x + y for x, y in zip(a, b)]


def concat(*parts: Sequence[float]) -> list[float]:
    out: list[float] = []
    for part in parts:
        out.extend(part)
    return out


def clamp(value: float, lo: float, hi: float) -> float:
    return lo if value < lo else hi if value > hi else value


def seeded_uniform(rng: random.Random, n: int, lo: float = -0.1, hi: float = 0.1) -> list[float]:
    return [rng.uniform(lo, hi) for _ in range(n)]


def seeded_matrix(
    rng: random.Random, rows: int, cols: int, lo: float = -0.1, hi: float = 0.1
) -> list[list[float]]:
    return [seeded_uniform(rng, cols, lo, hi) for _ in range(rows)]


def majority(labels: Iterable[int]) -> int:
    counts: dict[int, int] = {}
    for label in labels:
        counts[label] = counts.get(label, 0) + 1
    if not counts:
        raise ValueError("majority of empty labels")
    return max(counts.items(), key=lambda item: (item[1], -item[0]))[0]


def accuracy(preds: Sequence[int], labels: Sequence[int]) -> float:
    if len(preds) != len(labels):
        raise ValueError("accuracy length mismatch")
    if not labels:
        raise ValueError("accuracy of empty sequence")
    return sum(int(p == y) for p, y in zip(preds, labels)) / len(labels)


def confusion(preds: Sequence[int], labels: Sequence[int], n_classes: int = 3) -> list[list[int]]:
    table = [[0 for _ in range(n_classes)] for _ in range(n_classes)]
    for pred, label in zip(preds, labels):
        table[label][pred] += 1
    return table


def weighted_f1(preds: Sequence[int], labels: Sequence[int], n_classes: int = 3) -> float:
    table = confusion(preds, labels, n_classes)
    support = [sum(table[k]) for k in range(n_classes)]
    total = sum(support)
    if total == 0:
        return 0.0
    score = 0.0
    for k in range(n_classes):
        tp = table[k][k]
        fp = sum(table[i][k] for i in range(n_classes) if i != k)
        fn = support[k] - tp
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        score += f1 * support[k]
    return score / total


def zscore(values: Sequence[float]) -> list[float]:
    mu = mean(values)
    sigma = pstdev(values)
    if sigma == 0.0:
        return [0.0 for _ in values]
    return [(value - mu) / sigma for value in values]


def minmax(values: Sequence[float]) -> list[float]:
    lo, hi = min(values), max(values)
    if hi == lo:
        return [0.0 for _ in values]
    return [(value - lo) / (hi - lo) for value in values]
