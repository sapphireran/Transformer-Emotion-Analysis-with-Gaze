"""Small dense linear algebra used by the fusion demo."""

from __future__ import annotations

import math
import random
from typing import Iterable, List, Sequence

Matrix = List[List[float]]
Vector = List[float]


def zeros(rows: int, cols: int) -> Matrix:
    return [[0.0] * cols for _ in range(rows)]


def full(rows: int, cols: int, value: float) -> Matrix:
    return [[value] * cols for _ in range(rows)]


def identity(n: int) -> Matrix:
    out = zeros(n, n)
    for i in range(n):
        out[i][i] = 1.0
    return out


def shape(matrix: Matrix) -> tuple[int, int]:
    if not matrix:
        return (0, 0)
    return (len(matrix), len(matrix[0]))


def transpose(matrix: Matrix) -> Matrix:
    rows, cols = shape(matrix)
    return [[matrix[i][j] for i in range(rows)] for j in range(cols)]


def matvec(matrix: Matrix, vector: Sequence[float]) -> Vector:
    out: Vector = []
    for row in matrix:
        if len(row) != len(vector):
            raise ValueError("matvec() width mismatch")
        out.append(sum(a * b for a, b in zip(row, vector)))
    return out


def add_vec(a: Sequence[float], b: Sequence[float]) -> Vector:
    if len(a) != len(b):
        raise ValueError("add_vec() length mismatch")
    return [x + y for x, y in zip(a, b)]


def sub_vec(a: Sequence[float], b: Sequence[float]) -> Vector:
    if len(a) != len(b):
        raise ValueError("sub_vec() length mismatch")
    return [x - y for x, y in zip(a, b)]


def scale_vec(vector: Sequence[float], scalar: float) -> Vector:
    return [scalar * x for x in vector]


def concat_vec(parts: Iterable[Sequence[float]]) -> Vector:
    out: Vector = []
    for part in parts:
        out.extend(part)
    return out


def dot(a: Sequence[float], b: Sequence[float]) -> float:
    if len(a) != len(b):
        raise ValueError("dot() length mismatch")
    return sum(x * y for x, y in zip(a, b))


def l2(vector: Sequence[float]) -> float:
    return math.sqrt(sum(x * x for x in vector))


def normalize(vector: Sequence[float]) -> Vector:
    norm = l2(vector)
    if norm == 0.0:
        return [0.0 for _ in vector]
    return [x / norm for x in vector]


def softmax(logits: Sequence[float]) -> Vector:
    if not logits:
        return []
    peak = max(logits)
    exps = [math.exp(x - peak) for x in logits]
    total = sum(exps)
    if total == 0.0:
        return [1.0 / len(logits)] * len(logits)
    return [x / total for x in exps]


def xavier_matrix(rows: int, cols: int, rng: random.Random) -> Matrix:
    # Uniform[-sqrt(6/(in+out)), +sqrt(6/(in+out))]
    limit = math.sqrt(6.0 / (rows + cols)) if (rows + cols) else 0.05
    return [[rng.uniform(-limit, limit) for _ in range(cols)] for _ in range(rows)]


def xavier_vector(size: int, rng: random.Random) -> Vector:
    limit = math.sqrt(6.0 / max(size, 1))
    return [rng.uniform(-limit, limit) for _ in range(size)]


def one_hot(index: int, size: int) -> Vector:
    if not 0 <= index < size:
        raise ValueError(f"one_hot index {index} out of range {size}")
    vec = [0.0] * size
    vec[index] = 1.0
    return vec


def argmax(values: Sequence[float]) -> int:
    if not values:
        raise ValueError("argmax() of empty sequence")
    best_i = 0
    best_v = values[0]
    for i, value in enumerate(values):
        if value > best_v:
            best_i = i
            best_v = value
    return best_i


def clip_vec(vector: Sequence[float], max_abs: float = 5.0) -> Vector:
    return [max(-max_abs, min(max_abs, float(x))) for x in vector]


def clip_matrix(matrix: Matrix, max_abs: float = 5.0) -> Matrix:
    return [clip_vec(row, max_abs=max_abs) for row in matrix]


def finite(values: Iterable[float]) -> bool:
    return all(math.isfinite(float(x)) for x in values)
