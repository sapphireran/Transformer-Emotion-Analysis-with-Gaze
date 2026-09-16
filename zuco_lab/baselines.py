"""CPU baselines that do not touch transformers.

Used by examples to put a floor under the gaze-only story: majority class,
and a one-vs-rest linear readout on the five fusion gaze features.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from . import numbers


def _design(rows: Sequence[Sequence[float]]) -> list[list[float]]:
    return [[1.0, *row] for row in rows]


def _transpose(matrix: Sequence[Sequence[float]]) -> list[list[float]]:
    return [list(col) for col in zip(*matrix)]


def _matmul(a: Sequence[Sequence[float]], b: Sequence[Sequence[float]]) -> list[list[float]]:
    bt = _transpose(b)
    return [[sum(x * y for x, y in zip(row, col)) for col in bt] for row in a]


def _matvec(a: Sequence[Sequence[float]], v: Sequence[float]) -> list[float]:
    return [sum(x * y for x, y in zip(row, v)) for row in a]


def _solve(matrix: list[list[float]], rhs: list[float]) -> list[float]:
    """Gaussian elimination with partial pivoting. Matrix is square."""
    n = len(matrix)
    aug = [row[:] + [rhs[i]] for i, row in enumerate(matrix)]
    for col in range(n):
        pivot = max(range(col, n), key=lambda r: abs(aug[r][col]))
        if abs(aug[pivot][col]) < 1e-12:
            # Ridge fallback: add a small diagonal and restart once.
            return _solve_ridge(matrix, rhs, 1e-3)
        if pivot != col:
            aug[col], aug[pivot] = aug[pivot], aug[col]
        div = aug[col][col]
        for j in range(col, n + 1):
            aug[col][j] /= div
        for row in range(n):
            if row == col:
                continue
            factor = aug[row][col]
            for j in range(col, n + 1):
                aug[row][j] -= factor * aug[col][j]
    return [aug[i][n] for i in range(n)]


def _solve_ridge(matrix: list[list[float]], rhs: list[float], lam: float) -> list[float]:
    n = len(matrix)
    copy = [row[:] for row in matrix]
    for i in range(n):
        copy[i][i] += lam
    return _solve(copy, rhs)


def least_squares(x: Sequence[Sequence[float]], y: Sequence[float]) -> list[float]:
    xt = _transpose(x)
    xtx = _matmul(xt, x)
    xty = _matvec(xt, y)
    return _solve([row[:] for row in xtx], xty)


@dataclass(frozen=True)
class GazeLinearModel:
    weights: list[list[float]]  # one weight vector per class, including bias

    def scores(self, features: Sequence[float]) -> list[float]:
        design = [1.0, *features]
        return [sum(w * v for w, v in zip(weight, design)) for weight in self.weights]

    def predict(self, features: Sequence[float]) -> int:
        return numbers.argmax(self.scores(features))


def fit_gaze_ovr(features: Sequence[Sequence[float]], labels: Sequence[int], n_classes: int = 3) -> GazeLinearModel:
    x = _design(features)
    weights = []
    for k in range(n_classes):
        y = [1.0 if label == k else 0.0 for label in labels]
        weights.append(least_squares(x, y))
    return GazeLinearModel(weights=weights)


@dataclass(frozen=True)
class BaselineReport:
    name: str
    accuracy: float
    weighted_f1: float
    n: int


def evaluate_majority(labels: Sequence[int]) -> BaselineReport:
    mode = numbers.majority(labels)
    preds = [mode] * len(labels)
    return BaselineReport(
        name=f"majority({mode})",
        accuracy=numbers.accuracy(preds, labels),
        weighted_f1=numbers.weighted_f1(preds, labels),
        n=len(labels),
    )


def evaluate_gaze_linear(
    train_x: Sequence[Sequence[float]],
    train_y: Sequence[int],
    test_x: Sequence[Sequence[float]],
    test_y: Sequence[int],
) -> BaselineReport:
    model = fit_gaze_ovr(train_x, train_y)
    preds = [model.predict(row) for row in test_x]
    return BaselineReport(
        name="gaze_linear_ovr",
        accuracy=numbers.accuracy(preds, test_y),
        weighted_f1=numbers.weighted_f1(preds, test_y),
        n=len(test_y),
    )


def stratified_holdout(
    labels: Sequence[int],
    frac: float = 0.2,
    seed: int = 42,
) -> tuple[list[int], list[int]]:
    """Deterministic stratified index split without sklearn."""
    import random

    rng = random.Random(seed)
    by_label: dict[int, list[int]] = {}
    for i, label in enumerate(labels):
        by_label.setdefault(label, []).append(i)
    train, test = [], []
    for label in sorted(by_label):
        idxs = list(by_label[label])
        rng.shuffle(idxs)
        cut = max(1, int(round(len(idxs) * frac)))
        test.extend(idxs[:cut])
        train.extend(idxs[cut:])
    return sorted(train), sorted(test)
