"""CPU stand-ins for the late-fusion head. No torch, no Hugging Face.

``EyeTrackingModel`` concatenates a 768-d pooled transformer vector with a
Linear(5 → 16) gaze projection. Examples here replace the transformer with
a hashed bag-of-words and replace the MLP with ridge one-vs-rest so the
same five gaze columns can be probed on a laptop.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass

import numpy as np

from .metrics import Scores, weighted_scores

TOKEN_RE = re.compile(r"[A-Za-z0-9']+")
LABELS = (0, 1, 2)


def tokenize(text: str) -> list[str]:
    return [m.group(0).lower() for m in TOKEN_RE.finditer(text or "")]


def hash_bow(texts: list[str], dim: int = 128, seed: str = "gazebook") -> np.ndarray:
    """Signed feature-hashing. Stable across processes (md5, not PYTHONHASHSEED)."""
    X = np.zeros((len(texts), dim), dtype=np.float64)
    for i, text in enumerate(texts):
        for tok in tokenize(text):
            digest = hashlib.md5(f"{seed}:{tok}".encode("utf-8")).digest()
            idx = int.from_bytes(digest[:4], "little") % dim
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            X[i, idx] += sign
    return X


def add_bias(X: np.ndarray) -> np.ndarray:
    return np.column_stack([np.ones(len(X)), np.asarray(X, dtype=np.float64)])


def ridge_ovr_fit(X: np.ndarray, y: np.ndarray, l2: float = 1.0) -> np.ndarray:
    """Return a (n_classes × n_features+1) weight matrix, bias in column 0."""
    Xb = add_bias(X)
    n_features = Xb.shape[1]
    weights = []
    eye = np.eye(n_features)
    eye[0, 0] = 0.0  # do not regularize the intercept
    xtx = Xb.T @ Xb
    for label in LABELS:
        target = (np.asarray(y) == label).astype(np.float64)
        w = np.linalg.solve(xtx + l2 * eye, Xb.T @ target)
        weights.append(w)
    return np.stack(weights, axis=0)


def ridge_ovr_predict(W: np.ndarray, X: np.ndarray) -> np.ndarray:
    scores = add_bias(X) @ W.T
    return scores.argmax(axis=1)


def majority_predict(y_train: np.ndarray, n: int) -> np.ndarray:
    counts = np.bincount(np.asarray(y_train, dtype=int), minlength=3)
    return np.full(n, int(counts.argmax()), dtype=int)


def stratified_kfold(y: np.ndarray, n_splits: int = 5, seed: int = 42) -> list[tuple[np.ndarray, np.ndarray]]:
    """Deterministic stratified folds. Not bit-identical to sklearn, but stable."""
    y = np.asarray(y)
    rng = np.random.default_rng(seed)
    folds: list[list[int]] = [[] for _ in range(n_splits)]
    for label in LABELS:
        idx = np.flatnonzero(y == label)
        rng.shuffle(idx)
        for i, row in enumerate(idx):
            folds[i % n_splits].append(int(row))
    splits = []
    all_idx = np.arange(len(y))
    for k in range(n_splits):
        test = np.array(sorted(folds[k]), dtype=int)
        mask = np.ones(len(y), dtype=bool)
        mask[test] = False
        train = all_idx[mask]
        splits.append((train, test))
    return splits


@dataclass(frozen=True)
class FoldResult:
    name: str
    scores: list[Scores]

    @property
    def mean_acc(self) -> float:
        return float(np.mean([s.accuracy for s in self.scores]))

    @property
    def mean_f1(self) -> float:
        return float(np.mean([s.f1 for s in self.scores]))

    @property
    def std_acc(self) -> float:
        return float(np.std([s.accuracy for s in self.scores]))


def cv_ridge(X: np.ndarray, y: np.ndarray, n_splits: int = 5, seed: int = 42, l2: float = 1.0) -> FoldResult:
    scores = []
    for train, test in stratified_kfold(y, n_splits=n_splits, seed=seed):
        W = ridge_ovr_fit(X[train], y[train], l2=l2)
        pred = ridge_ovr_predict(W, X[test])
        scores.append(weighted_scores(y[test], pred))
    return FoldResult(name="ridge", scores=scores)


def cv_majority(y: np.ndarray, n_splits: int = 5, seed: int = 42) -> FoldResult:
    scores = []
    dummy = np.zeros((len(y), 1))
    for train, test in stratified_kfold(y, n_splits=n_splits, seed=seed):
        pred = majority_predict(y[train], len(test))
        scores.append(weighted_scores(y[test], pred))
        _ = dummy  # keep signature obvious; no features used
    return FoldResult(name="majority", scores=scores)


def permute_rows(X: np.ndarray, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    out = np.asarray(X, dtype=np.float64).copy()
    rng.shuffle(out)
    return out
