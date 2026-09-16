"""Shared helpers for the lightweight examples.

Only the standard library and numpy are required. Training scripts stay
untouched; these helpers exist so each example does not re-implement
CSV parsing, metric tables, or repo-root discovery.
"""

from __future__ import annotations

import csv
import os
from collections import Counter
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

LABEL_NAMES = {0: "negative", 1: "neutral", 2: "positive"}

# Columns the transformer fusion head actually consumes.
ZUCO_GAZE_COLS = ["nFixations", "FFD", "GPT", "TRT", "GD"]
SST_GAZE_COLS = ["nFix", "FFD", "GPT", "TRT", "GD"]
ZUCO_EXTRA_GAZE = ["omissionRate", "meanPupilSize", "SFD"]


def repo_root() -> Path:
    """Return the repository root even if cwd is examples/."""
    here = Path(__file__).resolve().parent
    return here.parent


def resolve(relative: str) -> Path:
    return repo_root() / relative


def read_csv(relative: str, *, has_header: bool = True) -> tuple[list[str], list[list[str]]]:
    path = resolve(relative)
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        if has_header:
            header = next(reader)
            rows = [row for row in reader if row]
            return header, rows
        rows = [row for row in reader if row]
        return [], rows


def column_index(header: Sequence[str], name: str) -> int:
    try:
        return list(header).index(name)
    except ValueError as exc:
        raise KeyError(f"column {name!r} not in {list(header)}") from exc


def as_float_column(rows: Sequence[Sequence[str]], header: Sequence[str], name: str) -> np.ndarray:
    idx = column_index(header, name)
    values = []
    for row in rows:
        raw = row[idx].strip()
        values.append(float(raw) if raw else np.nan)
    return np.asarray(values, dtype=np.float64)


def as_int_column(rows: Sequence[Sequence[str]], header: Sequence[str], name: str) -> np.ndarray:
    idx = column_index(header, name)
    return np.asarray([int(float(row[idx])) for row in rows], dtype=np.int64)


def matrix(rows: Sequence[Sequence[str]], header: Sequence[str], names: Sequence[str]) -> np.ndarray:
    return np.column_stack([as_float_column(rows, header, name) for name in names])


def label_counts(labels: Iterable[int]) -> dict[int, int]:
    return dict(sorted(Counter(int(x) for x in labels).items()))


def format_counts(counts: dict[int, int]) -> str:
    parts = []
    for key in (0, 1, 2):
        if key in counts:
            parts.append(f"{key} {LABEL_NAMES[key]}={counts[key]}")
    extras = [f"{k}={v}" for k, v in counts.items() if k not in LABEL_NAMES]
    return ", ".join(parts + extras)


def whitespace_token_count(sentence: str) -> int:
    return len(sentence.split())


def mean_std(values: np.ndarray) -> tuple[float, float]:
    clean = values[np.isfinite(values)]
    if clean.size == 0:
        return float("nan"), float("nan")
    return float(clean.mean()), float(clean.std(ddof=0))


def correlation_matrix(x: np.ndarray) -> np.ndarray:
    """Pearson correlation; columns with zero variance become NaN."""
    x = np.asarray(x, dtype=np.float64)
    centered = x - np.nanmean(x, axis=0)
    std = np.nanstd(x, axis=0)
    std = np.where(std == 0, np.nan, std)
    cov = np.dot(np.nan_to_num(centered).T, np.nan_to_num(centered)) / max(len(x), 1)
    outer = np.outer(std, std)
    return cov / outer


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int = 3) -> np.ndarray:
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    for t, p in zip(y_true.astype(int), y_pred.astype(int)):
        if 0 <= t < n_classes and 0 <= p < n_classes:
            cm[t, p] += 1
    return cm


def classification_scores(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Accuracy plus support-weighted P / R / F1 (sklearn-style)."""
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    cm = confusion_matrix(y_true, y_pred)
    support = cm.sum(axis=1)
    pred_count = cm.sum(axis=0)
    tp = np.diag(cm).astype(np.float64)
    precision = np.divide(tp, pred_count, out=np.zeros_like(tp), where=pred_count > 0)
    recall = np.divide(tp, support, out=np.zeros_like(tp), where=support > 0)
    denom = precision + recall
    f1 = np.divide(2 * precision * recall, denom, out=np.zeros_like(tp), where=denom > 0)
    total = support.sum()
    if total == 0:
        return {"accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0}
    weights = support / total
    return {
        "accuracy": float(tp.sum() / total),
        "precision": float(np.dot(precision, weights)),
        "recall": float(np.dot(recall, weights)),
        "f1": float(np.dot(f1, weights)),
    }


def format_scores(scores: dict[str, float]) -> str:
    return (
        f"acc={scores['accuracy']:.4f}  "
        f"P={scores['precision']:.4f}  "
        f"R={scores['recall']:.4f}  "
        f"F1={scores['f1']:.4f}"
    )


def stratified_kfold(y: np.ndarray, n_splits: int = 5, seed: int = 42) -> list[tuple[np.ndarray, np.ndarray]]:
    """Deterministic stratified folds (same idea as sklearn StratifiedKFold)."""
    rng = np.random.RandomState(seed)
    y = np.asarray(y, dtype=int)
    folds: list[list[int]] = [[] for _ in range(n_splits)]
    for label in np.unique(y):
        idx = np.where(y == label)[0]
        rng.shuffle(idx)
        for i, sample in enumerate(idx):
            folds[i % n_splits].append(int(sample))
    splits = []
    for i in range(n_splits):
        test = np.asarray(folds[i], dtype=int)
        train = np.concatenate(
            [np.asarray(folds[j], dtype=int) for j in range(n_splits) if j != i]
        )
        splits.append((train, test))
    return splits


def ensure_output_dir() -> Path:
    out = Path(__file__).resolve().parent / "sample_outputs"
    out.mkdir(parents=True, exist_ok=True)
    return out


def write_text(name: str, content: str) -> Path:
    path = ensure_output_dir() / name
    path.write_text(content if content.endswith("\n") else content + "\n", encoding="utf-8")
    return path


def table(headers: Sequence[str], rows: Sequence[Sequence[object]], col_width: int | None = None) -> str:
    str_rows = [[str(c) for c in row] for row in rows]
    widths = [len(h) for h in headers]
    for row in str_rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))
    if col_width:
        widths = [max(w, col_width) for w in widths]

    def fmt(cells: Sequence[str]) -> str:
        return "  ".join(cell.ljust(widths[i]) for i, cell in enumerate(cells))

    lines = [fmt(headers), fmt(["-" * w for w in widths])]
    lines.extend(fmt(row) for row in str_rows)
    return "\n".join(lines)
