"""Shared paths and CSV helpers for the personal gaze+sentiment examples.

Examples are meant to run from the repository root:

    python examples/inspect_sentence_gaze.py

They depend only on the Python standard library and numpy — not pandas,
PyTorch, or Hugging Face weights.
"""

from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]

ZUCO_SENT_DIR = ROOT / "ZuCo_et_csv_data"
ZUCO_WORD_DIR = ZUCO_SENT_DIR / "word"
ZUCO_SST_DIR = ROOT / "ZuCo_SST_data"
SST_DIR = ROOT / "SST_data"
GAZE_PRED_DIR = ROOT / "gaze_prediction" / "data"
RESULT_DIR = ROOT / "result"

# Fusion-head order used in model_ZuCo_SST.py / model_full_SST.py
GAZE5_ZUCO = ("nFixations", "FFD", "GPT", "TRT", "GD")
GAZE5_SST = ("nFix", "FFD", "GPT", "TRT", "GD")

SENTENCE_GAZE_COLS = (
    "SentLen",
    "omissionRate",
    "nFixations",
    "meanPupilSize",
    "GD",
    "TRT",
    "FFD",
    "SFD",
    "GPT",
)

WORD_GAZE_COLS = (
    "nFixations",
    "meanPupilSize",
    "GD",
    "TRT",
    "FFD",
    "SFD",
    "GPT",
    "WordLen",
)

LABEL_NAMES = {0: "negative", 1: "neutral", 2: "positive"}
SST_STRING_TO_INT = {"NEGATIVE": 0, "NEUTRAL": 1, "POSITIVE": 2}


def read_dicts(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def read_sst_string_labels(path: Path) -> list[tuple[str, str]]:
    """Load `stts_all_sentence_level.csv` (no header): sentence, LABEL."""
    rows: list[tuple[str, str]] = []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.reader(handle):
            if not row:
                continue
            if len(row) == 1:
                raise ValueError(f"Expected sentence+label, got one field: {row!r}")
            *sentence_parts, label = row
            sentence = ",".join(sentence_parts) if len(sentence_parts) > 1 else sentence_parts[0]
            rows.append((sentence, label.strip()))
    return rows


def floats(rows: Sequence[dict[str, str]], column: str) -> np.ndarray:
    values = np.empty(len(rows), dtype=np.float64)
    for i, row in enumerate(rows):
        raw = row[column]
        values[i] = np.nan if raw is None or raw == "" else float(raw)
    return values


def int_col(rows: Sequence[dict[str, str]], column: str) -> np.ndarray:
    return np.array([int(float(row[column])) for row in rows], dtype=np.int64)


def matrix(rows: Sequence[dict[str, str]], columns: Sequence[str]) -> np.ndarray:
    return np.column_stack([floats(rows, column) for column in columns])


def finite_stats(values: np.ndarray) -> dict[str, float]:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return {
            "count": 0.0,
            "nan_or_inf": float(values.size),
            "mean": float("nan"),
            "std": float("nan"),
            "min": float("nan"),
            "max": float("nan"),
        }
    return {
        "count": float(finite.size),
        "nan_or_inf": float(values.size - finite.size),
        "mean": float(finite.mean()),
        "std": float(finite.std(ddof=0)),
        "min": float(finite.min()),
        "max": float(finite.max()),
    }


def format_stats_table(columns: Sequence[str], arrays: Sequence[np.ndarray]) -> str:
    header = f"{'column':<16} {'n':>7} {'bad':>6} {'mean':>12} {'std':>12} {'min':>12} {'max':>12}"
    lines = [header, "-" * len(header)]
    for name, values in zip(columns, arrays):
        stats = finite_stats(values)
        lines.append(
            f"{name:<16} {int(stats['count']):>7d} {int(stats['nan_or_inf']):>6d} "
            f"{stats['mean']:12.4f} {stats['std']:12.4f} {stats['min']:12.4f} {stats['max']:12.4f}"
        )
    return "\n".join(lines)


def label_histogram(labels: Iterable[int]) -> str:
    counts = Counter(int(x) for x in labels)
    total = sum(counts.values()) or 1
    lines = [f"{'label':<10} {'name':<10} {'n':>6} {'pct':>7}"]
    lines.append("-" * 36)
    for key in sorted(set(LABEL_NAMES) | set(counts)):
        n = counts.get(key, 0)
        lines.append(
            f"{key:<10d} {LABEL_NAMES.get(key, '?'):<10} {n:>6d} {100.0 * n / total:6.1f}%"
        )
    lines.append(f"{'total':<10} {'':<10} {total:>6d} {100.0:6.1f}%")
    return "\n".join(lines)


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int = 3) -> np.ndarray:
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    for truth, pred in zip(y_true.astype(int), y_pred.astype(int)):
        if 0 <= truth < n_classes and 0 <= pred < n_classes:
            cm[truth, pred] += 1
    return cm


def precision_recall_f1_from_cm(cm: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    tp = np.diag(cm).astype(np.float64)
    fp = cm.sum(axis=0) - tp
    fn = cm.sum(axis=1) - tp
    precision = np.divide(tp, tp + fp, out=np.zeros_like(tp), where=(tp + fp) > 0)
    recall = np.divide(tp, tp + fn, out=np.zeros_like(tp), where=(tp + fn) > 0)
    denom = precision + recall
    f1 = np.divide(2 * precision * recall, denom, out=np.zeros_like(tp), where=denom > 0)
    return precision, recall, f1


def classification_report(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int = 3) -> dict:
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    cm = confusion_matrix(y_true, y_pred, n_classes=n_classes)
    precision, recall, f1 = precision_recall_f1_from_cm(cm)
    support = cm.sum(axis=1).astype(np.float64)
    total = support.sum() or 1.0
    accuracy = float((y_true == y_pred).mean()) if y_true.size else float("nan")
    macro = {
        "precision": float(precision.mean()),
        "recall": float(recall.mean()),
        "f1": float(f1.mean()),
    }
    weighted = {
        "precision": float(np.dot(precision, support) / total),
        "recall": float(np.dot(recall, support) / total),
        "f1": float(np.dot(f1, support) / total),
    }
    return {
        "accuracy": accuracy,
        "macro": macro,
        "weighted": weighted,
        "per_class": {
            i: {
                "name": LABEL_NAMES.get(i, str(i)),
                "precision": float(precision[i]),
                "recall": float(recall[i]),
                "f1": float(f1[i]),
                "support": int(support[i]),
            }
            for i in range(n_classes)
        },
        "confusion": cm,
    }


def format_report(report: dict) -> str:
    lines = [
        f"accuracy           {report['accuracy']:.4f}",
        f"macro  P/R/F1      {report['macro']['precision']:.4f} / "
        f"{report['macro']['recall']:.4f} / {report['macro']['f1']:.4f}",
        f"weight P/R/F1      {report['weighted']['precision']:.4f} / "
        f"{report['weighted']['recall']:.4f} / {report['weighted']['f1']:.4f}",
        "",
        f"{'class':<12} {'P':>8} {'R':>8} {'F1':>8} {'n':>6}",
        "-" * 46,
    ]
    for i, row in report["per_class"].items():
        lines.append(
            f"{i} {row['name']:<10} {row['precision']:8.4f} {row['recall']:8.4f} "
            f"{row['f1']:8.4f} {row['support']:6d}"
        )
    cm = report["confusion"]
    lines.append("")
    lines.append("confusion (rows=true, cols=pred):")
    lines.append("         " + " ".join(f"{i:>7d}" for i in range(cm.shape[1])))
    for i, row in enumerate(cm):
        lines.append(f"true {i}: " + " ".join(f"{v:7d}" for v in row))
    return "\n".join(lines)


def stratified_kfold_indices(y: np.ndarray, n_splits: int = 5, seed: int = 42) -> list[tuple[np.ndarray, np.ndarray]]:
    """Match the spirit of sklearn StratifiedKFold(shuffle=True, random_state=seed)."""
    rng = np.random.default_rng(seed)
    y = np.asarray(y, dtype=int)
    buckets: list[list[int]] = [[] for _ in range(n_splits)]
    for cls in np.unique(y):
        idx = np.where(y == cls)[0]
        rng.shuffle(idx)
        for i, sample in enumerate(idx):
            buckets[i % n_splits].append(int(sample))
    folds: list[tuple[np.ndarray, np.ndarray]] = []
    all_idx = np.arange(y.size)
    for k in range(n_splits):
        test = np.array(sorted(buckets[k]), dtype=int)
        mask = np.ones(y.size, dtype=bool)
        mask[test] = False
        train = all_idx[mask]
        folds.append((train, test))
    return folds


def majority_baseline(y_train: np.ndarray, n: int) -> np.ndarray:
    counts = np.bincount(y_train.astype(int), minlength=3)
    majority = int(np.argmax(counts))
    return np.full(n, majority, dtype=int)
