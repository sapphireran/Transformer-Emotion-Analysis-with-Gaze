#!/usr/bin/env python3
"""A CPU-only sketch of the late-fusion idea.

The real trainers fine-tune BERT / RoBERTa (~110M parameters) and glue a
16-d gaze projection onto pooler_output. This script throws that encoder
away and asks a smaller question:

    Do the five sentence-level gaze numbers have *any* linear signal
    for 3-class sentiment on the 400 measured ZuCo sentences — alone,
    or added to a handful of cheap text statistics?

That is not a substitute for the transformer. It is a sanity check you
can run in seconds before spending a GPU night. If gaze-only cannot
beat the majority class, do not expect miracles from Linear(5, 16).

Protocol (mirrors model_ZuCo_SST.py):
  - table: ZuCo_SST_data/combined_sst_et_standard.csv
  - five gaze columns in the same order as EyeTrackingModel
  - Stratified 5-fold, seed 42
  - multinomial logistic regression trained with numpy (softmax + L2)

Comparisons per fold, then mean ± std:
  majority          always predict the training-fold mode
  gaze              5-d z-scored gaze
  text-stats        length / punctuation / tiny polarity lexicon
  late-fusion       concat(text-stats, gaze)

    python3 examples/06_toy_late_fusion.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np

from common import (
    LABEL_NAMES,
    ZUCO_GAZE_COLS,
    as_int_column,
    classification_scores,
    confusion_matrix,
    format_counts,
    format_scores,
    label_counts,
    matrix,
    read_csv,
    stratified_kfold,
    table,
    write_text,
)

# Tiny, personal, movie-review flavored lists. Not VADER, not a paper lexicon.
NEG_WORDS = {
    "bad",
    "worst",
    "dull",
    "boring",
    "empty",
    "unsatisfying",
    "pointless",
    "violent",
    "flawed",
    "silly",
    "stupid",
    "waste",
    "fails",
    "failing",
    "mediocre",
    "bland",
    "ugly",
    "mess",
    "disaster",
    "witless",
    "inane",
}
POS_WORDS = {
    "good",
    "great",
    "beautiful",
    "beautifully",
    "brilliant",
    "engaging",
    "quality",
    "best",
    "love",
    "wonderful",
    "funny",
    "hilarious",
    "remarkable",
    "astonishing",
    "mesmerizing",
    "perfect",
    "excellent",
    "delight",
    "moving",
    "smart",
}


def tokenize(sentence: str) -> list[str]:
    cleaned = []
    buf = []
    for ch in sentence.lower():
        if ch.isalpha():
            buf.append(ch)
        else:
            if buf:
                cleaned.append("".join(buf))
                buf = []
    if buf:
        cleaned.append("".join(buf))
    return cleaned


def text_stats(sentences: list[str]) -> np.ndarray:
    """Cheap features that stand in for pooler_output in this toy only."""
    feats = []
    for sent in sentences:
        tokens = tokenize(sent)
        n = max(len(tokens), 1)
        chars = max(len(sent), 1)
        punct = sum(sent.count(p) for p in ".,;:!?")
        neg = sum(1 for t in tokens if t in NEG_WORDS)
        pos = sum(1 for t in tokens if t in POS_WORDS)
        feats.append(
            [
                n,
                chars / 10.0,
                punct,
                neg,
                pos,
                (pos - neg) / n,
                sum(len(t) for t in tokens) / n,
            ]
        )
    return np.asarray(feats, dtype=np.float64)


def standardize_fit(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = x.mean(axis=0)
    std = x.std(axis=0)
    std = np.where(std < 1e-8, 1.0, std)
    return mean, std


def standardize_apply(x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return (x - mean) / std


def softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def fit_softmax(
    x: np.ndarray,
    y: np.ndarray,
    *,
    lr: float = 0.25,
    epochs: int = 400,
    l2: float = 1e-2,
) -> tuple[np.ndarray, np.ndarray]:
    n, d = x.shape
    k = 3
    w = np.zeros((d, k), dtype=np.float64)
    b = np.zeros(k, dtype=np.float64)
    eye = np.eye(k)[y]
    for _ in range(epochs):
        probs = softmax(x @ w + b)
        err = (probs - eye) / n
        w -= lr * (x.T @ err + l2 * w)
        b -= lr * err.sum(axis=0)
    return w, b


def predict_softmax(x: np.ndarray, w: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.argmax(x @ w + b, axis=1)


def majority(y_train: np.ndarray, n_test: int) -> np.ndarray:
    counts = np.bincount(y_train, minlength=3)
    return np.full(n_test, int(np.argmax(counts)), dtype=int)


def run_fold(
    x: np.ndarray,
    y: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
) -> np.ndarray:
    mean, std = standardize_fit(x[train_idx])
    x_tr = standardize_apply(x[train_idx], mean, std)
    x_te = standardize_apply(x[test_idx], mean, std)
    w, b = fit_softmax(x_tr, y[train_idx])
    return predict_softmax(x_te, w, b)


def evaluate_splits(
    name: str,
    x: np.ndarray | None,
    y: np.ndarray,
    splits: list[tuple[np.ndarray, np.ndarray]],
    *,
    kind: str,
) -> tuple[list[dict[str, float]], np.ndarray, np.ndarray]:
    fold_scores = []
    all_true: list[np.ndarray] = []
    all_pred: list[np.ndarray] = []
    for train_idx, test_idx in splits:
        if kind == "majority":
            pred = majority(y[train_idx], len(test_idx))
        else:
            assert x is not None
            pred = run_fold(x, y, train_idx, test_idx)
        scores = classification_scores(y[test_idx], pred)
        fold_scores.append(scores)
        all_true.append(y[test_idx])
        all_pred.append(pred)
    _ = name
    return fold_scores, np.concatenate(all_true), np.concatenate(all_pred)


def mean_std_table(methods: dict[str, list[dict[str, float]]]) -> str:
    rows = []
    for name, folds in methods.items():
        acc = np.array([s["accuracy"] for s in folds])
        f1 = np.array([s["f1"] for s in folds])
        rows.append(
            (
                name,
                f"{acc.mean():.4f} ± {acc.std(ddof=0):.4f}",
                f"{f1.mean():.4f} ± {f1.std(ddof=0):.4f}",
            )
        )
    return table(["method", "accuracy (mean ± std)", "weighted F1 (mean ± std)"], rows)


def fold_detail(methods: dict[str, list[dict[str, float]]]) -> str:
    rows = []
    n_folds = len(next(iter(methods.values())))
    for i in range(n_folds):
        cells = [str(i)]
        for name in methods:
            cells.append(format_scores(methods[name][i]))
        rows.append(cells)
    return table(["fold", *methods.keys()], rows)


def cm_block(y_true: np.ndarray, y_pred: np.ndarray) -> str:
    cm = confusion_matrix(y_true, y_pred)
    rows = []
    for i in range(3):
        rows.append((f"true {i} {LABEL_NAMES[i]}", *[str(int(cm[i, j])) for j in range(3)]))
    return table(["", "pred 0", "pred 1", "pred 2"], rows)


def main() -> int:
    header, rows = read_csv("ZuCo_SST_data/combined_sst_et_standard.csv")
    sentences = [row[header.index("sentence")] for row in rows]
    y = as_int_column(rows, header, "sentiment_label")
    gaze = matrix(rows, header, ZUCO_GAZE_COLS)
    text = text_stats(sentences)
    fused = np.concatenate([text, gaze], axis=1)
    splits = stratified_kfold(y, n_splits=5, seed=42)

    setups = {
        "majority": (None, "majority"),
        "gaze": (gaze, "model"),
        "text-stats": (text, "model"),
        "late-fusion": (fused, "model"),
    }

    methods: dict[str, list[dict[str, float]]] = {}
    pooled: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for name, (x, kind) in setups.items():
        scores, yt, yp = evaluate_splits(name, x, y, splits, kind=kind)
        methods[name] = scores
        pooled[name] = (yt, yp)

    lines = [
        "# Toy late-fusion baseline (no transformer)",
        "",
        f"rows: {len(rows)}  labels: {format_counts(label_counts(y))}",
        f"gaze columns (EyeTrackingModel order): {', '.join(ZUCO_GAZE_COLS)}",
        f"text-stat dim: {text.shape[1]}  fused dim: {fused.shape[1]}",
        "folds: Stratified 5-fold, seed 42 (same idea as model_ZuCo_SST.py)",
        "",
        "text-stats = [token count, chars/10, punct count, #neg lexicon,",
        "              #pos lexicon, (pos-neg)/n, mean word length]",
        "lexicon is a tiny hand list in this file — not a published resource.",
        "",
        "## mean over folds",
        mean_std_table(methods),
        "",
        "## per-fold scores",
        fold_detail(methods),
        "",
        "## pooled-fold confusion (late-fusion)",
        cm_block(*pooled["late-fusion"]),
        "",
        "## pooled-fold confusion (gaze only)",
        cm_block(*pooled["gaze"]),
        "",
        "## how to read this",
        "- majority is the number a useless model gets for free.",
        "- gaze-only tells you whether the 5-d vector is linearly related",
        "  to the label after subject averaging and z-scoring.",
        "- text-stats is a weak stand-in for BERT; beating it is not the",
        "  same as beating roberta-base.",
        "- late-fusion here is concat + softmax. The real model is concat",
        "  of 768-d pooler_output + Linear(5,16), then CrossEntropy.",
        "- If fusion ≈ max(gaze, text-stats), the signals overlap. If it",
        "  is higher, they are at least partly complementary *in this",
        "  linear toy*. Confirm with model_ZuCo_SST.py before claiming it.",
        "",
    ]
    text_out = "\n".join(lines)
    print(text_out)
    write_text("06_toy_late_fusion.txt", text_out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
