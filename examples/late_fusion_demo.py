#!/usr/bin/env python3
"""Tiny late-fusion model that mirrors EyeTrackingModel without transformers.

Architecture (see docs/model-architecture.md):

    hashed bag-of-words  (B, 64)     gaze features (B, 5)
             │                              │
             │                         Linear(5 → 16)
             │                         ReLU          # extra vs original
             │                              │
             └────────── concat ────────────┘
                          (B, 80)
                            │
                       Linear(80 → 3)
                            │
                         softmax

The original EyeTrackingModel concatenates BERT/RoBERTa ``pooler_output``
(768-d) with a *linear* 5→16 map and has no ReLU. This demo swaps the
encoder for a stable hashed unigram vector so it runs in a few seconds
on CPU with only numpy.

Three heads are trained on the same folds so you can see the
contribution of each branch:

* text-only   — BOW → Linear → 3
* gaze-only   — ReLU(Linear(5→16)) → Linear → 3
* fused       — concat as above
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

_EXAMPLES = Path(__file__).resolve().parent
if str(_EXAMPLES) not in sys.path:
    sys.path.insert(0, str(_EXAMPLES))

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from paths import ZUCO_ET_COLS, ZUCO_STANDARD

BOW_DIM = 64
GAZE_HIDDEN = 16
N_CLASSES = 3
N_EPOCHS = 80
LR = 0.15
SEED = 42


def _token_index(token: str, dim: int) -> int:
    digest = hashlib.md5(token.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "little") % dim


def _tokenize(sentence: str) -> list[str]:
    cleaned = []
    buf: list[str] = []
    for ch in sentence.lower():
        if ch.isalnum():
            buf.append(ch)
        else:
            if buf:
                cleaned.append("".join(buf))
                buf = []
    if buf:
        cleaned.append("".join(buf))
    return cleaned


def hashed_bow(sentences: list[str], dim: int = BOW_DIM) -> np.ndarray:
    """Normalized hashed unigrams. md5 so the features are process-stable."""
    out = np.zeros((len(sentences), dim), dtype=np.float64)
    for i, sentence in enumerate(sentences):
        for tok in _tokenize(sentence):
            out[i, _token_index(tok, dim)] += 1.0
        z = out[i].sum()
        if z > 0:
            out[i] /= z
    return out


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


def _relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(x, 0.0)


def _one_hot(y: np.ndarray, n: int = N_CLASSES) -> np.ndarray:
    eye = np.eye(n, dtype=np.float64)
    return eye[y]


class SoftmaxHead:
    """Linear classifier with optional left-side features already built."""

    def __init__(self, in_dim: int, rng: np.random.RandomState) -> None:
        # Xavier-ish scale keeps the first softmax from saturating.
        scale = 1.0 / np.sqrt(in_dim)
        self.w = rng.randn(in_dim, N_CLASSES) * scale
        self.b = np.zeros(N_CLASSES, dtype=np.float64)

    def logits(self, h: np.ndarray) -> np.ndarray:
        return h @ self.w + self.b

    def step(self, h: np.ndarray, y: np.ndarray, lr: float) -> float:
        probs = _softmax(self.logits(h))
        target = _one_hot(y)
        n = max(len(y), 1)
        loss = float(-np.sum(target * np.log(probs + 1e-12)) / n)
        grad = (probs - target) / n
        self.w -= lr * (h.T @ grad)
        self.b -= lr * grad.sum(axis=0)
        return loss


class GazeMap:
    """The 5→16 branch. ReLU is the only deliberate departure from the paper code."""

    def __init__(self, rng: np.random.RandomState) -> None:
        scale = 1.0 / np.sqrt(5)
        self.w = rng.randn(5, GAZE_HIDDEN) * scale
        self.b = np.zeros(GAZE_HIDDEN, dtype=np.float64)

    def forward(self, gaze: np.ndarray) -> np.ndarray:
        return _relu(gaze @ self.w + self.b)

    def step(self, gaze: np.ndarray, upstream: np.ndarray, lr: float) -> None:
        pre = gaze @ self.w + self.b
        relu_mask = (pre > 0).astype(np.float64)
        grad_pre = upstream * relu_mask
        n = max(len(gaze), 1)
        self.w -= lr * (gaze.T @ grad_pre) / n
        self.b -= lr * grad_pre.sum(axis=0) / n


def _predict(logits: np.ndarray) -> np.ndarray:
    return np.argmax(logits, axis=1)


def _acc(y: np.ndarray, pred: np.ndarray) -> float:
    return float((y == pred).mean())


def train_text_only(
    x_text: np.ndarray, y: np.ndarray, rng: np.random.RandomState
) -> SoftmaxHead:
    head = SoftmaxHead(x_text.shape[1], rng)
    for _ in range(N_EPOCHS):
        head.step(x_text, y, LR)
    return head


def train_gaze_only(
    x_gaze: np.ndarray, y: np.ndarray, rng: np.random.RandomState
) -> tuple[GazeMap, SoftmaxHead]:
    gmap = GazeMap(rng)
    head = SoftmaxHead(GAZE_HIDDEN, rng)
    for _ in range(N_EPOCHS):
        hidden = gmap.forward(x_gaze)
        probs = _softmax(head.logits(hidden))
        n = max(len(y), 1)
        grad_logits = (probs - _one_hot(y)) / n
        # dL/dH = dL/dlogits @ W^T
        upstream = grad_logits @ head.w.T
        head.step(hidden, y, LR)
        gmap.step(x_gaze, upstream * n, LR)  # gmap.step re-divides by n
    return gmap, head


def train_fused(
    x_text: np.ndarray,
    x_gaze: np.ndarray,
    y: np.ndarray,
    rng: np.random.RandomState,
) -> tuple[GazeMap, SoftmaxHead]:
    gmap = GazeMap(rng)
    head = SoftmaxHead(x_text.shape[1] + GAZE_HIDDEN, rng)
    for _ in range(N_EPOCHS):
        hidden_g = gmap.forward(x_gaze)
        fused = np.concatenate([x_text, hidden_g], axis=1)
        probs = _softmax(head.logits(fused))
        n = max(len(y), 1)
        grad_logits = (probs - _one_hot(y)) / n
        upstream_fused = grad_logits @ head.w.T
        upstream_gaze = upstream_fused[:, x_text.shape[1] :]
        head.step(fused, y, LR)
        gmap.step(x_gaze, upstream_gaze * n, LR)
    return gmap, head


def evaluate_fold(
    x_text: np.ndarray,
    x_gaze: np.ndarray,
    y: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    rng: np.random.RandomState,
) -> dict[str, float]:
    scores: dict[str, float] = {}

    text_head = train_text_only(x_text[train_idx], y[train_idx], rng)
    scores["text"] = _acc(y[test_idx], _predict(text_head.logits(x_text[test_idx])))

    gmap, ghead = train_gaze_only(x_gaze[train_idx], y[train_idx], rng)
    scores["gaze"] = _acc(
        y[test_idx], _predict(ghead.logits(gmap.forward(x_gaze[test_idx])))
    )

    fmap, fhead = train_fused(x_text[train_idx], x_gaze[train_idx], y[train_idx], rng)
    fused_test = np.concatenate(
        [x_text[test_idx], fmap.forward(x_gaze[test_idx])], axis=1
    )
    scores["fused"] = _acc(y[test_idx], _predict(fhead.logits(fused_test)))
    return scores


def main() -> int:
    df = pd.read_csv(ZUCO_STANDARD)
    sentences = df["sentence"].astype(str).tolist()
    x_text = hashed_bow(sentences, BOW_DIM)
    x_gaze = df[ZUCO_ET_COLS].to_numpy(dtype=np.float64)
    # z-scored already, but re-center the train fold implicitly via the
    # linear layer; keep the committed scaling so we match model_ZuCo_SST.
    y = df["sentiment_label"].to_numpy(dtype=np.int64)

    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    rng = np.random.RandomState(SEED)
    bag: dict[str, list[float]] = {"text": [], "gaze": [], "fused": []}
    majority = float(np.bincount(y).max() / len(y))

    print(
        f"ZuCo ∩ SST late-fusion toy  (n={len(y)}, bow_dim={BOW_DIM}, "
        f"gaze_hidden={GAZE_HIDDEN}, epochs={N_EPOCHS}, lr={LR})"
    )
    print(f"majority-class floor: {majority:.4f}")
    print()

    for fold, (train_idx, test_idx) in enumerate(kf.split(x_text, y), start=1):
        scores = evaluate_fold(x_text, x_gaze, y, train_idx, test_idx, rng)
        for key, value in scores.items():
            bag[key].append(value)
        print(
            f"  fold {fold}:  text={scores['text']:.4f}  "
            f"gaze={scores['gaze']:.4f}  fused={scores['fused']:.4f}"
        )

    print()
    for key in ("text", "gaze", "fused"):
        arr = np.array(bag[key])
        print(f"  mean {key:<5} acc={arr.mean():.4f} ± {arr.std():.4f}")
    print()
    print(
        "This is a hashed-unigram model, not RoBERTa. A fused win here "
        "only means the 16-d gaze map added something the 64-d BOW "
        "missed on this 400-row table. Compare with "
        "examples/gaze_only_baseline.py for a linear ET-only number."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
