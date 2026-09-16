"""Late-fusion softmax used by the CPU demo.

This is the architecture in docs/architecture.md with the transformer
replaced by a hashed bag-of-words vector. It exists so the concat +
linear head can be trained without downloading weights.
"""

from __future__ import annotations

import hashlib
import math
import random
import re
from dataclasses import dataclass, field
from typing import Iterable, List, Optional, Sequence, Tuple

from . import linalg, metrics, stats

TOKEN_RE = re.compile(r"[A-Za-z]+")


def tokenize(text: str) -> List[str]:
    return [tok.lower() for tok in TOKEN_RE.findall(text or "")]


def hash_embed(text: str, dim: int = 32) -> List[float]:
    """Stable hashed bag-of-words. Do not use builtin hash() — it salts."""
    vec = [0.0] * dim
    for token in tokenize(text):
        digest = hashlib.md5(token.encode("utf-8")).hexdigest()
        idx = int(digest, 16) % dim
        vec[idx] += 1.0
    return linalg.normalize(vec)


def embed_texts(texts: Sequence[str], dim: int = 32) -> List[List[float]]:
    return [hash_embed(text, dim=dim) for text in texts]


@dataclass
class Linear:
    weight: List[List[float]]
    bias: List[float]

    @classmethod
    def xavier(cls, out_features: int, in_features: int, rng: random.Random) -> "Linear":
        return cls(
            weight=linalg.xavier_matrix(out_features, in_features, rng),
            bias=[0.0] * out_features,
        )

    def forward(self, vector: Sequence[float]) -> List[float]:
        return linalg.add_vec(linalg.matvec(self.weight, vector), self.bias)

    def step(self, grad_weight: List[List[float]], grad_bias: List[float], lr: float) -> None:
        rows, cols = linalg.shape(self.weight)
        for i in range(rows):
            self.bias[i] -= lr * grad_bias[i]
            for j in range(cols):
                self.weight[i][j] -= lr * grad_weight[i][j]


@dataclass
class SoftmaxHead:
    """Single linear layer + softmax. Used for text-only and gaze-only."""

    layer: Linear
    lr: float = 0.15

    @classmethod
    def create(cls, in_features: int, num_labels: int, rng: random.Random, lr: float = 0.15) -> "SoftmaxHead":
        return cls(layer=Linear.xavier(num_labels, in_features, rng), lr=lr)

    def logits(self, vector: Sequence[float]) -> List[float]:
        return self.layer.forward(vector)

    def predict(self, vector: Sequence[float]) -> int:
        return linalg.argmax(self.logits(vector))

    def train_step(self, vector: Sequence[float], label: int) -> float:
        logits = self.logits(vector)
        if not linalg.finite(logits):
            return float("nan")
        probs = linalg.softmax(logits)
        loss = _nll(probs, label)
        error = linalg.sub_vec(probs, linalg.one_hot(label, len(probs)))
        grad_w = [[error[i] * vector[j] for j in range(len(vector))] for i in range(len(error))]
        self.layer.step(linalg.clip_matrix(grad_w), linalg.clip_vec(error), self.lr)
        return loss


@dataclass
class FusionHead:
    """Linear(gaze) -> concat(text, gaze_hidden) -> Linear -> softmax.

    Matches EyeTrackingModel's late-fusion layout, just with smaller widths.
    """

    gaze_layer: Linear
    classifier: Linear
    lr: float = 0.12
    gaze_out: int = 16

    @classmethod
    def create(
        cls,
        text_dim: int,
        gaze_in: int,
        gaze_out: int,
        num_labels: int,
        rng: random.Random,
        lr: float = 0.12,
    ) -> "FusionHead":
        return cls(
            gaze_layer=Linear.xavier(gaze_out, gaze_in, rng),
            classifier=Linear.xavier(num_labels, text_dim + gaze_out, rng),
            lr=lr,
            gaze_out=gaze_out,
        )

    def hidden(self, text: Sequence[float], gaze: Sequence[float]) -> List[float]:
        gaze_h = self.gaze_layer.forward(gaze)
        return linalg.concat_vec((text, gaze_h))

    def logits(self, text: Sequence[float], gaze: Sequence[float]) -> List[float]:
        return self.classifier.forward(self.hidden(text, gaze))

    def predict(self, text: Sequence[float], gaze: Sequence[float]) -> int:
        return linalg.argmax(self.logits(text, gaze))

    def train_step(self, text: Sequence[float], gaze: Sequence[float], label: int) -> float:
        gaze_h = self.gaze_layer.forward(gaze)
        hidden = linalg.concat_vec((text, gaze_h))
        logits = self.classifier.forward(hidden)
        if not linalg.finite(logits):
            return float("nan")
        probs = linalg.softmax(logits)
        loss = _nll(probs, label)
        d_logits = linalg.sub_vec(probs, linalg.one_hot(label, len(probs)))

        # Backprop through the current classifier weights, then step both layers.
        w_t = linalg.transpose(self.classifier.weight)
        d_hidden = linalg.matvec(w_t, d_logits)
        d_gaze_h = d_hidden[len(text) :]

        grad_cw = [
            [d_logits[i] * hidden[j] for j in range(len(hidden))]
            for i in range(len(d_logits))
        ]
        grad_gw = [
            [d_gaze_h[i] * gaze[j] for j in range(len(gaze))]
            for i in range(len(d_gaze_h))
        ]
        self.classifier.step(
            linalg.clip_matrix(grad_cw), linalg.clip_vec(d_logits), self.lr
        )
        self.gaze_layer.step(
            linalg.clip_matrix(grad_gw), linalg.clip_vec(d_gaze_h), self.lr
        )
        return loss


@dataclass
class TrainResult:
    name: str
    train_acc: float
    holdout_acc: float
    holdout_macro_f1: float
    holdout_report: dict
    losses: List[float] = field(default_factory=list)


def _nll(probs: Sequence[float], label: int) -> float:
    p = min(max(probs[label], 1e-12), 1.0)
    return -math.log(p)


def stratified_split(
    items: Sequence[int],
    labels: Sequence[int],
    *,
    test_ratio: float = 0.25,
    seed: int = 42,
) -> Tuple[List[int], List[int]]:
    if len(items) != len(labels):
        raise ValueError("stratified_split length mismatch")
    rng = random.Random(seed)
    buckets: dict[int, List[int]] = {}
    for idx, label in zip(items, labels):
        buckets.setdefault(label, []).append(idx)
    train: List[int] = []
    test: List[int] = []
    for label in sorted(buckets):
        bucket = list(buckets[label])
        rng.shuffle(bucket)
        n_test = max(1, int(round(len(bucket) * test_ratio))) if len(bucket) > 1 else 1
        n_test = min(n_test, len(bucket) - 1) if len(bucket) > 1 else 0
        test.extend(bucket[:n_test])
        train.extend(bucket[n_test:])
    rng.shuffle(train)
    rng.shuffle(test)
    return train, test


def iterate_minibatches(
    indices: Sequence[int], batch_size: int, rng: random.Random
) -> Iterable[List[int]]:
    order = list(indices)
    rng.shuffle(order)
    for start in range(0, len(order), batch_size):
        yield order[start : start + batch_size]


def train_softmax_head(
    name: str,
    features: Sequence[Sequence[float]],
    labels: Sequence[int],
    train_idx: Sequence[int],
    test_idx: Sequence[int],
    *,
    epochs: int = 40,
    lr: float = 0.15,
    seed: int = 0,
) -> TrainResult:
    rng = random.Random(seed)
    width = len(features[0]) if features else 0
    n_labels = max(labels) + 1 if labels else 0
    model = SoftmaxHead.create(width, n_labels, rng, lr=lr)
    losses: List[float] = []
    for _ in range(epochs):
        epoch_loss = 0.0
        order = list(train_idx)
        rng.shuffle(order)
        for idx in order:
            epoch_loss += model.train_step(features[idx], labels[idx])
        losses.append(epoch_loss / max(len(order), 1))
    return _pack_result(name, model, features, None, labels, train_idx, test_idx, losses)


def train_fusion_head(
    name: str,
    text: Sequence[Sequence[float]],
    gaze: Sequence[Sequence[float]],
    labels: Sequence[int],
    train_idx: Sequence[int],
    test_idx: Sequence[int],
    *,
    epochs: int = 40,
    lr: float = 0.12,
    gaze_out: int = 16,
    seed: int = 0,
) -> TrainResult:
    rng = random.Random(seed)
    model = FusionHead.create(
        text_dim=len(text[0]),
        gaze_in=len(gaze[0]),
        gaze_out=gaze_out,
        num_labels=max(labels) + 1,
        rng=rng,
        lr=lr,
    )
    losses: List[float] = []
    for _ in range(epochs):
        epoch_loss = 0.0
        order = list(train_idx)
        rng.shuffle(order)
        for idx in order:
            epoch_loss += model.train_step(text[idx], gaze[idx], labels[idx])
        losses.append(epoch_loss / max(len(order), 1))
    return _pack_result(name, model, text, gaze, labels, train_idx, test_idx, losses)


def _pack_result(
    name: str,
    model: object,
    primary: Sequence[Sequence[float]],
    gaze: Optional[Sequence[Sequence[float]]],
    labels: Sequence[int],
    train_idx: Sequence[int],
    test_idx: Sequence[int],
    losses: List[float],
) -> TrainResult:
    def predict_one(idx: int) -> int:
        if gaze is None:
            return model.predict(primary[idx])  # type: ignore[attr-defined]
        return model.predict(primary[idx], gaze[idx])  # type: ignore[attr-defined]

    y_train = [labels[i] for i in train_idx]
    p_train = [predict_one(i) for i in train_idx]
    y_test = [labels[i] for i in test_idx]
    p_test = [predict_one(i) for i in test_idx]
    holdout = metrics.report(y_test, p_test)
    return TrainResult(
        name=name,
        train_acc=metrics.accuracy(y_train, p_train),
        holdout_acc=float(holdout["accuracy"]),
        holdout_macro_f1=float(holdout["macro"]["f1"]),  # type: ignore[index]
        holdout_report=holdout,
        losses=losses,
    )


def shuffle_rows(matrix: Sequence[Sequence[float]], seed: int = 7) -> List[List[float]]:
    """Row-wise permutation used as a gaze ablation."""
    rng = random.Random(seed)
    order = list(range(len(matrix)))
    rng.shuffle(order)
    return [list(matrix[i]) for i in order]


def majority_predict(train_labels: Sequence[int], n: int) -> List[int]:
    winner, _ = stats.majority_baseline(list(train_labels))
    return [winner] * n
