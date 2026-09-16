"""A tiny late-fusion forward pass that mirrors ``EyeTrackingModel``.

The real model is:

    pooler (hidden=768) ──┐
                          ├─ concat ─ dropout ─ Linear → 3 logits
    Linear(5 → 16) gaze ──┘

This demo keeps the same shapes and the same concat story, but replaces the
transformer with a hashed bag-of-words vector so it runs on CPU in a second
and does not download weights.

It is a *shape and wiring* demo, not a claim about RoBERTa accuracy.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from . import numbers

GAZE_DIM = 5
GAZE_HIDDEN = 16
TEXT_DIM = 32
N_LABELS = 3
COMBINED_DIM = TEXT_DIM + GAZE_HIDDEN  # 48, not 784 — same recipe, toy width


@dataclass
class LateFusionToy:
    text_proj: list[list[float]]
    gaze_layer: list[list[float]]
    gaze_bias: list[float]
    classifier: list[list[float]]
    classifier_bias: list[float]
    text_dim: int = TEXT_DIM
    gaze_dim: int = GAZE_DIM
    gaze_hidden: int = GAZE_HIDDEN
    n_labels: int = N_LABELS

    @classmethod
    def seeded(cls, seed: int = 7) -> "LateFusionToy":
        rng = random.Random(seed)
        # He-ish-small init so softmax is not saturated on random inputs.
        return cls(
            text_proj=numbers.seeded_matrix(rng, TEXT_DIM, TEXT_DIM, -0.15, 0.15),
            gaze_layer=numbers.seeded_matrix(rng, GAZE_HIDDEN, GAZE_DIM, -0.25, 0.25),
            gaze_bias=numbers.seeded_uniform(rng, GAZE_HIDDEN, -0.05, 0.05),
            classifier=numbers.seeded_matrix(rng, N_LABELS, COMBINED_DIM, -0.2, 0.2),
            classifier_bias=numbers.seeded_uniform(rng, N_LABELS, -0.05, 0.05),
        )

    def encode_text(self, tokens: list[str]) -> list[float]:
        hashed = [0.0] * self.text_dim
        if not tokens:
            tokens = ["<empty>"]
        for token in tokens:
            hashed[hash_token(token, self.text_dim)] += 1.0
        norm = max(1.0, sum(abs(value) for value in hashed))
        hashed = [value / norm for value in hashed]
        return numbers.matvec(self.text_proj, hashed)

    def encode_gaze(self, gaze: list[float]) -> list[float]:
        if len(gaze) != self.gaze_dim:
            raise ValueError(f"expected {self.gaze_dim} gaze features, got {len(gaze)}")
        return numbers.add(numbers.matvec(self.gaze_layer, gaze), self.gaze_bias)

    def logits(self, tokens: list[str], gaze: list[float]) -> list[float]:
        combined = numbers.concat(self.encode_text(tokens), self.encode_gaze(gaze))
        return numbers.add(numbers.matvec(self.classifier, combined), self.classifier_bias)

    def predict(self, tokens: list[str], gaze: list[float]) -> tuple[int, list[float], list[float]]:
        raw = self.logits(tokens, gaze)
        probs = numbers.softmax(raw)
        return numbers.argmax(probs), probs, raw


def hash_token(token: str, dim: int) -> int:
    # Deterministic 32-bit FNV-1a so the demo is stable across processes.
    h = 2166136261
    for char in token.lower():
        h ^= ord(char)
        h = (h * 16777619) & 0xFFFFFFFF
    return h % dim


def tokenize(sentence: str) -> list[str]:
    out: list[str] = []
    buf: list[str] = []
    for char in sentence:
        if char.isalnum() or char == "'":
            buf.append(char)
        else:
            if buf:
                out.append("".join(buf))
                buf = []
    if buf:
        out.append("".join(buf))
    return out


def architecture_lines() -> list[str]:
    return [
        "text tokens ─► hash bag-of-words (32) ─► Linear(32→32) ─┐",
        "                                                       ├─► concat (48) ─► Linear(48→3) ─► softmax",
        "gaze (5) ─► Linear(5→16) ──────────────────────────────┘",
        "",
        "Training scripts use the same concat, with 768-d pooler + 16-d gaze → 784.",
    ]
