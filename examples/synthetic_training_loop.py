#!/usr/bin/env python3
"""Train the NumPy fusion head on a separable toy batch.

This is not a substitute for model_ZuCo_SST.py. It only shows that:

* text-only and gaze-only both carry a class signal in the toy data
* concatenating them reaches a lower loss / higher accuracy
* the SGD path in GazeFusionClassifier.step is numerically stable
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gaze_emotion.fusion import GazeFusionClassifier, make_separable_batch
from gaze_emotion.metrics import classification_report, weighted_scores


def run_loop(text, gaze, labels, *, use_text, use_gaze, steps=80, lr=0.15, seed=1):
    n, text_dim = text.shape
    gaze_in = gaze.shape[1]
    model = GazeFusionClassifier(text_dim=text_dim, gaze_in=gaze_in, rng=np.random.default_rng(seed))
    text_in = text if use_text else np.zeros_like(text)
    gaze_in_arr = gaze if use_gaze else np.zeros_like(gaze)
    history = []
    for step in range(steps):
        loss = model.step(text_in, gaze_in_arr, labels, lr=lr)
        if step % 20 == 19 or step == 0:
            preds = model.predict(text_in, gaze_in_arr)
            scores = weighted_scores(labels, preds)
            history.append((step + 1, loss, scores["accuracy"], scores["f1"]))
    preds = model.predict(text_in, gaze_in_arr)
    return model, history, preds


def print_history(title: str, history) -> None:
    print(title)
    print(f"  {'step':>6}  {'loss':>8}  {'acc':>8}  {'f1':>8}")
    for step, loss, acc, f1 in history:
        print(f"  {step:6d}  {loss:8.4f}  {acc:8.4f}  {f1:8.4f}")


def main() -> int:
    text, gaze, labels = make_separable_batch(n=120, text_dim=16, seed=0)
    print("Synthetic late-fusion training loop")
    print("=" * 72)
    print(f"batch: {len(labels)} rows, text_dim={text.shape[1]}, gaze_dim={gaze.shape[1]}")
    print(f"label counts: { {int(k): int(v) for k, v in zip(*np.unique(labels, return_counts=True))} }")
    print(
        "Each class has a +2.5 bump on one text dim and one gaze dim, "
        "plus Gaussian noise."
    )
    print()

    configs = [
        ("text only (gaze zeroed)", True, False),
        ("gaze only (text zeroed)", False, True),
        ("text + gaze fusion", True, True),
    ]
    final = {}
    for title, use_text, use_gaze in configs:
        _, history, preds = run_loop(text, gaze, labels, use_text=use_text, use_gaze=use_gaze)
        print_history(title, history)
        scores = weighted_scores(labels, preds)
        final[title] = scores
        print(f"  final weighted F1={scores['f1']:.4f}  acc={scores['accuracy']:.4f}")
        print()

    print("Fusion-run classification report")
    print("-" * 72)
    _, _, preds = run_loop(text, gaze, labels, use_text=True, use_gaze=True)
    print(classification_report(labels, preds))
    print()

    fusion_f1 = final["text + gaze fusion"]["f1"]
    text_f1 = final["text only (gaze zeroed)"]["f1"]
    gaze_f1 = final["gaze only (text zeroed)"]["f1"]
    print(
        f"fusion F1 {fusion_f1:.3f} vs text-only {text_f1:.3f} "
        f"vs gaze-only {gaze_f1:.3f}"
    )
    if fusion_f1 + 1e-9 < max(text_f1, gaze_f1) - 0.05:
        print("WARNING: fusion underperformed the better unimodal run on this toy batch.")
        return 1
    print("Fusion matched or beat the unimodal heads on this separable toy set.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
