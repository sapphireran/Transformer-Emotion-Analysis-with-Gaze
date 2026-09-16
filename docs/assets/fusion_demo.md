# Fusion forward-pass demo

NumPy clone of `EyeTrackingModel` (random pooled vectors, **real** gaze rows).

- concat width = 784 (768 + 16)
- mean |Δ logit| vs zero gaze = [0.0747, 0.2639, 0.0261]
- pairs of ZuCo gaze vectors that flip argmax on a fixed pooled vector: 6
- train dropout diverges across RNG: True
- eval forward is deterministic: True
- cross-entropy on this random-pooled batch (not a trained score): 1.1252

Pooled text is Gaussian noise so class probabilities are not meaningful as
sentiment accuracy. The demo only shows that the gaze branch is wired the same
way as the PyTorch module: linear 5→16, concat, dropout, linear 784→3.
