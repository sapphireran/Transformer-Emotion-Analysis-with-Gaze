# Sidecar fusion walkthrough

This is the geometry of `EyeTrackingModel` without downloading RoBERTa.
Pooler vectors below are random N(0, 0.02) stand-ins so the example stays
CPU-only and offline. Gaze rows are the first five *real* full-SST train
examples.

```
gaze  (B, 5)  --Linear 5→16, no activation-->  eye (B, 16)
pooler(B, 768) ----------------------------\
                                           concat (B, 784)
                                           Dropout p=0.1
                                           Linear 784→3 logits
```

- concat dim: **784** (expected 784)
- W_eye shape: (16, 5)
- W_cls shape: (3, 784)
- mean |Δlogit| when dropout is on vs off (same seed): 0.066753

The training scripts never apply ReLU/GELU on the gaze branch. The sidecar
is a linear basis expansion of five already-collinear scalars.

| row | nFix | GD | eye_l2 | concat_dim | p_neg | p_neu | p_pos | pred | true_label |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 0.7665 | 0.9453 | 5.1832 | 784 | 0.2090 | 0.3479 | 0.4431 | 2 | 0 |
| 1 | 0.0889 | 0.1414 | 0.6699 | 784 | 0.3192 | 0.3332 | 0.3476 | 2 | 2 |
| 2 | 1.5396 | 1.2862 | 9.6949 | 784 | 0.1381 | 0.3508 | 0.5111 | 2 | 2 |
| 3 | 0.7819 | 0.9474 | 5.1151 | 784 | 0.2254 | 0.3378 | 0.4368 | 2 | 2 |
| 4 | 0.7786 | 0.9643 | 5.1316 | 784 | 0.2292 | 0.3448 | 0.4260 | 2 | 0 |

Random poolers plus untrained sidecar weights are **not** a sentiment
model. The table exists to pin shapes and to show how little of the 784-d
vector is gaze (16 / 784 ≈ 2.0%). Any claim that 'the model uses eye
tracking' has to survive an ablation that zeros or shuffles those 16
coordinates.
