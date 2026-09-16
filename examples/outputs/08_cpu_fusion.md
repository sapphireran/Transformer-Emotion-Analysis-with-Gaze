# CPU late-fusion wiring demo

```
text tokens ─► hash bag-of-words (32) ─► Linear(32→32) ─┐
                                                       ├─► concat (48) ─► Linear(48→3) ─► softmax
gaze (5) ─► Linear(5→16) ──────────────────────────────┘

Training scripts use the same concat, with 768-d pooler + 16-d gaze → 784.
```

Weights are seeded and tiny. Predictions are not an accuracy claim. This only shows that a 5-d gaze vector and a hashed sentence can share a concatenated classifier, the same way ``EyeTrackingModel`` concatenates ``pooler_output`` with ``Linear(5 → 16)``.

## First 6 ZuCo rows, seed=7

| id | gold | pred | p_neg | p_neu | p_pos | text |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | 1 | 1 | 0.2768 | 0.3663 | 0.3569 | Presents a good case while failing to provide a reason for us to… |
| 1 | 2 | 2 | 0.3157 | 0.2978 | 0.3865 | Beautifully crafted, engaging filmmaking that should attract ups… |
| 2 | 0 | 2 | 0.3188 | 0.3292 | 0.3520 | Bread, My Sweet has so many flaws it would be easy for critics t… |
| 3 | 1 | 2 | 0.2187 | 0.2021 | 0.5792 | Slow, silly and unintentionally hilarious. |
| 4 | 0 | 2 | 0.1670 | 0.3941 | 0.4390 | Ultimately feels empty and unsatisfying, like swallowing a Commu… |
| 5 | 2 | 2 | 0.2860 | 0.3181 | 0.3959 | Exudes the fizz of a Busby Berkeley musical and the visceral exc… |
