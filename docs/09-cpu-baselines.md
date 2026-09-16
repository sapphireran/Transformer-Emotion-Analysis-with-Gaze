# CPU baselines

`examples/08_cpu_baselines.py` answers a smaller question than the
transformers: *on these exact tables, does a linear gaze vector move a
hashed-text ridge?*

Recipe, locked in `gazebook.baselines`:

- 5 stratified folds, seed 42 (not sklearn's `StratifiedKFold` shuffle,
  but deterministic and class-balanced).
- Ridge one-vs-rest, L2 = 1, intercept not regularized.
- Hashed bag-of-words, 128 signed features, md5 (stable across
  `PYTHONHASHSEED`).
- Gaze columns: the same five names `EyeTrackingModel` reads.
- “gaze ⟂ length” residualizes each channel on whitespace word count.
- “shuffled gaze” permutes gaze rows as a negative control.

Run it after any CSV edit:

```bash
python3 examples/08_cpu_baselines.py
```

The snapshot from the first green run on this branch lives in
`examples/output/08_cpu_baselines.md` and is copied into
`docs/sample-results.md` once `examples/run_all.py` finishes.

## How to read the numbers

- **Majority** on ZuCo is 0.35 (140 / 400 positives). Full SST majority
  is ~0.42 (positive is the mode).
- **Gaze only** near majority means the five recorded channels are a
  weak linear sentiment feature. That matches the 0.03–0.07 label
  correlations.
- **Hashed text** should beat both. If it does not, the hash dimension
  or the ridge collapsed — fail the example.
- **Text + gaze** vs **text + shuffled gaze** is the cheap ablation.
  A real fusion gain should shrink when gaze is shuffled. A gain that
  survives shuffling is fold noise or a length leak in the concat.

These floors exist so a later RoBERTa number has something honest to
beat, and so projected-SST “fusion helps by 0.004 F1” claims can be
compared to a linear control that already sees the collinear five-pack.
