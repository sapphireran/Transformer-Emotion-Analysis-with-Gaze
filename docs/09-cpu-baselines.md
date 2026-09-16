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
- **Gaze only** on the 400 recorded-gaze rows beats majority (~0.41 vs
  0.35). Single-channel Pearson r vs the label is only 0.03–0.07; the
  five-d ridge still finds a weak joint signal. Residualizing length
  does not remove it.
- **Hashed text (128-d)** is a deliberately cheap stand-in, not TF-IDF.
  On 400 short reviews it loses to gaze. On 11.8k SST rows it reaches
  ~0.50 accuracy and finally dominates.
- **Text + gaze** on full SST adds a few hundredths of a point at most.
  That is what a rank-1 projected five-pack looks like when concatenated
  with a text vector. A jump larger than ~2 points is treated as a
  contract failure (possible leakage).
- **Text + shuffled gaze** is the negative control. If shuffle beats
  real gaze, the concat “gain” was fold noise.

These floors exist so a later RoBERTa number has something honest to
beat, and so projected-SST “fusion helps by 0.004 F1” claims can be
compared to a linear control that already sees the collinear five-pack.
