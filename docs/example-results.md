# Example script results

Numbers below come from `python examples/run_all.py` on the committed
CSVs (pandas 3 / scikit-learn 1.9, CPU). They are **not** RoBERTa
scores. Re-run the scripts if you change a CSV; this page is a snapshot
so you can tell whether your machine matches.

## Integrity

`examples/data_integrity.py` reported **82/82 checks passed**:

- Full-SST 80/10/10 is a partition of `combined_full_sst_et.csv`
  (9,482 + 1,185 + 1,186 = 11,853 unique `sentence_id`s).
- ZuCo 80/10/10 is a partition of `combined_sst_et_standard.csv`
  (320 + 40 + 40 = 400).
- `ssts_ZuCo.csv` sentences and ids match the standard join exactly.
- Subject `3_SR.csv` is the only short file (299 sentence rows, 5,293
  word rows). Everyone else is 400 / 7,129.
- No NaNs in the files the training scripts read.

## Gaze vs. polarity (linear)

Pearson *r* of each 5-d feature against the integer 0/1/2 label:

| Feature | ZuCo recorded (n=400) | Full SST train, projected (n=9,482) |
|---|---:|---:|
| nFix / nFixations | +0.033 | −0.066 |
| FFD | +0.072 | −0.065 |
| GPT | +0.030 | −0.065 |
| TRT | +0.051 | −0.062 |
| GD | +0.036 | −0.070 |

Two structural facts jump out of the pairwise tables:

1. **Recorded ZuCo gaze is internally consistent but not redundant.**
   `TRT`↔`nFixations` is 0.96 and `GPT`↔`nFixations` is 0.91, which is
   expected (more fixations cost more time). `FFD` is the most
   independent of the five (*r* ≈ 0.42 with nFixations) and also the
   strongest linear correlate of polarity.
2. **Projected full-SST gaze is almost rank-1.** `nFix`/`FFD`/`GPT`/`TRT`
   all correlate at ≥ 0.986. Whatever model produced those columns
   essentially emitted one latent “reading time” direction five times.
   A 5-d late-fusion branch on full SST is therefore much less
   informative than the same branch on ZuCo.

## Gaze-only logistic baseline

`examples/gaze_only_baseline.py` — `StandardScaler` + `LogisticRegression`
vs `DummyClassifier(strategy="most_frequent")`.

| Setting | Majority acc | Logreg acc | Δ acc | Logreg weighted-F1 |
|---|---:|---:|---:|---:|
| ZuCo ∩ SST, 5-fold, seed 42 | 0.3500 | 0.3675 ± 0.030 | **+0.0175** | 0.3535 |
| Full SST train → test | 0.4418 | 0.4368 | **−0.0050** | 0.3870 |

On ZuCo the linear ET probe barely beats the 35% majority floor and
lifts weighted F1 more than accuracy (0.18 → 0.35), which is the
unbalanced-class pattern: majority always predicts positive (140/400),
logreg at least uses the other two labels.

On full SST the same probe **loses to majority on accuracy** and only
wins on weighted F1. Combined with the near-duplicate projected
features, this is a weak prior for `roberta_eye_tracking` on that
split. If a transformer + gaze run looks like a big win there, check
[known-quirks.md](known-quirks.md) item 1 (test metrics are last-batch
only) before trusting it.

Mean |coef| on the full-SST logreg (after scaling) ranks
`nFix` (0.47) > `GPT` (0.43) > `FFD` (0.22) > `GD` (0.11) > `TRT` (0.08).
Because those columns are collinear, the ranking is unstable; do not
over-interpret it.

## Hashed-BOW late-fusion toy

`examples/late_fusion_demo.py` — 64-d md5-hashed unigrams, ReLU
Linear(5→16), 80 epochs of numpy SGD, same 5 ZuCo folds.

| Branch | Mean acc ± std |
|---|---|
| Majority floor | 0.3500 |
| Text only | 0.3550 ± 0.023 |
| Gaze only | 0.3550 ± 0.037 |
| Fused | **0.3850 ± 0.044** |

The 64-d bag-of-words is a weak text encoder (it sits on the majority
floor). Fusion still picked up about +3 points, which is consistent
with the small logreg lift: there is *some* residual in the 5-d ZuCo
vector, and a concat head can use it. This is **not** a substitute for
`model_ZuCo_SST.py`.

Per-fold fused scores ranged from 0.3375 to 0.4500 — 400 rows and a
tiny encoder, so treat the mean as a smoke test that the demo trains,
not as a published result.

## Sentence walkthrough

`examples/sentence_walkthrough.py` picks one mid-length sentence per
class. On this checkout those were:

| `sentence_id` | Label | Sentence (truncated) |
|---|---|---|
| 134 | 0 negative | *Festers in just such a dungpile…* |
| 10 | 1 neutral | *Frida is certainly no disaster…* |
| 184 | 2 positive | *A colorful, vibrant introduction…* |

Word-level rows come from `word_averages_v2.csv` and are still in
milliseconds. The 5-d vector printed above them is the z-scored
sentence aggregate `model_ZuCo_SST.py` actually consumes — not the sum
of the word table.

## Reproducing this page

```bash
python -m pip install -r requirements.txt
python examples/run_all.py
```

The transcript lands in `examples/output/run_all.txt` (gitignored).
If a number here disagrees with your transcript, trust the transcript
and update this page in the same commit.
