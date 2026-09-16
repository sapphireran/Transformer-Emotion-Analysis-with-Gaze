# Two tracks, two corpora

This repo is easy to misread as “SST with eye-tracking features.” It is two
experiments that share a fusion class and almost no sentences.

## Track A — ZuCo-SST (measured gaze, 400 reviews)

ZuCo Task 1 (normal reading / NR) recorded 12 readers on a 400-sentence
slice of movie reviews. The MATLAB extractor in `utils_ZuCo.py` writes
per-reader CSVs under `ZuCo_et_csv_data/`. Those tables are averaged
(with the packing bug documented in [subject-3-reindex.md](subject-3-reindex.md)),
scaled, and joined onto the 400 review strings in
`ZuCo_SST_data/ssts_ZuCo.csv`.

The file the trainer actually loads is
`ZuCo_SST_data/combined_sst_et_standard.csv`:

- 400 rows, labels `{0: 123, 1: 137, 2: 140}` (almost balanced)
- gaze columns: `omissionRate, nFixations, meanPupilSize, GD, TRT, FFD, SFD, GPT`
- `model_ZuCo_SST.py` consumes **five** of those: `nFixations, FFD, GPT, TRT, GD`
- protocol: `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`
- 20 epochs, batch size 16, Adam `5e-5`

The leftover `ZuCo_SST_data/{train,valid,test}.csv` (320/40/40) come from
`spilt.py` and are **not** used by the trainer. The 40-row valid file has
only **7** negatives, which is a good reason to prefer the k-fold.

## Track B — full SST (predicted gaze, ~11.8k reviews)

`SST_data/stts_all_sentence_level.csv` is a headerless dump of SST sentences
with string labels `POSITIVE|NEGATIVE|NEUTRAL`. Those become
`combined_full_sst_et.csv` with numeric labels `{0,1,2}` and five z-scored
gaze columns named `nFix, GD, TRT, FFD, GPT` (note: `nFix`, not
`nFixations`).

`SST_data/spilt.py` (the filename is a typo) does `train_test_split(..., random_state=42)`
twice to get 9482 / 1185 / 1186. Ids partition the combined table. Two
*strings* still leak across splits because the split is on rows, not unique
text. See [splits.md](splits.md).

`model_full_SST.py` trains 5 epochs, batch 256, and checkpoints on **validation
accuracy** (the comment says F1). The test loop then reports the last batch
only — [known-issues.md](known-issues.md).

## They barely overlap

`examples/05_split_fingerprint.py` finds **three** shared review strings
between the 400-row ZuCo table and the 11.8k full-SST table. You cannot
treat Track B as “the same items with predicted gaze instead of measured
gaze.” Track B is a transfer setting: a gaze predictor (see
[gaze-prediction-track.md](gaze-prediction-track.md)) filled five scalars for
reviews that ZuCo readers never saw, then a classifier used those scalars as
the sidecar.

## Shared fusion, different column names

Both trainers define the same `EyeTrackingModel`:

```
pooler (768)  ⊕  Linear(5 → 16)(gaze)  →  Dropout → Linear(784 → 3)
```

The five gaze names are not the same string:

| Role | ZuCo-SST | Full SST |
| --- | --- | --- |
| Fixation count | `nFixations` | `nFix` |
| First fixation duration | `FFD` | `FFD` |
| Go-past time | `GPT` | `GPT` |
| Total reading time | `TRT` | `TRT` |
| Gaze duration | `GD` | `GD` |

Copy-pasting a DataLoader from one trainer to the other will `KeyError`.
The ZuCo table also has `SFD`, `meanPupilSize`, and `omissionRate`, which the
model never sees.

## Why two batch sizes

400 rows × 5 folds × 20 epochs at batch 16 is a small run. 11.8k rows at
batch 256 is a different compute budget (and the last-batch test bug is
invisible on a 400-row loader if you never wrote that loop the same way —
`model_ZuCo_SST.py` correctly `extend`s).
