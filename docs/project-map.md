# Project map

Personal research workspace for **three-class sentiment** on movie-review
sentences, with an optional **eye-tracking** branch fused into BERT or
RoBERTa. This page is a map of the clone, not a paper.

```
text ──► BERT / RoBERTa pooler (768) ──┐
                                       ├─► concat ──► dropout ──► Linear(784 → 3)
gaze ──► Linear(5 → 16) ───────────────┘
```

Labels: `0` negative, `1` neutral, `2` positive.

## Two tracks, two scripts

| Track | Rows | Gaze | Script | Protocol |
| --- | ---: | --- | --- | --- |
| ZuCo–SST | 400 | Human, 12 readers, then z-scored | `model_ZuCo_SST.py` | Stratified 5-fold on the combined table |
| Full SST | 11,853 | Projected / predicted | `model_full_SST.py` | Fixed 80/10/10 + checkpoint |

Use ZuCo when the gaze values should be treated as reading behavior.
Use full SST when you need scale. Do not quote full-SST `TRT` as a
human measurement.

## Tree

```
docs/                      this reading layer
examples/                  stdlib inspectors (see examples/README.md)
zuco_lab/                  helpers the examples import
tests/                     unittest coverage for zuco_lab
model_ZuCo_SST.py          RoBERTa/BERT ± gaze, 5-fold
model_full_SST.py          same models, SST split + checkpoint
utils_ZuCo.py              MATLAB → sentence/word tables
read_ZuCo_mat.py           writes per-reader CSVs (path is stale)
get_average_sentence_level.py
convert_full_SST.py        SST folder → ssts_ZuCo.csv
ZuCo_SST_data/             400 labeled sentences + scaled gaze
ZuCo_et_csv_data/          per-reader and averaged ET
SST_data/                  11k labeled sentences + projected gaze
gaze_prediction/data/      PROVO + predicted word ET
result/                    older scatter/hist plots
```

## What is *not* in the clone

- The ZuCo MATLAB tree (`ZuCo_mat_data/`)
- The SST text folders (`ZuCo_SST_data/all/{NEGATIVE,POSITIVE,NEUTRAL}`)
- Trained `.pth` checkpoints
- A `models/` directory (the full-SST script wants to write one)

Producer scripts that point at those missing folders will raise
`FileNotFoundError`. The consumer CSVs they once wrote are already here.

## Personal layer vs original scripts

`zuco_lab` and `examples/` only *read* the checked-in tables. They do
not patch `model_full_SST.py` or re-average the twelve readers. The
subject-3 remapping and the test-loop overwrite are documented, not
silently "fixed," so a later training run still matches this clone.
