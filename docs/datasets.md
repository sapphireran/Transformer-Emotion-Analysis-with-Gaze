# Datasets in this repo

Everything below is already checked in. The original ZuCo `.mat` files and the
raw SST phrase files are **not** in git; only the CSVs that the training
scripts read are here.

There are two experiment tracks that share a 3-class sentiment label
(`0` negative, `1` neutral, `2` positive) but **do not share the same
eye-tracking source**.

```
ZuCo track (real gaze, 400 movie-review sentences)
  ZuCo .mat  ->  read_ZuCo_mat.py  ->  ZuCo_et_csv_data/*_SR.csv
             ->  averages + scaling
             ->  join with ssts_ZuCo.csv
             ->  combined_sst_et_standard.csv
             ->  model_ZuCo_SST.py  (5-fold CV)

Full SST track (predicted / projected gaze, 11,853 sentences)
  stts_all_sentence_level.csv
             ->  token skeleton (sst_et_test.csv)
             ->  gaze_prediction/ tables
             ->  combined_full_sst_et.csv
             ->  train/valid/test_full_sst.csv
             ->  model_full_SST.py
```

## ZuCo + SST subset (real eye tracking)

ZuCo task 1 (normal reading) recorded 12 readers on the same movie-review
sentences. This repo keeps sentence-level and word-level exports.

| File | Rows | Role |
| --- | ---: | --- |
| `ZuCo_SST_data/ssts_ZuCo.csv` | 400 | Sentence text + sentiment only |
| `ZuCo_SST_data/combined_sst_et_standard.csv` | 400 | Text joined to z-scored sentence ET |
| `ZuCo_SST_data/combined_sst_et_min_max.csv` | 400 | Same join, min-max ET |
| `ZuCo_SST_data/train.csv` | 320 | 80% split from the standard table |
| `ZuCo_SST_data/valid.csv` | 40 | 10% holdout (not used by `model_ZuCo_SST.py`) |
| `ZuCo_SST_data/test.csv` | 40 | 10% holdout (not used by `model_ZuCo_SST.py`) |
| `ZuCo_et_csv_data/{1-12}_SR.csv` | 400 each, except subject 3 = 299 | Per-reader sentence ET |
| `ZuCo_et_csv_data/average_data.csv` | 400 | Mean across readers, raw units |
| `ZuCo_et_csv_data/standard_scaled_average_data.csv` | 400 | Z-scored reader mean |
| `ZuCo_et_csv_data/min_max_scaled_average_data.csv` | 400 | Min-max reader mean |
| `ZuCo_et_csv_data/word/{1-12}_SR.csv` | 7129 each, subject 3 = 5293 | Per-reader word ET |
| `ZuCo_et_csv_data/word/word_averages_v2.csv` | 7129 | Mean word ET across readers |

Label counts on the 400-sentence ZuCo table:

| Label | Name | Count | Share |
| ---: | --- | ---: | ---: |
| 0 | NEGATIVE | 123 | 30.8% |
| 1 | NEUTRAL | 137 | 34.2% |
| 2 | POSITIVE | 140 | 35.0% |

Sentences are short movie-review lines (mean 17.8 words, range 3–43). Subject 3
is short because `utils_ZuCo.DataTransformer` skips a known bad block for
task 1, subject index 2 (the third reader).

`model_ZuCo_SST.py` ignores `train.csv` / `valid.csv` / `test.csv`. It reloads
`combined_sst_et_standard.csv` and runs `StratifiedKFold(n_splits=5)`. The
80/10/10 files exist because `ZuCo_SST_data/spilt.py` was a separate experiment.

## Full SST (predicted eye tracking)

Stanford Sentiment Treebank sentences, already mapped to the same 3 classes,
with five sentence-level gaze values that are **not** recorded on these
sentences. The values look like z-scores (combined-file means are ~0, stds ~1)
and are nearly collinear with each other. Treat them as model-predicted or
lexicon-projected features, not as a second ZuCo recording.

| File | Rows | Role |
| --- | ---: | --- |
| `SST_data/stts_all_sentence_level.csv` | 11,853 | Raw `sentence,LABEL` (no header) |
| `SST_data/sst_et_test.csv` | 191,971 | Word skeleton: tokens + zeroed ET columns |
| `SST_data/combined_full_sst_et.csv` | 11,853 | Sentence + 5 predicted ET features |
| `SST_data/train_full_sst.csv` | 9,482 | 80% (`random_state=42`) |
| `SST_data/valid_full_sst.csv` | 1,185 | 10% |
| `SST_data/test_full_sst.csv` | 1,186 | 10% |

Full SST label counts:

| Split | Negative | Neutral | Positive | Total |
| --- | ---: | ---: | ---: | ---: |
| combined | 4,649 | 2,241 | 4,963 | 11,853 |
| train | 3,710 | 1,833 | 3,939 | 9,482 |
| valid | 476 | 209 | 500 | 1,185 |
| test | 463 | 199 | 524 | 1,186 |

Neutral is the minority class in every full-SST split. Mean sentence length is
19.2 words (range 2–56).

## Gaze-prediction side tables

These sit next to the trainers but are not imported by them.

| File | Rows | Notes |
| --- | ---: | --- |
| `gaze_prediction/data/prediction_test.csv` | 1,751 | Word-level predicted `nFix,FFD,GPT,TRT,GD` |
| `gaze_prediction/data/prediction_test_v2.csv` | 191,971 | Same schema, full SST token count |
| `gaze_prediction/data/provo.csv` | 2,659 | PROVO-style word table (`fixProp` instead of `GD`) |
| `gaze_prediction/data/convert_zuco_data.py` | — | Scales word averages into the predicted-word schema |

`prediction_test_v2.csv` has the same row count as `SST_data/sst_et_test.csv`.
That is the breadcrumb for “predict gaze on every SST token, then pool to a
sentence vector.”

## What is intentionally missing

- `ZuCo_mat_data/` — MATLAB dumps expected by `read_ZuCo_mat.py`
- `ZuCo_SST_data/all/{NEGATIVE,POSITIVE,NEUTRAL}/*.txt` — inputs to
  `convert_full_SST.py` / `save_SST_data.py`
- `models/best_*_model.pth` — created only after a training run
- Any workplace or proprietary corpus

If a script looks for those folders and dies, it is because this git snapshot
keeps derived CSVs, not the original binary recordings.
