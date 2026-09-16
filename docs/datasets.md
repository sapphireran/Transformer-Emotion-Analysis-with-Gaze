# Datasets and schemas

Everything below was counted from the CSVs in this checkout. Paths are relative
to the repository root.

## Label convention

Used everywhere sentiment is stored as an integer:

| Integer | Folder / SST string | Meaning |
| ---: | --- | --- |
| 0 | `NEGATIVE` | Negative |
| 1 | `NEUTRAL` | Neutral |
| 2 | `POSITIVE` | Positive |

`convert_full_SST.py` and `ZuCo_SST_data/save_SST_data.py` apply that mapping
when they walk a directory of `.txt` files.

## ZuCo-aligned SST (real gaze)

These 400 movie-review sentences are the overlap between ZuCo Task 1 (normal
reading) and SST-style polarity labels.

| File | Rows | Columns |
| --- | ---: | --- |
| `ZuCo_SST_data/ssts_ZuCo.csv` | 400 | `sentence_id`, `sentence`, `sentiment_label` |
| `ZuCo_SST_data/combined_sst_et_standard.csv` | 400 | text + z-scored gaze |
| `ZuCo_SST_data/combined_sst_et_min_max.csv` | 400 | text + min-max gaze |
| `ZuCo_SST_data/train.csv` | 320 | same as combined (standard) |
| `ZuCo_SST_data/valid.csv` | 40 | same as combined (standard) |
| `ZuCo_SST_data/test.csv` | 40 | same as combined (standard) |

Gaze columns on the combined files:

```text
omissionRate, nFixations, meanPupilSize, GD, TRT, FFD, SFD, GPT
```

`model_ZuCo_SST.py` **does not** use the 320/40/40 files. It reads
`combined_sst_et_standard.csv` and runs `StratifiedKFold(n_splits=5)`. It also
only feeds **five** gaze channels into the network:

```text
nFixations, FFD, GPT, TRT, GD
```

`omissionRate`, `meanPupilSize`, and `SFD` stay in the CSV but are ignored by
the current training script.

### Label counts

| Split | Negative | Neutral | Positive | Total |
| --- | ---: | ---: | ---: | ---: |
| Combined (used for CV) | 123 | 137 | 140 | 400 |
| `train.csv` | 103 | 107 | 110 | 320 |
| `valid.csv` | 7 | 14 | 19 | 40 |
| `test.csv` | 13 | 16 | 11 | 40 |

The hold-out valid split is badly balanced (only 7 negatives). That is a good
reason the training script prefers stratified CV over those three files.

Whitespace-token sentence length on the 400 rows: min 3, max 43, mean 17.8.

## Per-subject sentence gaze

`ZuCo_et_csv_data/{1-12}_SR.csv` — one file per ZuCo subject, sentence level,
raw (unscaled) features from `read_ZuCo_mat.py`.

| File | Sentences | Notes |
| --- | ---: | --- |
| `1_SR.csv` … `2_SR.csv`, `4_SR.csv` … `12_SR.csv` | 400 | Full Task 1 set |
| `3_SR.csv` | 299 | Subject index 2 in `DataTransformer`; sentences 299–399 dropped |

Shared columns:

```text
id, SentLen, omissionRate, nFixations, meanPupilSize, GD, TRT, FFD, SFD, GPT
```

Aggregates built from those twelve files:

| File | Scaling |
| --- | --- |
| `ZuCo_et_csv_data/average_data.csv` | Mean across subjects, raw units |
| `ZuCo_et_csv_data/standard_scaled_average_data.csv` | Column z-score of the average |
| `ZuCo_et_csv_data/min_max_scaled_average_data.csv` | Column min-max of the average |

`id` on these tables is the ZuCo sentence index and joins to
`ZuCo_SST_data/ssts_ZuCo.csv:sentence_id`.

## Per-subject word gaze

`ZuCo_et_csv_data/word/{1-12}_SR.csv` and the averages:

| File | Rows | Columns |
| --- | ---: | --- |
| `word/{k}_SR.csv` | ~7k | `id`, `Sent_ID`, `Word_ID`, `Word`, gaze, `WordLen` |
| `word/word_averages.csv` | 7,129 | same |
| `word/word_averages_v2.csv` | 7,129 | same |

`Sent_ID` looks like `0_NR` (sentence 0, normal reading). There are 400 distinct
`Sent_ID` values in `word_averages_v2.csv`.

`gaze_prediction/data/convert_zuco_data.py` rescales word-level
`nFixations` / `FFD` / `GPT` / `TRT` / `GD` into a 0–100 range for an older
prediction-training format.

## Full Stanford Sentiment Treebank + transferred gaze

| File | Rows | Columns |
| --- | ---: | --- |
| `SST_data/stts_all_sentence_level.csv` | 11,852 | **no header** — `sentence, POLARITY_STRING` |
| `SST_data/combined_full_sst_et.csv` | 11,853 | `sentence_id`, `sentence`, `sentiment_label`, `nFix`, `GD`, `TRT`, `FFD`, `GPT` |
| `SST_data/train_full_sst.csv` | 9,482 | same |
| `SST_data/valid_full_sst.csv` | 1,185 | same |
| `SST_data/test_full_sst.csv` | 1,186 | same |
| `SST_data/sst_et_test.csv` | 191,971 | word-level placeholder zeros |

`model_full_SST.py` reads the three hold-out CSVs. Gaze column names here are
the short forms (`nFix` not `nFixations`).

### Label counts

| Split | Negative | Neutral | Positive | Total |
| --- | ---: | ---: | ---: | ---: |
| Combined | 4,649 | 2,241 | 4,963 | 11,853 |
| Train | 3,710 | 1,833 | 3,939 | 9,482 |
| Valid | 476 | 209 | 500 | 1,185 |
| Test | 463 | 199 | 524 | 1,186 |

Neutral is the minority class (~19%). Whitespace-token length on the combined
file: min 2, max 56, mean 19.2.

`SST_data/spilt.py` (typo for *split*) created the 80 / 10 / 10 cut with
`random_state=42` and **no** stratify argument. `examples/split_integrity.py`
checks that the three files are a disjoint cover of `combined_full_sst_et.csv`.

## Predicted word-level gaze

| File | Rows | Domain |
| --- | ---: | --- |
| `gaze_prediction/data/prediction_test.csv` | 1,751 | Small predicted-gaze slice (`sentence_id` starts at 300) |
| `gaze_prediction/data/prediction_test_v2.csv` | 191,971 | Predicted gaze for the tokenized full SST |
| `gaze_prediction/data/provo.csv` | 2,659 | PROVO corpus tokens; uses `fixProp` instead of `GD` |

`prediction_test_v2.csv` has the same row count as `SST_data/sst_et_test.csv`
(the zero-filled word skeleton). That is the intended join: replace zeros with
a gaze predictor, then pool to sentence level for `combined_full_sst_et.csv`.

## Exploratory plots

`result/` holds three scatter/histogram images (train / test / PROVO). They are
not produced by any script still in the tree; treat them as archived figures.

## Files that are not in git

The MATLAB dumps (`ZuCo_mat_data/`) and the original `ZuCo_SST_data/all/{NEGATIVE,POSITIVE,NEUTRAL}/*.txt`
tree are **not** checked in. `ZuCo_SST_data/ZuCo_SST_data.zip` is. Rebuild
notes are in [reproduction.md](reproduction.md).

## How to re-print these counts

```bash
python3 examples/inspect_datasets.py
```
