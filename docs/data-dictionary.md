# Data dictionary

Row counts and schemas as of this checkout. Generated again by `python3 examples/inspect_datasets.py`.

Sentiment integers: `0 = negative`, `1 = neutral`, `2 = positive`.

## Training tables

### `ZuCo_SST_data/combined_sst_et_standard.csv`

Primary ZuCo training table. 400 rows. Used by `model_ZuCo_SST.py`.

| Column | Type | Notes |
| --- | --- | --- |
| `sentence_id` | int | 0–399, aligned with `ssts_ZuCo.csv` |
| `sentence` | str | Movie-review sentence shown in ZuCo task 1 |
| `sentiment_label` | int | 123 / 137 / 140 for 0 / 1 / 2 |
| `omissionRate` | float | z-scored; **not** used by the model |
| `nFixations` | float | z-scored; fusion feature 1 |
| `meanPupilSize` | float | z-scored; **not** used by the model |
| `GD` | float | z-scored; fusion feature 5 |
| `TRT` | float | z-scored; fusion feature 4 |
| `FFD` | float | z-scored; fusion feature 2 |
| `SFD` | float | z-scored; **not** used by the model |
| `GPT` | float | z-scored; fusion feature 3 |

Holdout copies of the same schema:

| File | Rows | Label counts 0/1/2 |
| --- | --- | --- |
| `ZuCo_SST_data/train.csv` | 320 | 103 / 107 / 110 |
| `ZuCo_SST_data/valid.csv` | 40 | 7 / 14 / 19 |
| `ZuCo_SST_data/test.csv` | 40 | 13 / 16 / 11 |

The training script does **not** use those three files. It re-splits `combined_sst_et_standard.csv` with 5-fold CV. The 80/10/10 CSVs come from `ZuCo_SST_data/spilt.py` and are useful for ad-hoc inspection (`examples/split_sanity_check.py` verifies no sentence_id leakage).

### `ZuCo_SST_data/combined_sst_et_min_max.csv`

Same 400 sentences, min–max scaled gaze in `[0, 1]`. Not referenced by either training script. Useful as a control table if you want features on a bounded range.

### `ZuCo_SST_data/ssts_ZuCo.csv`

400 rows: `sentence_id, sentence, sentiment_label` only. Built by `convert_full_SST.py` from a folder of text files that is **not** checked in (`ZuCo_SST_data/all/{NEGATIVE,POSITIVE,NEUTRAL}`).

### `SST_data/combined_full_sst_et.csv`

11,853 rows. Standardized transferred gaze.

| Column | Type | Notes |
| --- | --- | --- |
| `sentence_id` | int | 0–based SST index in this export |
| `sentence` | str | Mean length ≈ 19.2 words (min 2, max 56) |
| `sentiment_label` | int | 4649 / 2241 / 4963 |
| `nFix` | float | z-scored transferred fixation count |
| `GD` | float | z-scored |
| `TRT` | float | z-scored |
| `FFD` | float | z-scored |
| `GPT` | float | z-scored |

Splits from `SST_data/spilt.py` (`train_test_split`, `test_size=0.2` then 0.5, `random_state=42`):

| File | Rows | Label counts 0/1/2 |
| --- | --- | --- |
| `SST_data/train_full_sst.csv` | 9482 | 3710 / 1833 / 3939 |
| `SST_data/valid_full_sst.csv` | 1185 | 476 / 209 / 500 |
| `SST_data/test_full_sst.csv` | 1186 | 463 / 199 / 524 |

`model_full_SST.py` reads the three split files.

### `SST_data/stts_all_sentence_level.csv`

11,852 rows, **no header**. Columns are raw sentence text and a string label (`POSITIVE` / `NEGATIVE` / `NEUTRAL`). One row shorter than `combined_full_sst_et.csv`. Used as input to `SST_data/convert_sst_to_et.py`.

### `SST_data/sst_et_test.csv`

191,971 word rows: `sentence_id, word_id, word, nFix, FFD, GPT, TRT, GD`. All gaze values are **0**. This is the token skeleton produced by `convert_sst_to_et.py` (NLTK `word_tokenize`, alphabetic tokens only). Predicted numbers belong in `gaze_prediction/data/`.

## ZuCo eye-tracking exports

### `ZuCo_et_csv_data/{1-12}_SR.csv`

Sentence-level features, one file per subject, from `read_ZuCo_mat.py` + `DataTransformer(..., level='sentence', scaling='raw')`.

| Column | Meaning |
| --- | --- |
| `id` | Sentence index 0–399 (subject 3 skips some ids; file has 299 rows) |
| `SentLen` | Number of words in the stimulus |
| `omissionRate` | Fraction of words without a recorded fixation |
| `nFixations` | Mean fixations per fixated word |
| `meanPupilSize` | Mean pupil size over those words |
| `GD`, `TRT`, `FFD`, `SFD`, `GPT` | Mean durations (ms) over fixated words |

Subject 3 has 299 rows; everyone else has 400. The skip ranges are hard-coded in `utils_ZuCo.DataTransformer.__call__` for task 1 / subject 2 (0-based subject index 2 → file `3_SR.csv`).

### `ZuCo_et_csv_data/average_data.csv`

400 rows. Mean across subjects of the sentence-level raw features. `id` is the sentence index.

Raw means (this checkout):

| Feature | Mean | Std | Min | Max |
| --- | --- | --- | --- | --- |
| SentLen | 17.81 | 8.06 | 3 | 42 |
| omissionRate | 0.319 | 0.067 | 0.156 | 0.601 |
| nFixations | 1.687 | 0.310 | 1.205 | 3.583 |
| meanPupilSize | 797.0 | 60.9 | 679.7 | 969.7 |
| GD | 141.4 | 21.5 | 107.6 | 272.7 |
| TRT | 202.6 | 47.9 | 131.1 | 427.0 |
| FFD | 116.9 | 8.0 | 101.5 | 165.9 |
| SFD | 71.6 | 10.9 | 39.3 | 121.8 |
| GPT | 241.8 | 56.7 | 153.2 | 586.9 |

### `ZuCo_et_csv_data/standard_scaled_average_data.csv`

Same 400 rows, `StandardScaler` over the feature columns (`get_average_sentence_level.py`). These gaze columns match `combined_sst_et_standard.csv` after the join.

### `ZuCo_et_csv_data/min_max_scaled_average_data.csv`

Same 400 rows, `MinMaxScaler`. Joined form is `combined_sst_et_min_max.csv`.

### `ZuCo_et_csv_data/word/{1-12}_SR.csv` and `word/word_averages_v2.csv`

Word-level records: `id, Sent_ID, Word_ID, Word, nFixations, meanPupilSize, GD, TRT, FFD, SFD, GPT, WordLen`.

- Complete subjects: 7,129 rows
- Subject 3: 5,293 rows
- `Sent_ID` looks like `0_NR` (natural reading / task 1)
- `word_averages_v2.csv` averages the numeric columns across the 12 subject files by row index (see caveats in [known-issues.md](known-issues.md))

## Predicted-gaze tables

### `gaze_prediction/data/prediction_test_v2.csv`

191,971 word rows, same columns as `sst_et_test.csv`, but filled with predicted values. Means: nFix ≈ 20.4, FFD ≈ 4.42, GPT ≈ 7.67, TRT ≈ 6.63, GD ≈ 5.02. These look like the 0–100 scaled units from `gaze_prediction/data/convert_zuco_data.py`, not milliseconds.

### `gaze_prediction/data/prediction_test.csv`

1,751 word rows. Smaller predicted set (starts at `sentence_id` 300 in the sample). Same five gaze columns.

### `gaze_prediction/data/provo.csv`

2,659 word rows from a Provo-style export: `sentence_id, word_id, word, nFix, FFD, GPT, TRT, fixProp`. No `GD` column. Used for the scatter plot in `result/provo_data_scatter_hist_plots.png`.

## Plots in `result/`

| File | What it shows |
| --- | --- |
| `train_data_scatter_hist_plots.png` | Pairwise scatter + marginal histograms of predicted train gaze |
| `test_data_scatter_hist_plots.png` | Same for predicted test gaze |
| `provo_data_scatter_hist_plots.png` | Same for the Provo sample |

The plots are diagnostic, not model metrics. nFix, TRT, and GD are strongly linearly related in the predicted sets; FFD is more tightly concentrated.

## Zip

`ZuCo_SST_data/ZuCo_SST_data.zip` is an archive of the ZuCo SST folder. Prefer the extracted CSVs.

## Who reads what

| Consumer | Files |
| --- | --- |
| `model_ZuCo_SST.py` | `ZuCo_SST_data/combined_sst_et_standard.csv` |
| `model_full_SST.py` | `SST_data/{train,valid,test}_full_sst.csv` |
| `examples/*.py` | the tables above (read-only) |
| `utils_ZuCo.py` / `read_ZuCo_mat.py` | MATLAB files under `ZuCo_mat_data/` (**not checked in**) |
