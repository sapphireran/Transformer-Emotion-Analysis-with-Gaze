# Datasets

Every numeric claim below was counted from the files in this checkout. Header rows are excluded from "rows".

## Inventory

| File | Rows | What it is |
| --- | --- | --- |
| `ZuCo_SST_data/ssts_ZuCo.csv` | 400 | Review text + 3-way label only |
| `ZuCo_SST_data/combined_sst_et_standard.csv` | 400 | Same sentences + z-scored sentence ET |
| `ZuCo_SST_data/combined_sst_et_min_max.csv` | 400 | Same sentences + min-max sentence ET |
| `ZuCo_SST_data/train.csv` | 320 | 80% holdout of the standard combined table |
| `ZuCo_SST_data/valid.csv` | 40 | 10% holdout |
| `ZuCo_SST_data/test.csv` | 40 | 10% holdout |
| `ZuCo_et_csv_data/{1–12}_SR.csv` | 400 each, except subject 3 = 299 | Per-reader sentence ET (raw-ish) |
| `ZuCo_et_csv_data/average_data.csv` | 400 | Mean across readers |
| `ZuCo_et_csv_data/standard_scaled_average_data.csv` | 400 | z-scored reader mean |
| `ZuCo_et_csv_data/min_max_scaled_average_data.csv` | 400 | min-max reader mean |
| `ZuCo_et_csv_data/word/word_averages_v2.csv` | 7,128 | Word-level reader means |
| `SST_data/stts_all_sentence_level.csv` | 11,852 | Full SST text + string label |
| `SST_data/combined_full_sst_et.csv` | 11,853 | SST + five sentence ET channels |
| `SST_data/train_full_sst.csv` | 9,482 | 80% of the fused full SST table |
| `SST_data/valid_full_sst.csv` | 1,185 | 10% |
| `SST_data/test_full_sst.csv` | 1,186 | 10% |
| `SST_data/sst_et_test.csv` | 191,971 | Word-level SST skeleton (ET often zeroed) |
| `gaze_prediction/data/prediction_test_v2.csv` | 191,968 | Predicted word-level gaze on SST |
| `gaze_prediction/data/prediction_test.csv` | ~1.7k | Smaller predicted-gaze sample |
| `gaze_prediction/data/provo.csv` | 2,652 | PROVO-style word gaze reference |

`ZuCo_SST_data/ZuCo_SST_data.zip` is a snapshot of the text dump; the CSVs are the source of truth for examples.

## ZuCo Task 1 (normal reading)

[ZuCo](https://osf.io/q3zws/) records simultaneous EEG and eye-tracking while 12 people read English sentences. This project only uses **Task 1**: normal reading of movie-review sentences (the NR / `_NR` suffix in word IDs). Tasks 2 (Wikipedia NR) and 3 (task-specific reading) are handled by `DataTransformer` but are not wired into the sentiment models.

`read_ZuCo_mat.py` instantiates:

```python
DataTransformer('task1', level='sentence', scaling='raw', fillna='zeros')
```

and writes `et_csv_data/{i+1}_SR.csv` for `i in 0..11`. In this checkout those files already live under `ZuCo_et_csv_data/`.

### Subject 3 is short

`ZuCo_et_csv_data/3_SR.csv` has 299 sentences, not 400. `DataTransformer` also drops known-bad trial ranges for several (task, subject) pairs — see [preprocessing.md](preprocessing.md). Any subject-average that `groupby(level=0).mean()`s the twelve files will silently average over fewer readers on those rows. The checked-in `average_data.csv` still has 400 rows, so the missing subject is treated as absent rather than as zeros.

### Word-level tables

`ZuCo_et_csv_data/word/{1–12}_SR.csv` plus `word/get_average.py` produce `word_averages_v2.csv`. Columns:

`id, Sent_ID, Word_ID, Word, nFixations, meanPupilSize, GD, TRT, FFD, SFD, GPT, WordLen`

`Sent_ID` looks like `0_NR`. `gaze_prediction/data/convert_zuco_data.py` parses the integer before `_` as `sentence_id` and rescales gaze onto a 0–100 display range for foreign tools.

## ZuCo SST labels

`convert_full_SST.py` (repo root) and `ZuCo_SST_data/save_SST_data.py` both walk a local `NEGATIVE/` `POSITIVE/` `NEUTRAL/` folder of `.txt` files. The folder is not in git; the product is `ssts_ZuCo.csv`:

| Column | Type | Notes |
| --- | --- | --- |
| `sentence_id` | int | Matches the ZuCo sentence index |
| `sentence` | str | Raw review sentence |
| `sentiment_label` | `{0,1,2}` | neg / neu / pos |

The combined files join those labels to the scaled sentence ET on `sentence_id`.

## Full SST

`SST_data/stts_all_sentence_level.csv` is a two-column, headerless dump: `sentence, LABEL` where `LABEL` is the string `POSITIVE` / `NEGATIVE` / `NEUTRAL`.

`SST_data/convert_sst_to_et.py` tokenizes with NLTK, keeps `[A-Za-z]+` tokens, and writes a word-level table with **zeroed** ET columns. That file is a scaffold for a later gaze predictor, not observed eye-tracking.

The fused modeling tables (`*_full_sst.csv`) already carry numeric `nFix, GD, TRT, FFD, GPT` and integer `sentiment_label`. Those gaze values are transferred / predicted, not recorded on SST readers.

## Split recipes

Both `ZuCo_SST_data/spilt.py` and `SST_data/spilt.py` (filename is a typo) do the same thing:

```python
train, valid_test = train_test_split(df, test_size=0.2, random_state=42)
valid, test = train_test_split(valid_test, test_size=0.5, random_state=42)
```

That is **not** stratified. Class balance can drift on the 40-row ZuCo valid/test slices. `examples/scripts/04_split_sanity_check.py` prints the drift.

`model_ZuCo_SST.py` ignores those files and re-splits with `StratifiedKFold`. `model_full_SST.py` uses the pre-written full-SST CSVs as-is.

## ID alignment rules

- ZuCo sentence tables: join key is `sentence_id` / `id` in `0..399`.
- ZuCo word tables: `Sent_ID` = `{sentence_id}_NR` for Task 1.
- Full SST fused tables: `sentence_id` is the row's original SST index, **not** a dense 0..N range after splitting. Do not assume `train_full_sst.csv` ids are contiguous.
- Predicted gaze: `(sentence_id, word_id)` is the join key. `word_id` is the in-sentence token index after the predictor's tokenizer, which may not match NLTK `word_tokenize`.

## Which file each script reads

| Script | Input | Output / effect |
| --- | --- | --- |
| `read_ZuCo_mat.py` | `ZuCo_mat_data/task1/*.mat` (local) | `et_csv_data/{1-12}_SR.csv` |
| `get_average_sentence_level.py` | `et_csv_data/{1-12}_SR.csv` | min-max and standard averages |
| `ZuCo_et_csv_data/word/get_average.py` | `word/{1-12}_SR.csv` | `word_averages_v2.csv` |
| `convert_full_SST.py` | `ZuCo_SST_data/all/{LABEL}/*.txt` | `ssts_ZuCo.csv` |
| `ZuCo_SST_data/save_SST_data.py` | `all/{LABEL}/*.txt` | `output.csv` |
| `ZuCo_SST_data/spilt.py` | `combined_sst_et_standard.csv` | train/valid/test |
| `SST_data/spilt.py` | `combined_full_sst_et.csv` | `*_full_sst.csv` |
| `SST_data/convert_sst_to_et.py` | `stts_all_sentence_level.csv` | `sst_et_test.csv` |
| `gaze_prediction/data/convert_zuco_data.py` | `training_data/word_averages_v2.csv` | scaled word gaze |
| `model_ZuCo_SST.py` | `combined_sst_et_standard.csv` | prints 5-fold metrics |
| `model_full_SST.py` | `SST_data/{train,valid,test}_full_sst.csv` | `models/best_{type}_model.pth` |

## Plots already in `result/`

`result/train_data_scatter_hist_plots.png` (and the test / PROVO siblings) are pairwise scatter + marginal histogram plots of `nFix`, `FFD`, `GPT`, `TRT`, `GD`. They show the usual reading-measure story: TRT and nFix are strongly coupled; GPT is heavy-tailed; FFD is more Gaussian after scaling. Use them as a visual check when regenerating feature reports.
