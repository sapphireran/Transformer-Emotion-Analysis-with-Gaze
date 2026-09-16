# Preprocessing pipeline

This is the path from ZuCo MATLAB structs to the two training CSVs.
Several scripts hard-code relative paths and were clearly run from
different working directories at different times. The derived files
**are already in the repo**; you only need to re-run a stage if you
change a scaling choice or obtain the original `.mat` files.

## Stage A — MATLAB → per-subject sentence CSVs

**Scripts:** `utils_ZuCo.py`, `read_ZuCo_mat.py`

```
ZuCo_mat_data/task1/*.mat   (12 files, not in this checkout)
        │  DataTransformer('task1', level='sentence', scaling='raw', fillna='zeros')
        ▼
ZuCo_et_csv_data/{1-12}_SR.csv
```

`get_matfiles()` builds `os.getcwd() + '\\ZuCo_mat_data\\' + task`.
The backslashes are a Windows leftover; on Linux that string is a
**literal** folder name unless you edit it. See
[08-known-quirks.md](08-known-quirks.md).

`read_ZuCo_mat.py` writes to `et_csv_data/` (no `ZuCo_` prefix). The
checked-in files live in `ZuCo_et_csv_data/`. Either change the path
or symlink before re-running.

`DataTransformer.__call__(subject)` (subject `0..11`):

1. `scipy.io.loadmat(..., squeeze_me=True, struct_as_record=False)`
   and take `['sentenceData']`.
2. Skip documented bad sentence index ranges for a few subject/task
   pairs.
3. Aggregate word attributes to one sentence row (see
   [03-eye-tracking-features.md](03-eye-tracking-features.md)).
4. Drop rows containing `±inf`.
5. Scale if requested; raw path keeps milliseconds.
6. Fill remaining NaNs.

Word-level mode (`level='word'`) writes `Sent_ID`, `Word_ID`, `Word`,
the gaze fields, and `WordLen`. That path is how the `word/` folder
was produced (not via `read_ZuCo_mat.py`, which is sentence-only).

## Stage B — Average readers, then scale

**Script:** `get_average_sentence_level.py`

```
et_csv_data/{1-12}_SR.csv
        │  0 → NaN on all but the first column
        │  concat → groupby(level=0).mean()
        ▼
average_data.csv
        ├── MinMaxScaler  → min_max_scaled_average_data.csv
        └── StandardScaler → standard_scaled_average_data.csv
```

The script again says `folder_path = 'et_csv_data'`. Point it at
`ZuCo_et_csv_data` to regenerate the files that are already here.

Averaging **by row index** assumes every subject file is aligned on
the same sentence `id`. That is true for the 400-row files. Subject 3
has 299 rows, so `groupby(level=0)` still averages whatever landed on
index `0..298` and will silently mix “sentence 299 for subject 3”
with “sentence 299 for everyone else” only if indices were reset
identically. After `reset_index` in `read_ZuCo_mat.py`, subject 3’s
`id` column is `0..298`, not the original ZuCo sentence numbers that
were skipped. **The average of the tail sentences is therefore
computed over 11 subjects, not 12**, and the dropped sentence ids are
absent for subject 3 rather than stored as NaN. Keep that in mind if
you treat `average_data.csv` as a clean 12-reader mean.

## Stage C — Attach SST labels to the 400 sentences

**Scripts:** `convert_full_SST.py` (root) or `ZuCo_SST_data/save_SST_data.py`

```
ZuCo_SST_data/all/{NEGATIVE,POSITIVE,NEUTRAL}/*.txt   (not in repo)
        │  filename stem → sentence_id
        │  folder name  → {0,1,2}
        ▼
ZuCo_SST_data/ssts_ZuCo.csv
```

The two scripts differ only in paths and whether `sentence_id` is
cast to `int`. The checked-in `ssts_ZuCo.csv` is the source of truth.

The **join** onto scaled gaze is not a dedicated script in this
checkout. It is a merge of `ssts_ZuCo.csv` with
`standard_scaled_average_data.csv` / `min_max_scaled_average_data.csv`
on sentence id (`id` in the gaze table). The results are:

- `ZuCo_SST_data/combined_sst_et_standard.csv`  ← used by training
- `ZuCo_SST_data/combined_sst_et_min_max.csv`

`ZuCo_SST_data/spilt.py` then writes `train.csv` / `valid.csv` /
`test.csv` (80/10/10, `random_state=42`, not stratified).

## Stage D — Word-level averages

**Script:** `ZuCo_et_csv_data/word/get_average.py`

```
word/{1-12}_SR.csv
        │  mean nFixations, meanPupilSize, GD, TRT, FFD, SFD, GPT
        │  take id, Sent_ID, Word_ID, Word, WordLen from 1_SR.csv
        ▼
word/word_averages_v2.csv
```

Zeros are **not** converted to NaN in the current file (that line is
commented out), so a never-fixated word contributes 0 to the mean.
Missing words become `unknown`.

## Stage E — Transfer gaze onto full SST

Two different helpers exist; they are not a single clean DAG.

### E1. Placeholder word rows

`SST_data/convert_sst_to_et.py` reads `stts_all_sentence_level.csv`
(no header), tokenizes with NLTK `word_tokenize`, keeps `[A-Za-z]+`
only, and writes `sst_et_test.csv` with gaze columns **all zero**.
This is a schema scaffold, not a model.

### E2. Predicted / scaled word rows

`gaze_prediction/data/convert_zuco_data.py` converts a word-average
table into `sentence_id, word_id, word, nFix, FFD, GPT, TRT, GD`
with 0–100 min–max scaling. `prediction_test_v2.csv` has 191,971
rows — the same order of magnitude as `sst_et_test.csv` — and is the
likely source of per-token predictions that were later **pooled** into
the sentence-level `nFix, GD, TRT, FFD, GPT` columns on
`combined_full_sst_et.csv`.

The pooling script that turns those word rows into one 5-d vector per
SST sentence is **not in this repository**. Treat
`SST_data/combined_full_sst_et.csv` as an imported artifact.

`SST_data/spilt.py` splits that artifact 80/10/10 (`random_state=42`,
no `stratify`).

## Stage F — Tokenize for the transformer

This happens **inside** the training scripts, not as a saved file.

1. `pandas.read_csv`
2. Hugging Face `Dataset.from_pandas` on `sentence` + `sentiment_label`
3. `tokenizer(..., padding='max_length', truncation=True, max_length=128)`
4. Concatenate the five gaze columns back on
5. `CustomDataset` stores `input_ids`, `attention_mask`, `labels`,
   `eye_tracking_features`

RoBERTa and BERT tokenizers are chosen from `model_type.startswith('bert')`.
There is no extra marker token for gaze.

## Checklist if you regenerate data

1. Put 12 Task-1 `.mat` files where `get_matfiles` can see them.
2. Fix Windows path separators in `utils_ZuCo.py`.
3. Align output directories (`et_csv_data` vs `ZuCo_et_csv_data`).
4. Re-average, then re-join labels on `sentence_id`.
5. Diff against the checked-in `combined_sst_et_standard.csv` before
   declaring a new training set.
6. Do not assume `train.csv` for ZuCo matches the KFold splits.

## What the example scripts assume

They only read committed CSVs:

- schemas and counts → `examples/01_inspect_datasets.py`
- column contracts → `examples/02_schema_check.py`
- no MATLAB, no NLTK, no Hugging Face
