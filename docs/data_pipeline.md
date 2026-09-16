# Data pipeline

How each checked-in CSV was (or would be) produced. Several steps need
inputs that are not in git; the outputs are.

```
ZuCo .mat ── read_ZuCo_mat.py ──► ZuCo_et_csv_data/{1-12}_SR.csv
                                      │
                                      ├─ get_average_sentence_level.py
                                      │     ──► average / min-max / standard CSVs
                                      │           └── join SST text
                                      │                 ──► ZuCo_SST_data/combined_sst_et_*.csv
                                      │
                                      └─ word/{1-12}_SR.csv
                                            └─ word/get_average.py
                                                  ──► word_averages_v2.csv
                                                        └─ gaze_prediction/data/convert_zuco_data.py

SST sentences ── convert_full_SST.py / save_SST_data.py ──► ssts_ZuCo.csv
             └─ stts_all_sentence_level.csv
                    ├─ convert_sst_to_et.py ──► sst_et_test.csv (zeros)
                    │                              └─ (external predictor)
                    │                                    ──► prediction_test_v2.csv
                    └─ (join predicted sentence gaze)
                          ──► combined_full_sst_et.csv
                                └─ SST_data/spilt.py
                                      ──► train / valid / test_full_sst.csv
```

## Stage A — ZuCo MATLAB to sentence CSV

`read_ZuCo_mat.py` constructs one `DataTransformer` for task 1,
sentence level, raw scaling, zero fill, then calls it for subjects
`0..11`. Each DataFrame's index becomes an `id` column. Files are
written as `et_csv_data/{i+1}_SR.csv` (the checked-in copies live under
`ZuCo_et_csv_data/`).

`utils_ZuCo.get_matfiles` builds

```text
os.getcwd() + '\\ZuCo_mat_data\\' + task
```

The backslashes are a Windows leftover. On Linux you need to change
`subdir` or the files will not be found. The function also **asserts
exactly 12** `.mat` files.

`DataTransformer.__call__` is where missing-sentence ranges are
skipped. If you ever re-extract task 2 or 3, read those `continue`
branches before trusting row counts.

`utils_ZuCo.split_data` exists to control order effects on task 1
(first half vs second half of each subject). Nothing in the training
path calls it.

## Stage B — Average and scale sentence gaze

`get_average_sentence_level.py`:

1. Read `et_csv_data/{1-12}_SR.csv`.
2. Replace `0` with NaN in every column except the first.
3. `concat` + `groupby(level=0).mean()` — this averages **on the
   default RangeIndex**, i.e. row position, which matches because each
   file has the same 400 aligned sentences.
4. Fit `MinMaxScaler` and `StandardScaler` on the numeric columns
   (including `id` handling that is a bit awkward: `id` is split off,
   scaled data is written with `id` as the index).

Join the scaled average to `ssts_ZuCo.csv` on `id` / `sentence_id` to
get `combined_sst_et_standard.csv` and `combined_sst_et_min_max.csv`.
That join step is not a committed script; the results are.

## Stage C — Word-level averages

`ZuCo_et_csv_data/word/get_average.py` concatenates the twelve word
files, `groupby(level=0).mean()` on the numeric gaze columns, then
pastes `id`, `Sent_ID`, `Word_ID`, `Word`, `WordLen` from subject 1.
Zeros are **not** replaced with NaN in the current file (the replace
line is commented out). Empty words become `"unknown"`.

`gaze_prediction/data/convert_zuco_data.py` then min-max scales
`nFixations` alone and `{FFD,GPT,TRT,GD}` on a **shared** min/max,
multiplies by 100, and writes the prediction-schema CSV. The hardcoded
paths (`training_data/word_averages_v2.csv`) are leftovers.

## Stage D — SST text

`convert_full_SST.py` walks `ZuCo_SST_data/all/{NEGATIVE,POSITIVE,NEUTRAL}/*.txt`,
maps folder name → `{0,1,2}`, sorts by the numeric filename, writes
`ssts_ZuCo.csv`.

`ZuCo_SST_data/save_SST_data.py` is the same idea relative to `all/`
and writes `output.csv`. `sentence_id` stays a string there.

Full SST lives in `SST_data/stts_all_sentence_level.csv` already.

## Stage E — Word scaffold for predicted gaze

`SST_data/convert_sst_to_et.py`:

- NLTK `word_tokenize`
- keep tokens matching `^[A-Za-z]+$` (drops punctuation and
  `-LRB-` / `-RRB-` style remnants if they contain non-letters)
- if a sentence has no surviving token, write a single `unknown`
- write zeros for all five gaze columns

Output: 191,971 rows in `sst_et_test.csv`. The predictor that filled
`prediction_test_v2.csv` is not in this repo; only the product is.

## Stage F — Splits

Both `SST_data/spilt.py` and `ZuCo_SST_data/spilt.py` are the same
80/10/10 `train_test_split` with `random_state=42`. Neither is
stratified. Neither touches gaze columns specially.

`model_full_SST.py` consumes the full-SST split.
`model_ZuCo_SST.py` ignores `ZuCo_SST_data/{train,valid,test}.csv` and
does 5-fold CV on the combined 400.

## Stage G — What the training scripts load

Full SST (`model_full_SST.py`):

```python
df[['nFix', 'FFD', 'GPT', 'TRT', 'GD']]
df[['sentence', 'sentiment_label']]
```

ZuCo (`model_ZuCo_SST.py`):

```python
df[['nFixations', 'FFD', 'GPT', 'TRT', 'GD']]
df[['sentence', 'sentiment_label']]
```

Column order in the CSV does not have to match; both scripts select by
name. Do not rename `nFix` ↔ `nFixations` without editing the script.

## Idempotence

Re-running the split scripts will overwrite `train` / `valid` / `test`
with the same rows (`random_state=42`) as long as the parent CSV is
unchanged. Re-running `read_ZuCo_mat.py` without the `.mat` tree will
crash. Re-running `convert_sst_to_et.py` requires NLTK `punkt` and will
wipe any predicted values if you point it at the same output path.

The examples suite never writes back into `SST_data/` or
`ZuCo_*`. It only writes under `examples/output/` when asked.
