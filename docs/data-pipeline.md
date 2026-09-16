# Data pipeline

This is the personal reproduction path for the tables already in the repo. Several scripts still contain local path names from the original machine (`et_csv_data`, `ZuCo_SST_data/all`, Windows-style `\\ZuCo_mat_data\\`). The notes below describe what the code *intends* to do, plus the path mismatches you will hit if you re-run it cold.

## Stage 0 — raw ZuCo MATLAB

Expected layout (not checked in):

```
ZuCo_mat_data/
  task1/
    <12 .mat files>
  task2/
    ...
  task3/
    ...
```

`utils_ZuCo.get_matfiles("task1")` builds `cwd + subdir + task` and asserts that it finds exactly 12 files. The default `subdir` is `\\ZuCo_mat_data\\`. On Linux that backslash prefix is a single directory name, not a nested path. If you re-run the reader, pass a POSIX subdir or edit the default.

`read_ZuCo_mat.py` constructs one `DataTransformer('task1', level='sentence', scaling='raw', fillna='zeros')` and writes `et_csv_data/{1-12}_SR.csv`. The checked-in files live in `ZuCo_et_csv_data/`, so either create that alias or change the output path.

## Stage 1 — sentence-level subject CSVs

For each of 12 readers, `DataTransformer.__call__(subject)`:

1. loads `sentenceData` from the `.mat`
2. skips known bad index ranges for that subject/task
3. aggregates word measures to one row per sentence
4. replaces infinities / NaNs
5. optionally scales

Output columns: `id`, `SentLen`, `omissionRate`, `nFixations`, `meanPupilSize`, `GD`, `TRT`, `FFD`, `SFD`, `GPT`.

Word-level mode writes one row per token with `Sent_ID`, `Word_ID`, `Word`, the same measures, and `WordLen`. Tokens are stripped of punctuation; the first word of a sentence is lowercased.

## Stage 2 — average across readers

`get_average_sentence_level.py`:

1. reads `{1-12}_SR.csv` from `et_csv_data`
2. turns zeros (except the id column) into NaN
3. concatenates and takes the mean aligned by row index
4. fits `MinMaxScaler` and `StandardScaler` on the non-id columns
5. writes `min_max_scaled_average_data.csv` and `standard_scaled_average_data.csv`

`ZuCo_et_csv_data/word/get_average.py` does the word-level analogue and writes `word_averages_v2.csv`. It keeps `id`, `Sent_ID`, `Word_ID`, `Word`, `WordLen` from subject 1 and averages the numeric measures.

## Stage 3 — SST labels for the ZuCo sentences

`convert_full_SST.py` (repo root) and `ZuCo_SST_data/save_SST_data.py` both walk:

```
<all>/NEGATIVE/*.txt
<all>/POSITIVE/*.txt
<all>/NEUTRAL/*.txt
```

Each file name is the sentence id; file contents are the sentence. Labels are mapped to `0/2/1`. The root script writes `ZuCo_SST_data/ssts_ZuCo.csv`. The folder script writes `output.csv`. The `all/` tree is not in this clone — only the resulting `ssts_ZuCo.csv` (400 rows) is.

Those 400 ids join to the averaged gaze tables to produce:

- `ZuCo_SST_data/combined_sst_et_standard.csv`
- `ZuCo_SST_data/combined_sst_et_min_max.csv`

The join itself is not a checked-in script. If you rebuild, merge on `sentence_id` / `id` and keep one sentiment column.

## Stage 4 — ZuCo train/valid/test

`ZuCo_SST_data/spilt.py` (typo preserved) runs `train_test_split` twice with `random_state=42`:

- 80% train (320)
- 10% valid (40)
- 10% test (40)

No stratification. That is why the valid slice is label-imbalanced. `model_ZuCo_SST.py` ignores these three files and uses stratified 5-fold on the full 400-row table instead.

## Stage 5 — full SST + projected gaze

`SST_data/convert_sst_to_et.py` tokenizes `stts_all_sentence_level.csv` with NLTK, keeps alphabetic tokens, and writes a word-level table with gaze columns set to `0`. That file is a *schema stub*, not a prediction.

Actual projected features live in `SST_data/combined_full_sst_et.csv` (11,853 rows). `gaze_prediction/data/convert_zuco_data.py` shows one scaling path from ZuCo word averages into the prediction schema (`nFix`, `FFD`, `GPT`, `TRT`, `GD` on a 0–100 min-max). The large `prediction_test_v2.csv` (191,971 word rows) is the bulky descendant of that path.

`SST_data/spilt.py` then splits `combined_full_sst_et.csv` 80/10/10 with `random_state=42` into `train_full_sst.csv`, `valid_full_sst.csv`, and `test_full_sst.csv`. Again, no stratification.

## Stage 6 — training

| Script | Reads | Writes |
| --- | --- | --- |
| `model_ZuCo_SST.py` | `ZuCo_SST_data/combined_sst_et_standard.csv` | stdout metrics |
| `model_full_SST.py` | `SST_data/{train,valid,test}_full_sst.csv` | `models/best_{model_type}_model.pth` |

Both tokenize with Hugging Face, wrap rows in `CustomDataset`, and train with Adam.

## Scripts vs. checked-in outputs

You do **not** need to re-run stages 0–5 to use the examples. The CSVs are already here. Re-run a stage only if you have the missing raw inputs (`ZuCo_mat_data/`, `all/*.txt`, or a gaze-prediction model).

`examples/schema_check.py` and `examples/split_audit.py` validate the checked-in products of this pipeline without requiring MATLAB or NLTK.
