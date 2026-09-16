# Pipeline

This is the order the original scripts expect. Several steps need folders that
are not in git (see the bottom of [datasets.md](datasets.md)). The checked-in
CSVs are the outputs of those steps, which is why the trainers can still run
without MATLAB.

## 1. ZuCo MATLAB → per-subject CSV

```
ZuCo_mat_data/task1/*.mat
        │
        ▼
read_ZuCo_mat.py
  uses utils_ZuCo.DataTransformer(task="task1", level="sentence",
                                  scaling="raw", fillna="zeros")
        │
        ▼
et_csv_data/{1-12}_SR.csv      # script name; repo folder is ZuCo_et_csv_data/
```

Word-level exports use the same transformer with `level="word"` and land in
`ZuCo_et_csv_data/word/`. `DataTransformer` skips known bad sentence ranges
for a few subject/task pairs. For task 1 that is subject index 2 (file
`3_SR.csv`), which is why that file has 299 sentence rows instead of 400.

Windows-style default `subdir='\\ZuCo_mat_data\\'` in `get_matfiles()` will
not resolve on this Linux laptop unless you pass a real path.

## 2. Average readers, then scale

```
ZuCo_et_csv_data/{1-12}_SR.csv
        │
        ▼
get_average_sentence_level.py
  0 → NaN on every column except the first
  mean across the 12 frames (aligned by row index)
  MinMaxScaler and StandardScaler on the ET columns
        │
        ├─ min_max_scaled_average_data.csv
        └─ standard_scaled_average_data.csv
```

The script's `folder_path = 'et_csv_data'` does not match the checked-in
folder name `ZuCo_et_csv_data`. Point it at the real directory before
re-running.

Word analogue: `ZuCo_et_csv_data/word/get_average.py` → `word_averages_v2.csv`.

## 3. Attach sentiment text

Two nearly identical scrapers walk
`ZuCo_SST_data/all/{NEGATIVE,POSITIVE,NEUTRAL}/*.txt`:

| Script | Output |
| --- | --- |
| `convert_full_SST.py` | `ZuCo_SST_data/ssts_ZuCo.csv` |
| `ZuCo_SST_data/save_SST_data.py` | `output.csv` in that folder |

Join key is `sentence_id` from the txt filename. Label map is
`NEGATIVE=0, NEUTRAL=1, POSITIVE=2`.

The combined tables already in git (`combined_sst_et_standard.csv`,
`combined_sst_et_min_max.csv`) are that text file plus the scaled sentence ET
columns. There is no separate “join.py” in the repo; the join was done once
and the result was committed.

## 4. Optional 80/10/10 on ZuCo

`ZuCo_SST_data/spilt.py` (the filename is a typo) reads
`combined_sst_et_standard.csv` and writes `train.csv` / `valid.csv` /
`test.csv` with `random_state=42`. `model_ZuCo_SST.py` does **not** use those
files. It cross-validates the full 400-row table instead.

## 5. Train on real gaze

```
python model_ZuCo_SST.py
```

Loads `combined_sst_et_standard.csv`, tokenizes `sentence`, takes
`nFixations, FFD, GPT, TRT, GD`, runs 5-fold stratified CV, prints mean
weighted metrics. See [architecture.md](architecture.md).

## 6. Full SST text → word skeleton

```
SST_data/stts_all_sentence_level.csv
        │
        ▼
SST_data/convert_sst_to_et.py
  nltk.word_tokenize
  keep [A-Za-z]+ only
  write zeros for nFix, FFD, GPT, TRT, GD
        │
        ▼
SST_data/sst_et_test.csv          # 191,971 word rows
```

## 7. Predict gaze, pool to sentences

`gaze_prediction/data/convert_zuco_data.py` shows the schema the prediction
side wants: `sentence_id, word_id, word, nFix, FFD, GPT, TRT, GD` with values
scaled to roughly 0–100. `prediction_test_v2.csv` has the same row count as
the SST word skeleton.

Somebody then pooled those word rows into
`SST_data/combined_full_sst_et.csv` (11,853 sentences, five z-scored
features). That pooling script is not in git.

## 8. Split full SST and train

```
SST_data/spilt.py
  combined_full_sst_et.csv
  train_test_split test_size=0.2, then 0.5, random_state=42
        │
        ├─ train_full_sst.csv
        ├─ valid_full_sst.csv
        └─ test_full_sst.csv
        │
        ▼
python model_full_SST.py
  5 epochs, batch 256, save best valid-accuracy checkpoint
```

## Personal mental model

```
recorded gaze (ZuCo, n=400)     predicted gaze (SST, n≈12k)
        │                                │
        │ weak linear link               │ almost collinear features
        │ to sentiment                   │ weak linear link to sentiment
        ▼                                ▼
   concat onto RoBERTa              concat onto RoBERTa
   5-fold CV, 20 epochs             5 epochs, 80/10/10
```

The shared hypothesis is “a 16-d glance at how the sentence was (or would be)
read can help a transformer decide the review’s polarity.” The shared
implementation is one linear sidecar, not a word-aligned gaze encoder.

## Helper package used by examples only

```
tea_gaze.paths   →  where the CSVs live
tea_gaze.io      →  load + schema check
tea_gaze.schema  →  column lists and label names
tea_gaze.features / baselines / models
        │
        ▼
examples/0*.py
```

The historical trainers do not import `tea_gaze`. That is on purpose so a
docs/examples branch cannot silently change old results.
