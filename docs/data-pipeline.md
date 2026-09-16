# Data pipeline

The checked-in CSVs are the end of a multi-step conversion. This page walks
that path so you can tell which script owns which file.

```
ZuCo Task-1 .mat (12 subjects, not in git)
        │
        │  read_ZuCo_mat.py
        │  DataTransformer(task1, level=sentence, scaling=raw)
        ▼
ZuCo_et_csv_data/{1..12}_SR.csv
        │
        │  get_average_sentence_level.py
        │  zeros → NaN, mean across subjects, MinMax + Standard
        ▼
average_data.csv
min_max_scaled_average_data.csv
standard_scaled_average_data.csv
        │
        │  join on sentence_id with ssts_ZuCo.csv
        ▼
ZuCo_SST_data/combined_sst_et_{standard,min_max}.csv
        │
        │  ZuCo_SST_data/spilt.py   (80 / 10 / 10, seed 42)
        ▼
train.csv   valid.csv   test.csv


SST reviews (stts_all_sentence_level.csv)
        │
        │  convert_sst_to_et.py  (NLTK tokenize, gaze = 0)
        │  gaze_prediction models (external)
        ▼
SST_data/combined_full_sst_et.csv
        │
        │  SST_data/spilt.py
        ▼
train_full_sst.csv  valid_full_sst.csv  test_full_sst.csv
```

## 1. MATLAB structs → per-subject CSV

`utils_ZuCo.get_matfiles("task1")` lists 12 `.mat` files and asserts that
count. Each file is `sentenceData` from scipy.

`DataTransformer.__call__(subject)` then:

- Drops known bad trial ranges (subject/task specific).
- Aggregates word ET to sentence level, or writes one row per word.
- Replaces ±inf rows (`check_inf`).
- Fills NaNs (`zeros` / `mean` / `min`).
- Scales (`raw` / `min-max` / `mean-norm` / `standard`).

`read_ZuCo_mat.py` uses `scaling='raw'` and writes `et_csv_data/{i}_SR.csv`.
The checked-in copies were later moved to `ZuCo_et_csv_data/`. If you re-run
the script as-is it will create a *new* `et_csv_data/` folder at the repo
root. See [known issues](known-issues.md).

Word-level dumps follow the same subject loop with `level='word'`.

## 2. Cross-subject average

`get_average_sentence_level.py` currently points at `et_csv_data`. For the
checked-in files, run it with that folder renamed or edit the path to
`ZuCo_et_csv_data`.

Steps:

1. Read `{1..12}_SR.csv`.
2. Replace zeros with NaN on every column except `id`.
3. `concat` + `groupby(level=0).mean()` — this is an **index** mean, so
   subject 3's missing tail does not contribute to those rows.
4. Fit sklearn `MinMaxScaler` and `StandardScaler` on the averaged table.
5. Write the two scaled CSVs.

Word-level averaging (`ZuCo_et_csv_data/word/get_average.py`) copies
`id, Sent_ID, Word_ID, Word, WordLen` from subject 1 and averages the ET
columns. It does **not** convert zeros to NaN (that line is commented out).

## 3. Attach SST labels (ZuCo track)

`ZuCo_SST_data/save_SST_data.py` and `convert_full_SST.py` walk
`all/NEGATIVE|POSITIVE|NEUTRAL/*.txt`, map folder names to `{0,1,2}`, and
write `sentence_id, sentence, sentiment_label`. The `all/` tree is not in
git; `ssts_ZuCo.csv` is the saved result.

Someone then joined that table to the scaled averages on `sentence_id` /
`id` to produce `combined_sst_et_standard.csv` and
`combined_sst_et_min_max.csv`. There is no dedicated join script in the
repo — only the outputs.

`ZuCo_SST_data/spilt.py` (sic) creates the 320/40/40 files. The CV trainer
ignores them.

## 4. Full SST + projected gaze

`SST_data/stts_all_sentence_level.csv` is a headerless two-column dump of
the review text and a `POSITIVE|NEUTRAL|NEGATIVE` string.

`SST_data/convert_sst_to_et.py` tokenizes with `nltk.word_tokenize`, keeps
`[A-Za-z]+` only, and writes a word-level CSV with gaze columns hard-coded
to 0. That file is the alignment skeleton for predicted gaze.

`gaze_prediction/data/convert_zuco_data.py` shows the other direction:
take word averages, min-max scale nFix on its own range and the duration
features on a **shared** range, then emit
`sentence_id, word_id, word, nFix, FFD, GPT, TRT, GD` with values in 0–100.

The sentence-level `combined_full_sst_et.csv` is the pooled product of that
prediction stack. `SST_data/spilt.py` then does the 80/10/10 split with
`random_state=42` (not stratified — class rates still stay close).

## 5. What the trainers load

```
model_ZuCo_SST.py
    dataset_path = ZuCo_SST_data/combined_sst_et_standard.csv
    gaze columns = nFixations, FFD, GPT, TRT, GD
    tokenizer    = bert-base-uncased | roberta-base
    max_length   = 128

model_full_SST.py
    train/valid/test = SST_data/*_full_sst.csv
    gaze columns     = nFix, FFD, GPT, TRT, GD
    same tokenizers and max_length
```

Both wrap Hugging Face `Dataset.map(tokenize)` and then a custom
`torch.utils.data.Dataset` that also returns the five-float gaze tensor.

## 6. Re-running only the documented examples

You do **not** need MATLAB, NLTK, or predicted-gaze weights to run
`examples/`. Those scripts read the CSVs that are already here.
