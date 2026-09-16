# Data pipeline

This is the path from ZuCo MATLAB structs and SST text files to the two
training tables. Folder names in the **scripts** and the **checked-in tree**
do not always match; see the callouts and [known-pitfalls.md](known-pitfalls.md).

```
ZuCo .mat (12 subjects, task 1)
        │  DataTransformer, level=sentence, scaling=raw
        ▼
et_csv_data/{1-12}_SR.csv          ← read_ZuCo_mat.py
        │  mean across subjects; 0 → NaN on numeric cols
        ▼
average_data.csv
        │  MinMaxScaler / StandardScaler
        ▼
min_max_scaled_average_data.csv
standard_scaled_average_data.csv
        │  join on sentence id
        ▼
ZuCo_SST_data/ssts_ZuCo.csv        ← convert_full_SST.py (txt folders)
        │
        ▼
combined_sst_et_{min_max,standard}.csv
        │  train_test_split 80 / 10 / 10, seed 42
        ▼
train.csv  valid.csv  test.csv     ← ZuCo_SST_data/spilt.py
        │
        └── model_ZuCo_SST.py actually ignores this split and
            re-does StratifiedKFold on combined_sst_et_standard.csv


SST sentences (string labels)
        │  convert_sst_to_et.py (NLTK tokens, gaze = 0)
        ▼
sst_et_test.csv                    ← word rows, placeholder gaze
        │  external gaze predictor (not in this repo)
        ▼
gaze_prediction/data/prediction_test_v2.csv
        │  aggregate word → sentence (not checked in as a script)
        ▼
combined_full_sst_et.csv
        │  80 / 10 / 10, seed 42
        ▼
train_full_sst.csv  valid_full_sst.csv  test_full_sst.csv
        │
        └── model_full_SST.py
```

## 1. MATLAB → per-subject CSV

`utils_ZuCo.py:get_matfiles` looks at:

```text
{cwd}\\ZuCo_mat_data\\{task}
```

The backslashes are a Windows leftover. On Linux you need either to patch
`subdir` or to run from a layout that still joins correctly with
`os.path.join` after you change the default. There must be **exactly 12**
`.mat` files.

`DataTransformer` then:

1. Loads `sentenceData` with `scipy.io.loadmat(..., squeeze_me=True, struct_as_record=False)`.
2. Walks each sentence's `word` list.
3. For **sentence level**, accumulates word gaze fields and divides by the
   number of words that had a non-all-zero feature vector (`nwords_fixated`).
   `SentLen` is `len(sent.word)`. `omissionRate` comes from the sentence
   struct itself.
4. Drops rows with `±inf`.
5. Fills remaining NaNs (`zeros`, `mean`, or `min`).
6. Optionally min-max / mean-normalizes / z-scores **per column**.

`read_ZuCo_mat.py` asks for `scaling='raw'` so the CSVs in
`ZuCo_et_csv_data/` are in **original units** (counts, milliseconds, pupil
size), not scaled.

### Bad-trial skips (task 1, sentence and word)

Hard-coded in `DataTransformer.__call__`. Indices are **0-based MATLAB
sentence indices**, not the later `id` column.

| Task | Subject index | File | What is skipped |
| --- | --- | --- | --- |
| task1 | 2 | `3_SR.csv` | sentences `150–249` inclusive, and `399` |
| task2 | 6 | — | first 50 sentences |
| task2 | 11 | — | sentences `50–99` |
| task3 | 3, 7, 11 | — | several ranges (relation-extraction dump) |

This repo’s checked-in CSVs are **task 1 / NR only**, so only the subject-2
gap is visible: `3_SR.csv` has 300 sentences and 5293 word rows; everyone
else has 400 sentences and 7129 word rows.

After skips, `id` is a **dense** index `0 … N-1` for that subject. Subject 3
therefore does **not** share a 1-to-1 `id` with the other subjects for
sentences after the hole. `get_average_sentence_level.py` still
`groupby(level=0).mean()` on the concatenated frames, which aligns on the
**reset index**, not on raw MATLAB index. Treat the averaged files as “mean
of whoever has a row at this table index,” and inspect `examples/subject_coverage.py`
before you assume 12-reader coverage on every line.

## 2. Average and scale (sentence)

`get_average_sentence_level.py`:

1. Reads `et_csv_data/{1–12}_SR.csv` (checked-in copy lives in
   `ZuCo_et_csv_data/` — copy or symlink if you re-run the script).
2. Replaces `0` with NaN on every column except the first (`id`). That is
   aggressive: `SFD` is often genuinely 0 when a word was fixated more than
   once, and `omissionRate` can be 0 when nobody skipped. The intent was
   “missing fixation,” which is closer to the word-level meaning of 0.
3. Means across the stacked rows by index.
4. Fits `MinMaxScaler` and `StandardScaler` on the averaged numeric block
   (excluding `id`).

The classification join uses the **standard** (z-score) table.

## 3. Word-level average

`ZuCo_et_csv_data/word/get_average.py` means
`nFixations, meanPupilSize, GD, TRT, FFD, SFD, GPT` across 12 files. Identity
columns (`Sent_ID`, `Word`, `WordLen`) are copied from `1_SR.csv`. Empty
`Word` becomes `'unknown'`; NaN gaze becomes 0.

v2 leaves zeros as zeros (the `replace(0, nan)` line is commented out), so
skipped words stay at 0 and pull the mean down. That is the file
`convert_zuco_data.py` expected under a different path
(`training_data/word_averages_v2.csv`).

## 4. SST labels for the 400 overlap

`convert_full_SST.py` walks `ZuCo_SST_data/all/{NEGATIVE,POSITIVE,NEUTRAL}/`
and parses the **filename stem** as `sentence_id`. Mapping:

```python
NEGATIVE → 0
NEUTRAL  → 1
POSITIVE → 2
```

Output: `ZuCo_SST_data/ssts_ZuCo.csv`.

The `all/` text folders are not in git (only the CSV and a zip). Re-running
the converter needs that tree.

Join key: `sentence_id` ↔ averaged gaze `id`. After the join you get
`combined_sst_et_standard.csv` (400 rows). How that join was performed is
not a remaining script; the combined files are the source of truth.

## 5. ZuCo hold-out split (optional)

`ZuCo_SST_data/spilt.py` does sklearn `train_test_split` twice (test_size
0.2, then 0.5) with `random_state=42`. It is **not** stratified in the
script. `model_ZuCo_SST.py` does **not** use these files; it folds
`combined_sst_et_standard.csv` itself.

## 6. Full SST + predicted gaze

`SST_data/stts_all_sentence_level.csv` has no header. Each row is
`sentence, LABEL` with `LABEL ∈ {POSITIVE, NEGATIVE, NEUTRAL}`.

`convert_sst_to_et.py` tokenizes with `nltk.word_tokenize`, keeps
`^[A-Za-z]+$` tokens only, and writes word rows with gaze **zeros**. That
placeholder file is the input shape the predictor fills.

`gaze_prediction/data/convert_zuco_data.py` is a **training-data** formatter
for a predictor: it min-max scales `nFixations` alone and scales
`FFD, GPT, TRT, GD` together onto 0–100. It is not the predictor.

`SST_data/spilt.py` then splits `combined_full_sst_et.csv` the same 80/10/10
way (also not stratified). `model_full_SST.py` uses those three files.

## 7. Column rename at the track boundary

| Human (ZuCo) | Predicted (full SST) |
| --- | --- |
| `nFixations` | `nFix` |
| `FFD, GPT, TRT, GD` | same names |
| also `SFD, omissionRate, meanPupilSize` | absent |
| `sentiment_label` 0/1/2 | same |

`EyeTrackingModel` does not care about names, only the **order** of the five
columns selected in each training script:

```text
ZuCo:   nFixations, FFD, GPT, TRT, GD
full:   nFix,       FFD, GPT, TRT, GD
```

That order is **not** the same as the CSV left-to-right order on the full
SST files (`nFix, GD, TRT, FFD, GPT`). The pandas column list in
`model_full_SST.py` reorders them to match the ZuCo fusion order.
