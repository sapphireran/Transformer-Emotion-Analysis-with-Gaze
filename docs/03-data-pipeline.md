# Data pipeline, file by file

Two pipelines share a label scheme and a five-feature gaze vector, then
diverge. This page is the map I use when I cannot remember whether
`model_ZuCo_SST.py` reads the 80/10/10 split or does its own CV (it
does its own CV).

```
                    ZuCo .mat (not in this clone)
                              │
                              ▼
                     utils_ZuCo.DataTransformer
                     read_ZuCo_mat.py
                              │
                              ▼
              ZuCo_et_csv_data/{1-12}_SR.csv          (sentence, raw)
              ZuCo_et_csv_data/word/{1-12}_SR.csv     (word, raw)
                              │
                              ▼
              get_average_sentence_level.py
              word/get_average.py
                              │
              ┌───────────────┴────────────────┐
              ▼                                ▼
     average_data.csv                  word_averages_v2.csv
     min_max / standard scaled         7,129 word rows
              │
              ▼
     join with ssts_ZuCo.csv labels
              │
              ▼
     combined_sst_et_{standard,min_max}.csv     400 rows
              │
              ├─► model_ZuCo_SST.py   (StratifiedKFold on this file)
              └─► ZuCo_SST_data/spilt.py → train/valid/test.csv
```

```
     SST sentences (stts_all_sentence_level.csv)
              │
              ├─► convert_sst_to_et.py  (tokenize, gaze columns = 0)
              │
              ▼
     predicted / transferred gaze
     (gaze_prediction/data/prediction_test*.csv, convert_zuco_data.py)
              │
              ▼
     combined_full_sst_et.csv                   11,853 rows
              │
              ▼
     SST_data/spilt.py  (80 / 10 / 10, seed 42)
              │
              ▼
     train_full_sst.csv / valid_full_sst.csv / test_full_sst.csv
              │
              ▼
     model_full_SST.py
```

---

## Stage 1 — MATLAB to per-subject CSV

`utils_ZuCo.py` is the only file that understands the ZuCo MATLAB
struct. `DataTransformer(task, level, scaling, fillna)`:

- `task`: `'task1'` (sentiment / normal reading), `'task2'`, `'task3'`.
  This clone only kept task-1 derivatives.
- `level`: `'sentence'` or `'word'`.
- `scaling`: `'min-max' | 'mean-norm' | 'standard' | 'raw'`.
- `fillna`: `'zeros' | 'mean' | 'min'`.

`read_ZuCo_mat.py` builds a **raw** sentence-level transformer and
writes `et_csv_data/{i+1}_SR.csv`. The checked-in files live under
`ZuCo_et_csv_data/` instead. If you re-run the script on a machine that
has the `.mat` files, either create `et_csv_data/` or change the path.

Subject-level landmines, copied from the `if` ladder in
`DataTransformer.__call__` (task 1, sentence level):

- Subject 2 (file `3_SR.csv`) drops sentences 150–249 and 399.
- Other task-1 subjects keep all rows.

Those skips exist because the original MATLAB dump has empty or
corrupt blocks. Do not "fix" them by padding zeros; that would invent
gaze.

Default `get_matfiles` joins `os.getcwd() + '\\ZuCo_mat_data\\' + task`.
The backslashes are a Windows leftover. On Linux you want
`os.path.join(os.getcwd(), 'ZuCo_mat_data', task)`.

### Sentence-level aggregation rule

For each kept sentence the transformer:

1. Walks `sent.word`.
2. Strips punctuation with `re.sub('[^\w\s]', '', word.content)`.
3. Lowercases only `j == 0` (sentence-initial).
4. Reads `nFixations, meanPupilSize, GD, TRT, FFD, SFD, GPT` if the
   attribute exists and is not an `ndarray` (ZuCo stores missing as
   empty arrays).
5. Sums those word vectors, then **divides by `nwords_fixated`** — the
   count of words that were not all-zero.
6. Writes `SentLen = len(sent.word)` and `omissionRate` from the
   sentence struct.

So a 22-word sentence with 5 skipped words is averaged over 17 words,
not 22. That is why sentence-level nFixations lives around 1.7, not
around 0.something.

`check_inf` drops any sentence row that still contains `±inf` after
that pass.

### Word-level dump

Same walk, one row per word, extra columns `Sent_ID`, `Word_ID`,
`Word`, `WordLen`. `Sent_ID` looks like `0_NR` on task 1/2 and `0_TSR`
on task 3. `word/get_average.py` averages the seven numeric gaze
columns across the 12 subject files by row index (not by word string)
and fills remaining NaNs with 0, empty words with `unknown`.

---

## Stage 2 — 12-reader mean and scaling

`get_average_sentence_level.py` is the sentence-level counterpart:

1. Read `et_csv_data/{1-12}_SR.csv`.
2. Replace `0` with `NaN` on every column except the first (`id`).
3. `concat` + `groupby(level=0).mean()` — average aligned by row
   position, which is correct only because every file is the same 400
   sentences in the same order (after the subject-2 skip, subject 2 is
   *shorter* — this is a real footgun; the checked-in
   `average_data.csv` has 400 rows, so the version that produced it
   either padded or used a different skip).
4. `MinMaxScaler` → `min_max_scaled_average_data.csv`.
5. `StandardScaler` → `standard_scaled_average_data.csv`.

I treat the checked-in `average_data.csv` / scaled files as the
canonical 400-row tables and I do not re-run this script in the
current environment (it points at a folder that is not the one in
git).

---

## Stage 3 — attach SST labels (ZuCo 400)

`convert_full_SST.py` walks `ZuCo_SST_data/all/{NEGATIVE,POSITIVE,NEUTRAL}/*.txt`,
maps folder name → `{0, 1, 2}`, sorts by the numeric filename, writes
`ZuCo_SST_data/ssts_ZuCo.csv` (`sentence_id, sentence, sentiment_label`).

`ZuCo_SST_data/save_SST_data.py` is the same idea with relative `all/`
and `output.csv`. I keep both; `ssts_ZuCo.csv` is the one that is
actually present (400 labeled sentences).

Those labels are then joined to the scaled gaze tables to produce:

- `combined_sst_et_standard.csv` — **this is `dataset_path` in
  `model_ZuCo_SST.py`**
- `combined_sst_et_min_max.csv` — same sentences, different scale

Join key is sentence order / `sentence_id`. Sentence 0 in the gaze
file is sentence 0 in `ssts_ZuCo.csv`:

> Presents a good case while failing to provide a reason for us to
> care beyond the very basic dictums of human decency. → label 1
> (neutral)

---

## Stage 4 — optional 80/10/10 on ZuCo

`ZuCo_SST_data/spilt.py` reads `combined_sst_et_standard.csv` and
writes `train.csv` (320), `valid.csv` (40), `test.csv` (40) with
`train_test_split(..., test_size=0.2, random_state=42)` then a 50/50
split of the remainder.

**`model_ZuCo_SST.py` ignores these files.** It reads the combined
table and runs `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`.

I still use the 80/10/10 files when I want a fixed tiny validation
set without waiting for five folds. Label counts there are *not*
stratified on the 40-row slices (valid is 7/14/19). The CV split is
the one that respects class balance.

---

## Stage 5 — full SST + predicted gaze

`SST_data/stts_all_sentence_level.csv` is two columns without a header:
`sentence, POSITIVE|NEGATIVE|NEUTRAL` (quoted). 11,853 rows.

`SST_data/convert_sst_to_et.py` NLTK-tokenizes each sentence, drops
tokens that are not `[A-Za-z]+`, and writes a word-level table with
**all gaze values = 0**. That file is a template for a predictor, not
training data.

`gaze_prediction/data/convert_zuco_data.py` is the other direction:
take a ZuCo-like word table, scale nFixations to 0–100 from its own
min/max, scale FFD/GPT/TRT/GD onto one shared min/max, write
`sentence_id, word_id, word, nFix, FFD, GPT, TRT, GD`.

`SST_data/combined_full_sst_et.csv` is the sentence-level table the
full-SST model trains on. Five gaze columns are already standardized
(mean ≈ 0, std ≈ 1 on the 11,853 rows). How each row's vector was
predicted is **not** fully scripted in this clone; `gaze_prediction/`
holds intermediate word-level predictions (`prediction_test.csv` is
1,751 rows starting at `sentence_id=300`; `prediction_test_v2.csv` is
~11 MB). I treat the combined file as a snapshot and do not pretend
I can regenerate it from the scripts alone.

`SST_data/spilt.py` then makes the 9,482 / 1,185 / 1,186 split that
`model_full_SST.py` hard-codes.

---

## What each training script reads

| Script | Path | Gaze columns | Labels |
| --- | --- | --- | --- |
| `model_ZuCo_SST.py` | `ZuCo_SST_data/combined_sst_et_standard.csv` | `nFixations, FFD, GPT, TRT, GD` | `sentiment_label` |
| `model_full_SST.py` | `SST_data/{train,valid,test}_full_sst.csv` | `nFix, FFD, GPT, TRT, GD` | `sentiment_label` |

Both tokenize `sentence` with `max_length=128`, `padding='max_length'`.
A 43-word ZuCo review fits easily. The long tail of SST (max 56 words)
also fits; 128 is not a tight constraint here.

## Tokenizer mismatch to watch

`load_dataset` in the full-SST script builds a Hugging Face `Dataset`,
maps the tokenizer, converts back to pandas, then `concat` with the
gaze frame **by axis=1 / row position**. That is safe only if
`Dataset.from_pandas` + `map` + `to_pandas` preserves row order, which
it does when you do not shuffle. Do not insert a `.shuffle()` between
those calls.

The ZuCo script tokenizes the *entire* 400-row frame once, then slices
with `iloc[train_index]` and `eye_tracking_features.iloc[train_index]`.
The two slices must stay aligned. They do, as long as nobody resets
the gaze frame index independently.

## Files I do not treat as sources of truth

| File | Why |
| --- | --- |
| `ZuCo_SST_data/ZuCo_SST_data.zip` | Duplicate archive of the folder |
| `ZuCo_et_csv_data/word/word_averages.csv` | Older average; v2 is the one `convert_zuco_data.py` comments refer to |
| `result/*.png` | Diagnostic plots, not inputs |
| `gaze_prediction/data/prediction_test_v2.csv` | Large intermediate; not read by either training script |
