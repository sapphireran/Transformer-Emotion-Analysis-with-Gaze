# Data pipeline

How the checked-in CSVs were produced. Scripts are one-shot personal converters; several still contain hardcoded folder names from an older layout.

```text
ZuCo .mat (not in repo)
        │
        ▼
read_ZuCo_mat.py  +  utils_ZuCo.DataTransformer
        │
        ▼
ZuCo_et_csv_data/{1..12}_SR.csv          (sentence, raw)
ZuCo_et_csv_data/word/{1..12}_SR.csv     (word, raw)
        │
        ▼
get_average_sentence_level.py
ZuCo_et_csv_data/word/get_average.py
        │
        ▼
average_data.csv
min_max_scaled_average_data.csv
standard_scaled_average_data.csv
word/word_averages_v2.csv
        │
        │   ZuCo_SST_data/all/{NEG,NEU,POS}/*.txt   (not in repo)
        │               │
        │               ▼
        │      convert_full_SST.py
        │               │
        │               ▼
        │        ssts_ZuCo.csv
        │               │
        └─────── join on sentence_id / id ──────────┘
                        │
                        ▼
        combined_sst_et_{standard,min_max}.csv
                        │
                        ▼
        ZuCo_SST_data/spilt.py  →  train/valid/test.csv
        model_ZuCo_SST.py       →  5-fold on the combined file


SST sentence dump (stts_all_sentence_level.csv)
        │
        ├─► convert_sst_to_et.py  →  sst_et_test.csv (zeros)
        │                              │
        │                              ▼
        │                    gaze predictor (weights not in repo)
        │                              │
        │                              ▼
        │              gaze_prediction/data/prediction_test_v2.csv
        │                              │
        │                              ▼
        │                    (offline sentence pool) 
        │
        └─► labels + pooled gaze → combined_full_sst_et.csv
                                      │
                                      ▼
                         SST_data/spilt.py
                                      │
                                      ▼
                    train_full_sst / valid_full_sst / test_full_sst
                                      │
                                      ▼
                               model_full_SST.py
```

## Stage 1 — MATLAB to CSV (`utils_ZuCo.py`, `read_ZuCo_mat.py`)

`get_matfiles(task, subdir)` builds `cwd + subdir + task` and expects **exactly 12** `.mat` files. Default `subdir` is a Windows path (`\\ZuCo_mat_data\\`). On this Linux clone you would pass something like `subdir='/ZuCo_mat_data/'` and `task='task1'`.

`DataTransformer(task, level, scaling, fillna)`:

- `task`: `task1` (sentiment / natural reading), `task2`, `task3` (task-specific reading). Sentiment work uses `task1`.
- `level`: `sentence` or `word`.
- `scaling`: `min-max` | `mean-norm` | `standard` | `raw`.
- `fillna`: `zeros` | `mean` | `min`.

`read_ZuCo_mat.py` instantiates `task1`, `level='sentence'`, `scaling='raw'`, `fillna='zeros'`, walks subjects `0..11`, writes `{i+1}_SR.csv` into `et_csv_data/` (again, an older folder name). Each frame gets an `id` column from the reset index.

Word-level dumps use the same transformer with `level='word'`. Those files already exist under `ZuCo_et_csv_data/word/`.

### Subject-specific row drops

`DataTransformer.__call__` skips MATLAB indices that ZuCo marked unusable:

| Task | Subject (0-based) | Skipped sentence indices |
| --- | ---: | --- |
| task1 | 2 | 150–249 and 399 |
| task2 | 6 | 0–49 |
| task2 | 11 | 50–99 |
| task3 | 3 | 178–224 |
| task3 | 7 | ≥ 359 |
| task3 | 11 | 270–313 and 362–406 |

Only the task1 / subject 2 hole is visible in this clone (`3_SR.csv` has 299 sentences).

`split_data` in `utils_ZuCo.py` halves each subject’s sentence table. Comment: control for order effects in Task 1. Nothing in the training scripts calls it.

## Stage 2 — Subject average and scale

`get_average_sentence_level.py`:

1. Read `et_csv_data/{1..12}_SR.csv`.
2. Replace `0` with NaN on every column except the first.
3. `concat` + `groupby(level=0).mean()` — this averages **by row position**, not by `id`. Subject 3’s 299 rows therefore mis-align if you re-run the script as written. The checked-in `average_data.csv` has 400 rows, so the published average was produced with a layout where every file had 400 rows, or subject 3 was omitted / padded. Re-running blindly is unsafe.
4. Fit `MinMaxScaler` and `StandardScaler` on the 400 × F matrix.
5. Write `min_max_scaled_average_data.csv` and `standard_scaled_average_data.csv`.

Word average (`ZuCo_et_csv_data/word/get_average.py`) concatenates twelve word files, means the seven numeric gaze columns by row position, and stitches `id, Sent_ID, Word_ID, Word, WordLen` from `1_SR.csv`. Same positional-mean caveat for subject 3.

## Stage 3 — Sentiment text

`convert_full_SST.py` (repo root) reads `ZuCo_SST_data/all/{NEGATIVE,POSITIVE,NEUTRAL}/*.txt`, maps folder → `{0,1,2}`, sorts by integer `sentence_id`, writes `ZuCo_SST_data/ssts_ZuCo.csv`.

`ZuCo_SST_data/save_SST_data.py` is an older sibling that writes `output.csv` and does not cast `sentence_id` to int.

The `all/` tree is not in git. `ssts_ZuCo.csv` is.

## Stage 4 — Join text to gaze

There is no dedicated join script in the snapshot. The combined files are the merge of:

- `ssts_ZuCo.csv` on `sentence_id`
- scaled averages on `id`

`examples/sentence_gaze_join.py` reconstructs that join from the files that *are* present and reports mismatched ids or label drift.

## Stage 5 — Hold-out splits (optional)

`ZuCo_SST_data/spilt.py` and `SST_data/spilt.py` are the same logic:

```python
train, rest = train_test_split(df, test_size=0.2, random_state=42)
valid, test = train_test_split(rest, test_size=0.5, random_state=42)
```

No `stratify=` argument, so class balance can drift in the 40-row ZuCo valid/test files. The 5-fold script avoids that by using `StratifiedKFold`.

## Stage 6 — Full SST word placeholders

`SST_data/convert_sst_to_et.py`:

1. `nltk.download('punkt')`
2. Read headerless SST CSV rows; column 0 is the sentence.
3. `nltk.word_tokenize`, keep `[A-Za-z]+` only.
4. Empty token list → one token `unknown`.
5. Write `sentence_id, word_id, word, nFix, FFD, GPT, TRT, GD` with gaze zeros.

That zero table is the input shape a gaze predictor is expected to fill. `prediction_test_v2.csv` is a filled copy with 191971 rows — same count as `sst_et_test.csv`.

`gaze_prediction/data/convert_zuco_data.py` is the other direction: ZuCo word averages → PROVO-like columns with 0–100 min-max scaling. It reads `training_data/word_averages_v2.csv`, which is not the current path (`ZuCo_et_csv_data/word/word_averages_v2.csv`).

## Stage 7 — Train

See `docs/model-architecture.md`. Device is `cuda` if `torch.cuda.is_available()` else `cpu`. No mixed precision, no gradient accumulation.

## Path leftovers (personal checklist)

| Script | Hardcoded path | Current location |
| --- | --- | --- |
| `read_ZuCo_mat.py` | `et_csv_data/` | `ZuCo_et_csv_data/` |
| `get_average_sentence_level.py` | `et_csv_data` | `ZuCo_et_csv_data/` |
| `utils_ZuCo.get_matfiles` | `\\ZuCo_mat_data\\` | not in repo |
| `convert_zuco_data.py` | `training_data/word_averages_v2.csv` | `ZuCo_et_csv_data/word/` |
| `model_full_SST.py` | `models/best_*.pth` | directory must be created |

The example scripts always use the **current** relative paths from the repo root.
