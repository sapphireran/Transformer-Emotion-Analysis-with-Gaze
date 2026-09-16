# Datasets in this repository

Every numeric claim below was counted from the checked-in CSVs. Re-run
`python3 examples/inspect_datasets.py` if a file is replaced.

There are three families of tables:

1. **Measured ZuCo gaze** — 12 subjects, 400 sentiment-reading sentences.
2. **ZuCo ∩ SST labels** — the same 400 sentences with ternary sentiment.
3. **Full SST + projected gaze** — ~11.8k reviews, gaze *not* measured.

A fourth family under `gaze_prediction/data/` holds word-level predicted
gaze and a PROVO extract used as an auxiliary reading-time corpus.

## Label convention

Joined experiment tables use integers:

| `sentiment_label` | Meaning | Typical source folder / SST tag |
| ---: | --- | --- |
| 0 | Negative | `NEGATIVE` |
| 1 | Neutral | `NEUTRAL` |
| 2 | Positive | `POSITIVE` |

`SST_data/stts_all_sentence_level.csv` is the odd one out: it has **no
header** and stores the string labels `NEGATIVE` / `NEUTRAL` / `POSITIVE`
in the second column.

---

## 1. Measured gaze — sentence level

Directory: `ZuCo_et_csv_data/`

| File | Rows | Role |
| --- | ---: | --- |
| `1_SR.csv` … `12_SR.csv` | 400 each | One subject, sentiment reading (SR) |
| `average_data.csv` | 400 | Mean across subjects (raw units) |
| `min_max_scaled_average_data.csv` | 400 | Min-max of the subject mean |
| `standard_scaled_average_data.csv` | 400 | Z-score of the subject mean |

Shared columns:

```text
id, SentLen, omissionRate, nFixations, meanPupilSize, GD, TRT, FFD, SFD, GPT
```

- `id` is the sentence index **0 … 399**. It is the join key to
  `ZuCo_SST_data/ssts_ZuCo.csv`.
- `SentLen` is the number of words ZuCo stored for that sentence. It is
  constant across subjects for a given `id` (the text does not change).
- The remaining columns are **subject-specific** in `*_SR.csv` and
  **subject-averaged** in `average_data.csv`.

`get_average_sentence_level.py` is the script that built the scaled
averages. The copy in the repo still points at a folder named
`et_csv_data`; the checked-in files live in `ZuCo_et_csv_data/`. See
[known-issues.md](known-issues.md).

### Subject files

`read_ZuCo_mat.py` writes `{subject}_SR.csv` for subjects `1 … 12` from
ZuCo Task 1 (normal / sentiment reading of movie reviews). The transformer
in `utils_ZuCo.py` already skips known bad sentence ranges for a few
subject–task pairs. For Task 1 the special case is subject index 2
(file `3_SR.csv`), which drops a 100-sentence block plus one extra row in
the MATLAB dump. The checked-in CSVs are already aligned to 400 rows, so
the join to SST does not re-apply those filters.

---

## 2. Measured gaze — word level

Directory: `ZuCo_et_csv_data/word/`

| File | Rows | Role |
| --- | ---: | --- |
| `1_SR.csv` … `12_SR.csv` | 7,129 each | Per-word gaze for one subject |
| `word_averages.csv` | 7,129 | Older subject mean |
| `word_averages_v2.csv` | 7,129 | Current subject mean (zeros kept) |

Columns:

```text
id, Sent_ID, Word_ID, Word, nFixations, meanPupilSize, GD, TRT, FFD, SFD, GPT, WordLen
```

- `id` is a dense row index, **not** the sentence id.
- `Sent_ID` looks like `0_NR`, `1_NR`, … — sentence index plus the ZuCo
  task tag `NR` (normal reading). Task 3 would have used `_TSR`.
- `Word_ID` is the 0-based token position inside the sentence.
- `Word` is a lightly cleaned surface form (leading capital stripped on
  the first token; punctuation stripped). Empty tokens are stored as
  `unknown` in the averaged file.
- A row of **all-zero** gaze with a real `Word` is a skip: the subject
  never fixated that token. Those zeros are load-bearing. Averaging
  scripts that blindly turn `0` into `NaN` change the meaning of a skip.

`word/get_average.py` concatenates the 12 subject files and takes the
mean of the numeric gaze columns **by row position**, then glues back
`id, Sent_ID, Word_ID, Word, WordLen` from subject 1. That only works
because every subject file has the same 7,129-row alignment.

---

## 3. ZuCo ∩ SST experiment tables

Directory: `ZuCo_SST_data/`

| File | Rows | Labels (0 / 1 / 2) | Role |
| --- | ---: | --- | --- |
| `ssts_ZuCo.csv` | 400 | 123 / 137 / 140 | Text + label only |
| `combined_sst_et_standard.csv` | 400 | 123 / 137 / 140 | Text + z-scored gaze |
| `combined_sst_et_min_max.csv` | 400 | 123 / 137 / 140 | Text + min-max gaze |
| `train.csv` | 320 | 103 / 107 / 110 | 80% split (seed 42) |
| `valid.csv` | 40 | 7 / 14 / 19 | 10% split |
| `test.csv` | 40 | 13 / 16 / 11 | 10% split |

`combined_*` columns:

```text
sentence_id, sentence, sentiment_label,
omissionRate, nFixations, meanPupilSize, GD, TRT, FFD, SFD, GPT
```

`model_ZuCo_SST.py` does **not** use `train.csv` / `valid.csv` / `test.csv`.
It reloads `combined_sst_et_standard.csv` and runs `StratifiedKFold(n_splits=5)`.
The 80/10/10 files exist for inspection and for anyone who wants a fixed
hold-out instead of CV. The hold-out is small: 40 valid rows, and the
valid label counts are visibly unbalanced (7 negatives vs 19 positives).
That is why the trainer prefers folds.

`convert_full_SST.py` / `save_SST_data.py` rebuild `ssts_ZuCo.csv` from a
local folder `all/{NEGATIVE,POSITIVE,NEUTRAL}/*.txt`. Those `.txt` files
are not in the clone; the CSV is.

---

## 4. Full SST + projected gaze

Directory: `SST_data/`

| File | Rows | Labels (0 / 1 / 2) | Role |
| --- | ---: | --- | --- |
| `stts_all_sentence_level.csv` | 11,852 | 4649 / 2241 / 4962 * | Raw SST phrases, no header |
| `combined_full_sst_et.csv` | 11,853 | 4649 / 2241 / 4963 | Text + 5 projected gaze features |
| `train_full_sst.csv` | 9,482 | 3710 / 1833 / 3939 | 80% |
| `valid_full_sst.csv` | 1,185 | 476 / 209 / 500 | 10% |
| `test_full_sst.csv` | 1,186 | 463 / 199 / 524 | 10% |
| `sst_et_test.csv` | 191,971 | — | Word skeleton (gaze columns zeroed) |
| `convert_sst_to_et.py` | — | — | Builds the word skeleton |
| `spilt.py` | — | — | 80/10/10 split (filename is historical) |

\* String labels in `stts_all_sentence_level.csv`. Neutral/negative match
the combined table; positive is off by one (4962 vs 4963) because the
headerless file has no extra row.

`combined_full_sst_et.csv` columns:

```text
sentence_id, sentence, sentiment_label, nFix, GD, TRT, FFD, GPT
```

Compared with the ZuCo tables this set:

- Drops `omissionRate`, `meanPupilSize`, `SFD`.
- Renames `nFixations` → `nFix`.
- Uses **projected** values. They are not a 12-subject mean. Many rows
  are negative, which is normal for a standardized or model-emitted
  feature and impossible for a raw fixation count.

`model_full_SST.py` reads the three split files and expects exactly those
five gaze columns.

`sst_et_test.csv` is the token table used to *request* word-level
predictions: one row per alphabetic token, gaze columns filled with `0`.
`gaze_prediction/data/prediction_test_v2.csv` is the filled-in counterpart
(same 191,971 rows, non-zero predicted features).

---

## 5. Gaze prediction / auxiliary tables

Directory: `gaze_prediction/data/`

| File | Rows | Columns of interest | Role |
| --- | ---: | --- | --- |
| `prediction_test.csv` | 1,751 | `nFix, FFD, GPT, TRT, GD` | Small predicted-gaze sample |
| `prediction_test_v2.csv` | 191,971 | same | Predicted gaze for full SST tokens |
| `provo.csv` | 2,659 | `nFix, FFD, GPT, TRT, fixProp` | PROVO extract (`fixProp` instead of `GD`) |
| `convert_zuco_data.py` | — | — | Min-max to a 0–100 display scale |

`convert_zuco_data.py` rescales `nFixations` on its own min/max and the
four duration features on a **shared** min/max, then writes the
`sentence_id, word_id, word, nFix, FFD, GPT, TRT, GD` layout. That 0–100
scale is for the prediction-side format, not for `model_ZuCo_SST.py`.

---

## 6. Join keys you can trust

| Left | Right | Key | Notes |
| --- | --- | --- | --- |
| `ZuCo_et_csv_data/average_data.csv` | `ZuCo_SST_data/ssts_ZuCo.csv` | `id` = `sentence_id` | 400 = 400 |
| `ZuCo_et_csv_data/word/word_averages_v2.csv` | sentence tables | `Sent_ID` prefix | `"{id}_NR"` |
| `SST_data/combined_full_sst_et.csv` | split CSVs | `sentence_id` | Splits are a partition |
| `SST_data/sst_et_test.csv` | `prediction_test_v2.csv` | `(sentence_id, word_id)` | Same length |

`examples/inspect_datasets.py` asserts the 400-row join and the 80/10/10
partition. `examples/split_balance.py` prints label percentages so a
re-split with a different seed is obvious.

---

## 7. What is *not* in the clone

- ZuCo `.mat` files (`ZuCo_mat_data/`).
- The `all/{NEGATIVE,NEUTRAL,POSITIVE}/*.txt` dump.
- Trained `*.pth` checkpoints (`models/` is gitignored).
- EEG channels. `utils_ZuCo.py` can see them in the MATLAB structs; the
  CSVs kept here are eye-tracking only.

If you only want to study fusion on real gaze, `ZuCo_SST_data/` plus
`ZuCo_et_csv_data/` are enough. Full-SST training additionally needs
`SST_data/train_full_sst.csv` and its siblings.
