# Datasets

All counts below were computed from the CSVs in this checkout. Scripts
that regenerate a file are listed next to it. Several of those scripts
expect extra inputs (ZuCo `.mat` files, an `all/NEGATIVE|POSITIVE|NEUTRAL`
folder) that are **not** in the repo; the derived CSVs are.

Related examples: `examples/01_inspect_datasets.py`,
`examples/02_label_and_length_profile.py`,
`examples/07_split_leakage_check.py`.

## Label convention

Processed tables use integers:

| `sentiment_label` | Class | Origin |
| ---: | --- | --- |
| `0` | Negative | `convert_full_SST.py` / `ZuCo_SST_data/save_SST_data.py` map `NEGATIVE → 0` |
| `1` | Neutral | `NEUTRAL → 1` |
| `2` | Positive | `POSITIVE → 2` |

`SST_data/stts_all_sentence_level.csv` is the odd one out: two unlabeled
columns `(sentence, POSITIVE|NEGATIVE|NEUTRAL)` and no header.

---

## 1. Full SST + sentence-level gaze

Directory: `SST_data/`

### `stts_all_sentence_level.csv`

- **Rows:** 11,853 sentences (the file has no header; a header-aware
  reader will report 11,852).
- **Columns:** `sentence`, `sentiment` as strings.
- **Role:** raw-ish SST dump used by `SST_data/convert_sst_to_et.py` to
  emit a word-level placeholder table.

### `combined_full_sst_et.csv`

- **Rows:** 11,853
- **Columns:** `sentence_id`, `sentence`, `sentiment_label`, `nFix`,
  `GD`, `TRT`, `FFD`, `GPT`
- **Labels:** 4,649 negative / 2,241 neutral / 4,963 positive
- **Gaze:** already **standardized** across the full table (feature
  means ≈ 0, stds ≈ 1). Extremes are real: `GD` goes down to about
  −12.6. Neutral is the minority class (~19%).

This is the parent table for the official training split.

### `train_full_sst.csv` / `valid_full_sst.csv` / `test_full_sst.csv`

Produced by `SST_data/spilt.py` (filename is a typo of "split"):

```
train_test_split(df, test_size=0.2, random_state=42)
train_test_split(valid_test, test_size=0.5, random_state=42)
```

| Split | Rows | Neg | Neu | Pos |
| --- | ---: | ---: | ---: | ---: |
| train | 9,482 | 3,710 | 1,833 | 3,939 |
| valid | 1,185 | 476 | 209 | 500 |
| test | 1,186 | 463 | 199 | 524 |

`sentence_id` values are unique inside each split and do not overlap
across splits (checked by `examples/07_split_leakage_check.py`). The
split is **not** stratified — you can see the positive class drift
slightly upward in the test set (524 / 1,186 ≈ 44% vs 42% in train).

`model_full_SST.py` reads these three files directly.

### `sst_et_test.csv`

- **Rows:** 191,971 word tokens
- **Columns:** `sentence_id`, `word_id`, `word`, `nFix`, `FFD`, `GPT`,
  `TRT`, `GD`
- **Gaze values:** all zeros. This is a **schema scaffold** produced by
  `SST_data/convert_sst_to_et.py`: NLTK `word_tokenize`, keep
  `[A-Za-z]+` only, write a row per token. The filled-in predictions
  are `gaze_prediction/data/prediction_test_v2.csv` (same row count).

---

## 2. ZuCo SST (measured gaze)

Directory: `ZuCo_SST_data/`

ZuCo task 1 (sentiment reading) used SST movie-review sentences. This
repo keeps the 400-sentence overlap with recorded eye tracking.

### `ssts_ZuCo.csv`

- **Rows:** 400
- **Columns:** `sentence_id`, `sentence`, `sentiment_label`
- **Producer:** `convert_full_SST.py` (expects
  `ZuCo_SST_data/all/{NEGATIVE,POSITIVE,NEUTRAL}/*.txt`, not vendored)

### `combined_sst_et_standard.csv` and `combined_sst_et_min_max.csv`

Same 400 sentences joined to subject-averaged sentence-level gaze.

| Column | Meaning |
| --- | --- |
| `omissionRate` | Fraction of words with no fixation |
| `nFixations` | Mean fixation count on fixated words |
| `meanPupilSize` | Mean pupil size |
| `GD` / `TRT` / `FFD` / `SFD` / `GPT` | Timing features (see features doc) |

- **Standard file:** z-scored per feature. Used by `model_ZuCo_SST.py`.
- **Min-max file:** each feature scaled to `[0, 1]`.
- **Labels:** 123 / 137 / 140 — much more balanced than full SST.

### `train.csv` / `valid.csv` / `test.csv`

320 / 40 / 40 from `ZuCo_SST_data/spilt.py` with the same 80/10/10
logic. **The training script does not use these files.** It reloads the
full 400-row standard table and runs `StratifiedKFold(n_splits=5)`.
Treat the 320/40/40 CSVs as an optional holdout if you want to compare
protocols.

`ZuCo_SST_data/save_SST_data.py` is an earlier variant of
`convert_full_SST.py` that writes `output.csv`.

---

## 3. Per-subject ZuCo eye tracking

Directory: `ZuCo_et_csv_data/`

### Sentence level: `{1–12}_SR.csv`

- **Rows:** 400 per subject **except subject 3 (299 rows)**
- **Columns:** `id`, `SentLen`, `omissionRate`, `nFixations`,
  `meanPupilSize`, `GD`, `TRT`, `FFD`, `SFD`, `GPT`
- **Units:** raw milliseconds / counts / rates, **not** standardized
- **Producer:** `read_ZuCo_mat.py` → `DataTransformer('task1',
  level='sentence', scaling='raw', fillna='zeros')`
- **Subject 3 caveat:** filename `3_SR.csv` is `DataTransformer`
  subject index 2. That path drops original sentences 150–249 and 399
  (101 rows), then writes a **new** `id` 0..298. Rows 0–149 are still
  the original sentences; rows 150–298 are original 250–398. Averaging
  the twelve files on row index therefore mixes sentences from id 150
  onward. Details and a remap: `examples/10_zuco_index_alignment.py`
  and [known_issues.md](known_issues.md).

`average_data.csv` is the 12-subject mean of those tables.
`standard_scaled_average_data.csv` and `min_max_scaled_average_data.csv`
are that average after `StandardScaler` / `MinMaxScaler`
(`get_average_sentence_level.py`, which still refers to a folder named
`et_csv_data` — see [known_issues.md](known_issues.md)).

`examples/04_subject_variability.py` measures how much the twelve
readers disagree on the same sentence.

### Word level: `word/{1–12}_SR.csv` and `word/word_averages_v2.csv`

- **Rows:** 7,129 word observations
- **Columns:** `id`, `Sent_ID` (e.g. `0_NR`), `Word_ID`, `Word`,
  `nFixations`, `meanPupilSize`, `GD`, `TRT`, `FFD`, `SFD`, `GPT`,
  `WordLen`
- A `nFixations == 0` row is a **skip** (the word was in the sentence
  but never fixated). `examples/05_word_level_skip_rates.py` reports
  skip rate by word length and by sentence sentiment.

`word/get_average.py` averages the twelve word-level files. `Sent_ID`
suffix `_NR` means ZuCo normal-reading / sentiment-reading task 1.

`DataTransformer` also knows about ZuCo tasks 2 and 3 and hard-codes
missing-sentence ranges for a few subjects. Only task 1 outputs are
checked in.

---

## 4. Gaze prediction and PROVO

Directory: `gaze_prediction/data/`

| File | Rows | Notes |
| --- | ---: | --- |
| `prediction_test.csv` | 1,751 | Short extract; `sentence_id` starts at 300. |
| `prediction_test_v2.csv` | 191,971 | Predicted `nFix`, `FFD`, `GPT`, `TRT`, `GD` for every SST token in the scaffold. |
| `provo.csv` | 2,659 | PROVO words with `nFix`, `FFD`, `GPT`, `TRT`, **`fixProp`** (not `GD`). |
| `convert_zuco_data.py` | — | Rescales a word-average table into the prediction schema (nFix on its own min-max; the four durations share one min-max, then × 100). |

`result/provo_data_scatter_hist_plots.png`,
`result/train_data_scatter_hist_plots.png`, and
`result/test_data_scatter_hist_plots.png` are pair-plots of those
word-level features. Predicted durations are tightly correlated with
each other (almost linear `TRT` vs `nFix`); PROVO shows the same
qualitative shape, which is why the plots exist.

`examples/08_gaze_prediction_compare.py` reprints those distribution
notes as numbers.

---

## 5. Files that are **not** in the repo

You will hit these if you rerun the original converters from scratch:

| Expected path | Needed by |
| --- | --- |
| `ZuCo_mat_data/task1/*.mat` (12 files) | `utils_ZuCo.get_matfiles`, `read_ZuCo_mat.py` |
| `ZuCo_SST_data/all/{NEGATIVE,POSITIVE,NEUTRAL}/*.txt` | `convert_full_SST.py` |
| NLTK `punkt` | `SST_data/convert_sst_to_et.py` |

The processed CSVs are enough for every script under `examples/` and
for both training entry points.

---

## Suggested mental model

```
ZuCo .mat (12 subjects, task 1)
        │  DataTransformer
        ▼
per-subject sentence CSVs ──► average ──► scale ──► join SST text ──► ZuCo 400
per-subject word CSVs     ──► average ──► optional gaze-prediction training

SST text (11.8k)
        │
        ├─► label map + predicted sentence gaze ──► full SST tables
        └─► tokenize ──► word scaffold ──► predicted word gaze
```
