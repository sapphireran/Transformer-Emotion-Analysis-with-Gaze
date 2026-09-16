# Dataset catalog

All counts below were taken from the CSVs currently in the repository
(header row excluded). Paths are relative to the repo root.

## ZuCo-SST (real gaze)

Movie-review sentences that appear in ZuCo Task 1, joined to
**subject-averaged** sentence-level eye tracking.

| File | Rows | Role |
| --- | ---: | --- |
| `ZuCo_SST_data/ssts_ZuCo.csv` | 400 | text + label only |
| `ZuCo_SST_data/combined_sst_et_standard.csv` | 400 | training default (`model_ZuCo_SST.py`) |
| `ZuCo_SST_data/combined_sst_et_min_max.csv` | 400 | same rows, min-max gaze |
| `ZuCo_SST_data/train.csv` | 320 | 80% hold-out (not used by the CV script) |
| `ZuCo_SST_data/valid.csv` | 40 | 10% hold-out |
| `ZuCo_SST_data/test.csv` | 40 | 10% hold-out |
| `ZuCo_SST_data/ZuCo_SST_data.zip` | — | archive of the folder |

**Columns (combined / split files):**
`sentence_id`, `sentence`, `sentiment_label`, `omissionRate`, `nFixations`,
`meanPupilSize`, `GD`, `TRT`, `FFD`, `SFD`, `GPT`.

**Label map:** `0 = NEGATIVE`, `1 = NEUTRAL`, `2 = POSITIVE`
(`convert_full_SST.py`, `ZuCo_SST_data/save_SST_data.py`).

| Table | NEG | NEU | POS |
| --- | ---: | ---: | ---: |
| combined | 123 | 137 | 140 |
| train.csv | 103 | 107 | 110 |
| valid.csv | 7 | 14 | 19 |
| test.csv | 13 | 16 | 11 |

The 40-row valid split is **not** balanced (19 positive vs 7 negative).
Treat hold-out numbers from those files as exploratory; the paper-style
protocol is 5-fold CV on the combined table.

`examples/label_and_split_audit.py` checks that the 320/40/40 files do not
reuse `sentence_id`s.

## Per-subject ZuCo sentence ET

`read_ZuCo_mat.py` writes one CSV per participant. The checked-in copies
live in `ZuCo_et_csv_data/{1..12}_SR.csv`.

| File | Sentence rows |
| --- | ---: |
| `1_SR.csv`, `2_SR.csv`, `4_SR.csv`–`12_SR.csv` | 400 |
| `3_SR.csv` | 299 |

Subject 3 is 0-indexed subject `2` in `DataTransformer`. For Task 1 that
code skips trials `150–249` and `399`, which is why the file is short.

**Columns:** `id`, `SentLen`, `omissionRate`, `nFixations`, `meanPupilSize`,
`GD`, `TRT`, `FFD`, `SFD`, `GPT`.

Aggregates:

| File | What it is |
| --- | --- |
| `average_data.csv` | index-aligned mean across subjects (raw units) |
| `min_max_scaled_average_data.csv` | sklearn `MinMaxScaler` on the mean table |
| `standard_scaled_average_data.csv` | sklearn `StandardScaler` on the mean table |

## Per-subject ZuCo word ET

`ZuCo_et_csv_data/word/{1..12}_SR.csv` plus `word_averages.csv` and
`word_averages_v2.csv`.

Typical subject files have **7,129** word rows. Subject 3 has **5,293**.
`get_average.py` mean-stacks on row index and copies identity columns from
`1_SR.csv`.

**Columns:** `id`, `Sent_ID`, `Word_ID`, `Word`, `nFixations`,
`meanPupilSize`, `GD`, `TRT`, `FFD`, `SFD`, `GPT`, `WordLen`.

`Sent_ID` looks like `0_NR` because Task 1 is Normal / Sentiment Reading.

## Full SST (projected gaze)

| File | Rows | Notes |
| --- | ---: | --- |
| `SST_data/stts_all_sentence_level.csv` | 11,853 | no header; `sentence,LABEL_NAME` |
| `SST_data/combined_full_sst_et.csv` | 11,853 | sentence + id + label + 5 gaze cols |
| `SST_data/train_full_sst.csv` | 9,482 | 80% (`random_state=42`) |
| `SST_data/valid_full_sst.csv` | 1,185 | 10% |
| `SST_data/test_full_sst.csv` | 1,186 | 10% |
| `SST_data/sst_et_test.csv` | 191,971 | word-level skeleton (`nFix`… all 0) |

`stts_all_sentence_level.csv` label strings: POSITIVE 4,963, NEGATIVE 4,649,
NEUTRAL 2,241.

**Columns (combined / split):** `sentence_id`, `sentence`,
`sentiment_label`, `nFix`, `GD`, `TRT`, `FFD`, `GPT`.

| Split | NEG | NEU | POS |
| --- | ---: | ---: | ---: |
| train | 3,710 | 1,833 | 3,939 |
| valid | 476 | 209 | 500 |
| test | 463 | 199 | 524 |

Neutral is ~19% everywhere. A majority-class baseline on train is about
41.5% (always POSITIVE).

`SST_data/convert_sst_to_et.py` tokenizes SST with NLTK and writes a
word-level table of **zeros** — a placeholder for a later gaze-prediction
model, not measured ET.

## Gaze-prediction side tables

| File | Rows | Columns of interest |
| --- | ---: | --- |
| `gaze_prediction/data/prediction_test.csv` | 1,751 | smaller predicted-gaze dump |
| `gaze_prediction/data/prediction_test_v2.csv` | 191,971 | word-level SST predictions |
| `gaze_prediction/data/provo.csv` | 2,659 | PROVO words; `fixProp` instead of `GD` |
| `gaze_prediction/data/convert_zuco_data.py` | — | min-max to 0–100, rename columns |

`prediction_test_v2.csv` has the same row count as `SST_data/sst_et_test.csv`
(one predicted row per placeholder word).

## Plots

`result/` holds three exploratory figures from earlier analysis:

- `train_data_scatter_hist_plots.png`
- `test_data_scatter_hist_plots.png`
- `provo_data_scatter_hist_plots.png`

They are not produced by the current example scripts.

## Files that are not checked in

- Raw ZuCo `.mat` files (`utils_ZuCo.get_matfiles` looks for
  `ZuCo_mat_data/task1` with a Windows-style slash).
- The `ZuCo_SST_data/all/{NEGATIVE,POSITIVE,NEUTRAL}/*.txt` tree that
  `save_SST_data.py` and `convert_full_SST.py` expect.
- `gaze_prediction/data/training_data/word_averages_v2.csv` (input of
  `convert_zuco_data.py`).
- Trained `models/best_*_model.pth` checkpoints.
