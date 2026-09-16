# Datasets and file inventory

Everything listed here lives in this personal clone. Counts are from the files on disk, not from a paper table.

## Label convention

| Integer | Meaning | Source folder name |
| ---: | --- | --- |
| 0 | Negative | `NEGATIVE` |
| 1 | Neutral | `NEUTRAL` |
| 2 | Positive | `POSITIVE` |

`convert_full_SST.py` and `ZuCo_SST_data/save_SST_data.py` both implement that mapping.

---

## A. ZuCo sentiment (real gaze)

Zurich Cognitive Language Processing Corpus, Task 1: subjects read movie-review sentences and the lab recorded eye tracking (and EEG, unused here).

### Text + labels

| File | Rows | Columns |
| --- | ---: | --- |
| `ZuCo_SST_data/ssts_ZuCo.csv` | 400 | `sentence_id`, `sentence`, `sentiment_label` |

Label counts: **123 negative / 137 neutral / 140 positive**.

`sentence_id` is an integer 0…399 that matches the `id` column in the sentence-level gaze CSVs.

### Sentence-level gaze, per subject

Directory: `ZuCo_et_csv_data/{1..12}_SR.csv`

| Subject file | Rows |
| --- | ---: |
| `1_SR.csv` … `2`, `4`–`12` | 400 |
| `3_SR.csv` | **299** |

Subject 3 is the known short file: `utils_ZuCo.py` drops a contiguous block of Task 1 sentences for that reader (the `task1` / `subject == 2` zero-based branch). Do not average subject 3 with the others without aligning on `id`.

Columns:

```text
id, SentLen, omissionRate, nFixations, meanPupilSize, GD, TRT, FFD, SFD, GPT
```

Values in the raw subject CSVs are **unscaled** (milliseconds / counts). `read_ZuCo_mat.py` requests `scaling='raw'`.

### Sentence-level gaze, averaged

| File | Rows | Scaling |
| --- | ---: | --- |
| `ZuCo_et_csv_data/average_data.csv` | 400 | raw mean |
| `ZuCo_et_csv_data/min_max_scaled_average_data.csv` | 400 | min-max |
| `ZuCo_et_csv_data/standard_scaled_average_data.csv` | 400 | z-score |

Built by `get_average_sentence_level.py` (script still points at a folder name `et_csv_data`; the checked-in files live under `ZuCo_et_csv_data/`).

### Joined ZuCo training tables

| File | Rows | Notes |
| --- | ---: | --- |
| `ZuCo_SST_data/combined_sst_et_standard.csv` | 400 | text + standard-scaled gaze |
| `ZuCo_SST_data/combined_sst_et_min_max.csv` | 400 | text + min-max gaze |
| `ZuCo_SST_data/train.csv` | 320 | 80% of the standard table |
| `ZuCo_SST_data/valid.csv` | 40 | 10% |
| `ZuCo_SST_data/test.csv` | 40 | 10% |

`model_ZuCo_SST.py` **does not use** the 80/10/10 CSVs. It reads `combined_sst_et_standard.csv` and runs `StratifiedKFold`. The split files exist because `ZuCo_SST_data/spilt.py` (typo intended; that is the filename) was used for an earlier hold-out run.

Joined columns:

```text
sentence_id, sentence, sentiment_label,
omissionRate, nFixations, meanPupilSize, GD, TRT, FFD, SFD, GPT
```

The model then **selects only** `nFixations, FFD, GPT, TRT, GD`.

### Word-level gaze

Directory: `ZuCo_et_csv_data/word/`

| File | Rows | Notes |
| --- | ---: | --- |
| `{1..12}_SR.csv` except 3 | 7129 | one row per word in 400 sentences |
| `3_SR.csv` | 5293 | same subject-3 gap |
| `word_averages.csv` | 7129 | older mean |
| `word_averages_v2.csv` | 7129 | current mean used by gaze conversion |

Columns:

```text
id, Sent_ID, Word_ID, Word, nFixations, meanPupilSize,
GD, TRT, FFD, SFD, GPT, WordLen
```

`Sent_ID` looks like `0_NR`, `1_NR`, … (natural reading). `gaze_prediction/data/convert_zuco_data.py` splits on `_` and keeps the integer prefix.

---

## B. Full Stanford Sentiment Treebank (predicted gaze)

### Sentence-level tables used by `model_full_SST.py`

| File | Rows | Role |
| --- | ---: | --- |
| `SST_data/combined_full_sst_et.csv` | 11853 | all SST + 5 gaze columns |
| `SST_data/train_full_sst.csv` | 9482 | 80% |
| `SST_data/valid_full_sst.csv` | 1185 | 10% |
| `SST_data/test_full_sst.csv` | 1186 | 10% |

Split: `SST_data/spilt.py`, `train_test_split(..., random_state=42)` twice (80 / 10 / 10).

Columns:

```text
sentence_id, sentence, sentiment_label, nFix, GD, TRT, FFD, GPT
```

Label counts on the combined file: **4649 / 2241 / 4963** (neg / neu / pos). Neutral is the minority class, which is why the metrics use `average='weighted'`.

### Raw SST dump

`SST_data/stts_all_sentence_level.csv` is a headerless two-column file: sentence text, then `POSITIVE` / `NEGATIVE` / `NEUTRAL`. 11852 data rows. It is the source `SST_data/convert_sst_to_et.py` tokenizes into word-level placeholders.

### Word-level SST gaze placeholders / predictions

| File | Rows | Meaning |
| --- | ---: | --- |
| `SST_data/sst_et_test.csv` | 191971 | word rows; many gaze cells are 0 (placeholder) |
| `gaze_prediction/data/prediction_test_v2.csv` | 191971 | same shape, filled predicted gaze |
| `gaze_prediction/data/prediction_test.csv` | 1751 | smaller prediction slice (`sentence_id` starting at 300) |

Word columns:

```text
sentence_id, word_id, word, nFix, FFD, GPT, TRT, GD
```

The sentence-level `nFix, GD, TRT, FFD, GPT` on the full SST tables are aggregations of these word predictions (mean or model-specific pooling done offline), then **standard-scaled**. `examples/feature_stats.py` reports train means ≈ 0 and stds ≈ 1; the word-level predicted files are still on the predictor’s 0–100-ish scale. This snapshot does not include that aggregation / scaling script.

---

## C. PROVO (gaze-prediction side corpus)

`gaze_prediction/data/provo.csv` — 2659 word rows.

```text
sentence_id, word_id, word, nFix, FFD, GPT, TRT, fixProp
```

`fixProp` is fixation probability, not gaze duration. PROVO is used as *training material for predicting gaze on unseen words*, not as a sentiment set. There are no polarity labels in this file.

---

## D. Diagnostic plots

`result/` holds three PNGs from an earlier scatter/histogram pass:

- `provo_data_scatter_hist_plots.png`
- `train_data_scatter_hist_plots.png`
- `test_data_scatter_hist_plots.png`

They are not produced by any script still in the repo. Keep them as personal figures; do not treat them as the current evaluation.

---

## E. Files that are *not* in the clone

| Expected by | Missing path | Effect |
| --- | --- | --- |
| `read_ZuCo_mat.py` / `get_matfiles` | `ZuCo_mat_data/task1/*.mat` (12 files) | cannot regenerate subject CSVs from MATLAB |
| `convert_full_SST.py` | `ZuCo_SST_data/all/{NEGATIVE,POSITIVE,NEUTRAL}/*.txt` | cannot rebuild `ssts_ZuCo.csv` from raw text |
| `get_average_sentence_level.py` | folder named `et_csv_data` | script must be pointed at `ZuCo_et_csv_data` |
| gaze predictor | model weights / training script | cannot re-predict SST word gaze |

The checked-in CSVs are enough to train `model_ZuCo_SST.py` and `model_full_SST.py` and to run every script in `examples/`.

---

## Quick size check

Run from the repo root:

```bash
python3 examples/inspect_datasets.py
```

That script is the machine-readable version of the tables on this page.
