# Datasets

All counts below were computed from the CSVs in this checkout (see
`examples/01_inspect_datasets.py`). None of these files are downloaded
at runtime.

## 1. ZuCo ∩ SST — measured sentence-level gaze

**Role:** gold pair of (review sentence, real eye-tracking, 3-class label).

| File | Rows | Columns |
| --- | ---: | --- |
| `ZuCo_SST_data/ssts_ZuCo.csv` | 400 | `sentence_id`, `sentence`, `sentiment_label` |
| `ZuCo_SST_data/combined_sst_et_standard.csv` | 400 | labels + 8 scaled gaze fields |
| `ZuCo_SST_data/combined_sst_et_min_max.csv` | 400 | same columns, min–max scaled |
| `ZuCo_SST_data/train.csv` | 320 | copy of the standard table, 80% |
| `ZuCo_SST_data/valid.csv` | 40 | 10% |
| `ZuCo_SST_data/test.csv` | 40 | 10% |

`model_ZuCo_SST.py` reads **only** `combined_sst_et_standard.csv` and
performs its own `StratifiedKFold`. The 320/40/40 files are leftovers
from `ZuCo_SST_data/spilt.py` (note the filename typo) and are useful
for the toy example, not for the transformer script.

### Label balance (400 sentences)

| `sentiment_label` | Count | Share |
| ---: | ---: | ---: |
| 0 negative | 123 | 30.8% |
| 1 neutral | 137 | 34.2% |
| 2 positive | 140 | 35.0% |

Whitespace token length: min 3, mean 17.82, max 43.

### Columns on the combined tables

```
sentence_id, sentence, sentiment_label,
omissionRate, nFixations, meanPupilSize, GD, TRT, FFD, SFD, GPT
```

The transformer only consumes `nFixations, FFD, GPT, TRT, GD`.

### How the 400 sentences were labeled

`convert_full_SST.py` (repo root) and `ZuCo_SST_data/save_SST_data.py`
walk `NEGATIVE/`, `POSITIVE/`, `NEUTRAL/` folders of `.txt` files. The
filename stem is `sentence_id`. Those folders are **not** in the
checkout; only the resulting `ssts_ZuCo.csv` is.

## 2. Per-subject ZuCo sentence tables

**Role:** the 12 readers before averaging.

Path: `ZuCo_et_csv_data/{1-12}_SR.csv`

Shared header:

```
id, SentLen, omissionRate, nFixations, meanPupilSize, GD, TRT, FFD, SFD, GPT
```

| File | Rows | Notes |
| --- | ---: | --- |
| `1_SR` … `2_SR`, `4_SR` … `12_SR` | 400 | one row per sentence |
| `3_SR.csv` | 299 | subject index 2 in `DataTransformer` (0-based) drops blocks of sentences |

Values here are **raw** (milliseconds / counts), produced by
`read_ZuCo_mat.py` with `scaling='raw'`. They are **not** what the
classifier sees.

`average_data.csv` (400 rows) is the element-wise mean across subjects
after `get_average_sentence_level.py` (zeros → NaN before the mean, so
a skipped fixation does not drag the mean to 0). Then:

- `min_max_scaled_average_data.csv`
- `standard_scaled_average_data.csv`

Those two scaled averages are the gaze half of the combined ZuCo–SST
tables once joined on sentence id.

## 3. Word-level ZuCo averages

**Role:** per-token gaze for gaze-prediction / SST transfer.

| File | Rows | Header |
| --- | ---: | --- |
| `ZuCo_et_csv_data/word/word_averages_v2.csv` | 7,129 | `id, Sent_ID, Word_ID, Word, nFixations, meanPupilSize, GD, TRT, FFD, SFD, GPT, WordLen` |
| `ZuCo_et_csv_data/word/{1-12}_SR.csv` | varies | per-subject word tables |
| `ZuCo_et_csv_data/word/word_averages.csv` | older average | prefer `*_v2` |

`Sent_ID` looks like `0_NR`, `1_NR`, … (`NR` = normal reading, ZuCo
Task 1 / 2). `get_average.py` in that folder groups by row index across
subjects, then fills missing `Word` with `unknown` and numeric NaNs
with 0.

## 4. Full SST with transferred gaze

**Role:** scale the same 5-d fusion head to the classic SST sentence set.

| File | Rows | Header |
| --- | ---: | --- |
| `SST_data/stts_all_sentence_level.csv` | 11,853 lines (no header) | `sentence, POSITIVE\|NEUTRAL\|NEGATIVE` as raw text |
| `SST_data/combined_full_sst_et.csv` | 11,853 | `sentence_id, sentence, sentiment_label, nFix, GD, TRT, FFD, GPT` |
| `SST_data/train_full_sst.csv` | 9,482 | same |
| `SST_data/valid_full_sst.csv` | 1,185 | same |
| `SST_data/test_full_sst.csv` | 1,186 | same |
| `SST_data/sst_et_test.csv` | 191,971 | word-level placeholders (`nFix=0`, …) from `convert_sst_to_et.py` |

### Label balance (combined 11,853)

| Label | Count | Share |
| ---: | ---: | ---: |
| 0 | 4,649 | 39.2% |
| 1 | 2,241 | 18.9% |
| 2 | 4,963 | 41.9% |

Neutral is scarce relative to the ZuCo 400. Token length: min 2, mean
19.17, max 56.

`SST_data/spilt.py` created the 80/10/10 files from
`combined_full_sst_et.csv` with `random_state=42` and **no
stratify=**. That is why valid/test class ratios wobble a little.

Gaze columns on this track are **not** ZuCo recordings of these
sentences. They are estimated / mapped values (see
[04-preprocessing-pipeline.md](04-preprocessing-pipeline.md)). Many
rows are already z-scored-looking (negative and >1 both appear).

## 5. Gaze-prediction helper tables

Under `gaze_prediction/data/`:

| File | Rows | Notes |
| --- | ---: | --- |
| `provo.csv` | 2,659 | PROVO-style word rows; last column is `fixProp` not `GD` |
| `prediction_test.csv` | 1,751 | `sentence_id, word_id, word, nFix, FFD, GPT, TRT, GD` |
| `prediction_test_v2.csv` | 191,971 | same header; one predicted row per SST token |

`gaze_prediction/data/convert_zuco_data.py` min–max scales nFixations
and the duration fields **separately** onto a 0–100 range and rewrites
`Sent_ID` / `Word_ID` / `Word` into the prediction schema. The script
hard-codes `training_data/word_averages_v2.csv`, which is not in this
tree — run it only after pointing `input_csv` at
`ZuCo_et_csv_data/word/word_averages_v2.csv`.

## 6. What is *not* in the repo

- `ZuCo_mat_data/` MATLAB recordings (required by `read_ZuCo_mat.py`)
- `ZuCo_SST_data/all/{NEGATIVE,POSITIVE,NEUTRAL}/*.txt`
- Hugging Face weights (`bert-base-uncased`, `roberta-base`)
- Trained `models/best_*_model.pth` checkpoints
- EEG arrays (ZuCo has them; this pipeline never exports them)

## Quick mental model

```
400 sentences   =  “I know how people actually read this review.”
11,853 sentences = “I pretend I know, using a gaze model / mapping.”
7,129 word rows  =  “Average reader on each ZuCo token.”
12 *_SR.csv      =  “One human each.”
```

Use the 400 when you care about the cognitive claim. Use the 11k when
you care about whether the **engineering** of late fusion still trains
stably. Do not mix their metrics in one table without a bold footnote.
