# Datasets

Everything in this repository is built from two public sources:

1. **ZuCo** (Zurich Cognitive Language Processing Corpus) — EEG and eye-tracking recorded while people read sentences, including movie-review sentences that overlap Stanford Sentiment Treebank style labels.
2. **SST** (Stanford Sentiment Treebank) — movie-review sentences with sentiment labels, used here as the large text-only set that later receives *projected* gaze features.

No raw `.mat` ZuCo recordings are checked in. The MATLAB reader (`read_ZuCo_mat.py` + `utils_ZuCo.py`) expects a local `ZuCo_mat_data/` tree that is not part of this clone.

## Label convention

All sentiment CSVs in this repo use integer labels:

| Integer | Folder / name | Meaning |
| ---: | --- | --- |
| `0` | `NEGATIVE` | negative review |
| `1` | `NEUTRAL` | mixed or weakly polarized |
| `2` | `POSITIVE` | positive review |

`convert_full_SST.py` and `ZuCo_SST_data/save_SST_data.py` both apply that mapping when they walk a `NEGATIVE` / `NEUTRAL` / `POSITIVE` folder of `.txt` files.

## Inventory (checked-in files)

Counts below were measured from the files in this clone.

### ZuCo sentence-level sentiment + gaze

| File | Rows | Role |
| --- | ---: | --- |
| `ZuCo_SST_data/ssts_ZuCo.csv` | 400 | text + label only |
| `ZuCo_SST_data/combined_sst_et_standard.csv` | 400 | text + label + z-scored gaze |
| `ZuCo_SST_data/combined_sst_et_min_max.csv` | 400 | text + label + min-max gaze |
| `ZuCo_SST_data/train.csv` | 320 | 80% split of the standard file |
| `ZuCo_SST_data/valid.csv` | 40 | 10% split |
| `ZuCo_SST_data/test.csv` | 40 | 10% split |

Label counts on the 400-row combined set: **123 / 137 / 140** (neg / neu / pos). That is close to balanced. The 40-row valid/test slices are not: valid is 7 / 14 / 19 and test is 13 / 16 / 11. Prefer the 5-fold script over those tiny slices when you report a number.

Combined ZuCo columns:

```
sentence_id, sentence, sentiment_label,
omissionRate, nFixations, meanPupilSize, GD, TRT, FFD, SFD, GPT
```

### ZuCo eye-tracking only (no sentiment column)

| File | Rows | Granularity |
| --- | ---: | --- |
| `ZuCo_et_csv_data/{1-12}_SR.csv` | 400 each | one subject, sentence-level raw-ish features |
| `ZuCo_et_csv_data/average_data.csv` | 400 | mean across subjects |
| `ZuCo_et_csv_data/standard_scaled_average_data.csv` | 400 | z-score of the average |
| `ZuCo_et_csv_data/min_max_scaled_average_data.csv` | 400 | min-max of the average |
| `ZuCo_et_csv_data/word/{1-12}_SR.csv` | ~7.1k each | one subject, word-level |
| `ZuCo_et_csv_data/word/word_averages.csv` | 7129 | mean across subjects |
| `ZuCo_et_csv_data/word/word_averages_v2.csv` | 7129 | cleaned mean across subjects |

Sentence-level subject columns:

```
id, SentLen, omissionRate, nFixations, meanPupilSize, GD, TRT, FFD, SFD, GPT
```

Word-level columns:

```
id, Sent_ID, Word_ID, Word, nFixations, meanPupilSize, GD, TRT, FFD, SFD, GPT, WordLen
```

`Sent_ID` values look like `0_NR`, `1_NR`, … — sentence index plus a ZuCo task tag (`NR` = normal reading).

Twelve subject files is a ZuCo convention: one `.mat` / CSV per reader.

### Full SST with projected gaze

| File | Rows | Role |
| --- | ---: | --- |
| `SST_data/combined_full_sst_et.csv` | 11853 | all labeled SST rows + 5 gaze columns |
| `SST_data/train_full_sst.csv` | 9482 | 80% |
| `SST_data/valid_full_sst.csv` | 1185 | 10% |
| `SST_data/test_full_sst.csv` | 1186 | 10% |
| `SST_data/stts_all_sentence_level.csv` | (source text) | used by the SST→ET stub |
| `SST_data/sst_et_test.csv` | word-level stub | gaze columns filled with zeros |

Full SST columns:

```
sentence_id, sentence, sentiment_label, nFix, GD, TRT, FFD, GPT
```

Label counts on the 11,853-row combined file: **4649 / 2241 / 4963**. Neutral is the minority class. The split files keep that skew.

These five gaze columns are **not** recordings from people reading the full SST. They are projected or predicted features, intended to let the same fusion head run at SST scale. Treat them as model-generated covariates, not as human ground truth.

### Gaze-prediction word tables

| File | Rows | Notes |
| --- | ---: | --- |
| `gaze_prediction/data/prediction_test.csv` | 1751 | compact word-level prediction sample |
| `gaze_prediction/data/prediction_test_v2.csv` | 191971 | large word-level prediction table |
| `gaze_prediction/data/provo.csv` | (Provo) | extra reading corpus, used for plots |
| `gaze_prediction/data/convert_zuco_data.py` | — | scales ZuCo word averages into the prediction schema |

Prediction schema:

```
sentence_id, word_id, word, nFix, FFD, GPT, TRT, GD
```

`result/` holds scatter/histogram plots for train, test, and Provo gaze distributions.

## How the tables join

```
ZuCo .mat (not in repo)
        │
        ▼
utils_ZuCo.DataTransformer  ──►  ZuCo_et_csv_data/{k}_SR.csv
        │
        ▼
subject mean + scaling      ──►  average / standard / min-max CSVs
        │
        ▼
join on sentence id         ──►  ZuCo_SST_data/combined_sst_et_*.csv
        │
        ├── model_ZuCo_SST.py          (5-fold on 400 rows)
        └── optional 80/10/10 split    (train.csv / valid.csv / test.csv)

SST text + labels
        │
        ▼
predicted / projected ET    ──►  SST_data/combined_full_sst_et.csv
        │
        ▼
sklearn split, seed 42      ──►  train / valid / test_full_sst.csv
        │
        ▼
model_full_SST.py
```

`sentence_id` is the join key on every sentiment table. On word-level ZuCo files the key is `Sent_ID` (string) plus `Word_ID`.

## Which file should an example use?

| Goal | File |
| --- | --- |
| Inspect real human gaze + labels | `ZuCo_SST_data/combined_sst_et_standard.csv` |
| Inspect a single reader | `ZuCo_et_csv_data/1_SR.csv` |
| Inspect word-level reading | `ZuCo_et_csv_data/word/word_averages_v2.csv` |
| Inspect large-scale fusion inputs | `SST_data/train_full_sst.csv` |
| Compare split leakage | `examples/split_audit.py` |

## Licensing and citation (personal reminder)

ZuCo and SST are third-party corpora. If you publish or share derived tables, cite the original papers and respect their licenses. This repository is a personal research workspace, not a redistribution of the raw ZuCo MATLAB release.
