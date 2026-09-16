# Datasets

All row counts below were computed from the files in this clone. Re-run
`python examples/08_write_dataset_report.py` if the CSVs change.

## Label convention

Every table that has `sentiment_label` uses the mapping from
`convert_full_SST.py` / `ZuCo_SST_data/save_SST_data.py`:

| Folder / string | Code | Meaning |
| --- | ---: | --- |
| `NEGATIVE` | 0 | negative |
| `NEUTRAL` | 1 | mixed or neither |
| `POSITIVE` | 2 | positive |

`SST_data/stts_all_sentence_level.csv` still stores the string in the second
column (`POSITIVE` / `NEUTRAL` / `NEGATIVE`). The joined modeling tables use
the integer code.

## Track A — ZuCo ∩ SST (human gaze)

ZuCo Task 1 is normal reading of movie reviews (NR). After extraction and
join this repo keeps **400** sentences, ids `0..399`.

| File | Rows | Role |
| --- | ---: | --- |
| `ZuCo_SST_data/ssts_ZuCo.csv` | 400 | text + label only |
| `ZuCo_SST_data/combined_sst_et_standard.csv` | 400 | **training table** for `model_ZuCo_SST.py` |
| `ZuCo_SST_data/combined_sst_et_min_max.csv` | 400 | same join, min-max gaze |
| `ZuCo_SST_data/train.csv` | 320 | 80% split (seed 42) |
| `ZuCo_SST_data/valid.csv` | 40 | 10% split |
| `ZuCo_SST_data/test.csv` | 40 | 10% split |
| `ZuCo_et_csv_data/average_data.csv` | 400 | subject-averaged raw units |
| `ZuCo_et_csv_data/{1-12}_SR.csv` | 400 except subject 3 (299) | per-reader sentence gaze |
| `ZuCo_et_csv_data/word/word_averages_v2.csv` | 7,129 | subject-averaged words |
| `ZuCo_et_csv_data/standard_scaled_average_data.csv` | 400 | z-scored raw average |
| `ZuCo_et_csv_data/min_max_scaled_average_data.csv` | 400 | min-max raw average |

Label mix on the 400-sentence join:

| label | name | count | percent |
| ---: | --- | ---: | ---: |
| 0 | negative | 123 | 30.8% |
| 1 | neutral | 137 | 34.2% |
| 2 | positive | 140 | 35.0% |

The 80/10/10 split is disjoint on `sentence_id` and covers the parent table.
It is **not** what `model_ZuCo_SST.py` uses. That script ignores the split
files and runs `StratifiedKFold(n_splits=5)` on the full 400 rows.

### Sentence-level schema (joined)

```
sentence_id, sentence, sentiment_label,
omissionRate, nFixations, meanPupilSize, GD, TRT, FFD, SFD, GPT
```

The training script only forwards five of those channels:
`nFixations, FFD, GPT, TRT, GD`. `omissionRate`, `meanPupilSize`, and `SFD`
are available for analysis and are included in the examples.

### Word-level schema (ZuCo)

```
id, Sent_ID, Word_ID, Word,
nFixations, meanPupilSize, GD, TRT, FFD, SFD, GPT, WordLen
```

`Sent_ID` looks like `0_NR`. There are 400 distinct sentence ids and 7,129
word rows (mean 17.8 words / sentence, max 43).

### Per-subject files

Subjects 1–2 and 4–12 each have 400 sentence rows. Subject 3 (`3_SR.csv`) has
**299** rows, ids `0..298`. That matches `DataTransformer` dropping 101 Task 1
sentences for 0-based subject 2 (file `3_SR.csv`): original indices 150–249
and 399. See [known limitations](notes/known-limitations.md) for what that
does to index-based averaging.

## Track B — full SST (predicted gaze)

| File | Rows | Role |
| --- | ---: | --- |
| `SST_data/stts_all_sentence_level.csv` | 11,852 text rows + header | original SST snippets + string label |
| `SST_data/combined_full_sst_et.csv` | 11,853 | **parent table** for the full-SST track |
| `SST_data/train_full_sst.csv` | 9,482 | 80% |
| `SST_data/valid_full_sst.csv` | 1,185 | 10% |
| `SST_data/test_full_sst.csv` | 1,186 | 10% |
| `SST_data/sst_et_test.csv` | 191,971 | word skeleton with **zeros** |
| `gaze_prediction/data/prediction_test_v2.csv` | 191,971 | word-level **predicted** gaze |
| `gaze_prediction/data/prediction_test.csv` | 1,751 | small predicted sample (100 sentences) |
| `gaze_prediction/data/provo.csv` | 2,659 | Provo words (134 sentences) |

Label mix on `combined_full_sst_et.csv`:

| label | name | count | percent |
| ---: | --- | ---: | ---: |
| 0 | negative | 4,649 | 39.2% |
| 1 | neutral | 2,241 | 18.9% |
| 2 | positive | 4,963 | 41.9% |

The 80/10/10 split is disjoint and covers all 11,853 parent ids. Neutral is
the rare class on every slice.

### Sentence-level schema (full SST)

```
sentence_id, sentence, sentiment_label, nFix, GD, TRT, FFD, GPT
```

Names drop the ZuCo `nFixations` spelling. Units are already z-scored
(column means ≈ 0, std ≈ 1). They are **not** milliseconds.

### Word-level predicted schema

```
sentence_id, word_id, word, nFix, FFD, GPT, TRT, GD
```

`prediction_test_v2.csv` covers every SST sentence (mean 16.2 words, max 51).
`sst_et_test.csv` has the same shape but every gaze cell is 0 — it is the
placeholder `SST_data/convert_sst_to_et.py` writes before a predictor runs.

## Provo

`gaze_prediction/data/provo.csv` is the natural-reading source used to train
or calibrate the gaze predictor. Columns include `fixProp` (fixation
proportion) instead of `GD`. 134 sentences, 2,659 words.

## Catalog keys

The examples library addresses these tables by key. See
`examples/gaze_emotion_examples/catalog.py` for the authoritative list
(`zuco_sst_standard`, `sst_full`, `zuco_word_raw`, `provo_word`, …).
