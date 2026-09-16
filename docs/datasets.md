# Datasets

All counts below were measured from the committed CSVs (see `examples/scripts/summarize_datasets.py`). Do not mix column names across tables: ZuCo uses `nFixations`, the full-SST predicted tables use `nFix`, and Provo uses `fixProp` instead of `GD`.

## Label convention

Used everywhere sentiment is numeric:

| Folder / string | `sentiment_label` |
| --- | ---: |
| `NEGATIVE` | 0 |
| `NEUTRAL` | 1 |
| `POSITIVE` | 2 |

Defined in `convert_full_SST.py` and `ZuCo_SST_data/save_SST_data.py`.

---

## 1. ZuCo Task 1 (measured gaze + SST movie reviews)

[ZuCo](https://osf.io/q3zws/) records simultaneous EEG and eye-tracking while people read. **This repo only keeps eye-tracking features**, and only Task 1: *normal reading* of 400 Stanford Sentiment Treebank movie-review sentences.

### Subject sentence tables

`ZuCo_et_csv_data/{1–12}_SR.csv`

| Column | Meaning |
| --- | --- |
| `id` | compacted row index after `DataTransformer` skips bad trials |
| `SentLen` | number of words in the sentence |
| `omissionRate` | fraction of words without a reported fixation |
| `nFixations` | mean fixation count over fixated words |
| `meanPupilSize` | mean pupil size over those words |
| `GD`, `TRT`, `FFD`, `SFD`, `GPT` | standard reading-time measures (milliseconds in the raw dump) |

| File | Rows | Notes |
| --- | ---: | --- |
| `1_SR.csv`, `2_SR.csv`, `4_SR.csv`–`12_SR.csv` | 400 | `id` 0–399 |
| `3_SR.csv` | **299** | Task 1, subject index 2 in `utils_ZuCo.py`; original sentences 150–249 and 399 dropped |

Raw (unscaled) corpus-level means on `average_data.csv` (400 rows, 12-subject mean where available):

| Feature | Mean | Std | Min | Max |
| --- | ---: | ---: | ---: | ---: |
| `SentLen` | 17.81 | 8.06 | 3 | 42 |
| `omissionRate` | 0.319 | 0.067 | 0.156 | 0.601 |
| `nFixations` | 1.69 | 0.31 | 1.20 | 3.58 |
| `meanPupilSize` | 797.0 | 60.9 | 679.7 | 969.7 |
| `GD` | 141.4 ms | 21.5 | 107.6 | 272.7 |
| `TRT` | 202.6 ms | 47.9 | 131.1 | 427.0 |
| `FFD` | 116.9 ms | 8.0 | 101.5 | 165.9 |
| `SFD` | 71.6 ms | 10.9 | 39.3 | 121.8 |
| `GPT` | 241.8 ms | 56.7 | 153.2 | 586.9 |

### Word-level tables

`ZuCo_et_csv_data/word/{1–12}_SR.csv` and `word_averages_v2.csv`

| Column | Meaning |
| --- | --- |
| `id` | row index |
| `Sent_ID` | `{compacted_idx}_NR` (NR = normal reading) |
| `Word_ID` | 0-based token index in the sentence |
| `Word` | token (leading punctuation stripped; first word lowercased) |
| `nFixations`, `meanPupilSize`, `GD`, `TRT`, `FFD`, `SFD`, `GPT` | per-word gaze |
| `WordLen` | character length of `Word` |

`word_averages_v2.csv`: **7,129** rows, **400** distinct `Sent_ID` values. Subject 3 has 5,293 word rows / 299 sentences.

### Labeled 400-sentence table

`ZuCo_SST_data/ssts_ZuCo.csv` — `sentence_id`, `sentence`, `sentiment_label`.

| Label | Count | Share |
| ---: | ---: | ---: |
| 0 negative | 123 | 30.8% |
| 1 neutral | 137 | 34.2% |
| 2 positive | 140 | 35.0% |

Sentence length in characters: mean 106.7, min 23, max 251.

### Joined training tables

`combined_sst_et_standard.csv` and `combined_sst_et_min_max.csv` are inner joins of `ssts_ZuCo.csv` with the matching scaled gaze file on `sentence_id == id`, **without** `SentLen`.

`model_ZuCo_SST.py` reads the **standard-scaled** join and selects `nFixations, FFD, GPT, TRT, GD`.

The 80/10/10 files `train.csv` / `valid.csv` / `test.csv` exist but are unused by that script (5-fold CV on the full 400 instead). Valid/test are 40 rows each — label counts there are noisy (`valid` is 7 / 14 / 19).

---

## 2. Full Stanford Sentiment Treebank (predicted gaze)

### Raw sentences

`SST_data/stts_all_sentence_level.csv` — no header, two columns: sentence text, `{NEGATIVE,NEUTRAL,POSITIVE}`.

| String label | Count |
| --- | ---: |
| POSITIVE | 4,963 |
| NEGATIVE | 4,649 |
| NEUTRAL | 2,241 |
| **Total** | **11,853** |

Character length: mean ≈ 103, min 4, max 283.

### Combined predicted gaze

`SST_data/combined_full_sst_et.csv`

Columns: `sentence_id`, `sentence`, `sentiment_label`, `nFix`, `GD`, `TRT`, `FFD`, `GPT`.

Gaze columns are **z-scored** (mean ≈ 0, std ≈ 1 on the full 11,853). They are **not** ZuCo milliseconds. Extremely high Pearson correlations among `nFix`, `FFD`, `GPT`, `TRT` (see [gaze-features.md](gaze-features.md)) are a signature of predicted rather than measured features.

`sentence_id` is 0–11852 in file order. It correlates with the label (r ≈ −0.37) because the raw SST file is **not shuffled by class**. Do not treat `sentence_id` as a feature.

### Splits used by `model_full_SST.py`

| File | Rows | 0 / 1 / 2 |
| --- | ---: | --- |
| `train_full_sst.csv` | 9,482 | 3710 / 1833 / 3939 |
| `valid_full_sst.csv` | 1,185 | 476 / 209 / 500 |
| `test_full_sst.csv` | 1,186 | 463 / 199 / 524 |

`sentence_id` sets are disjoint and cover the 11,853 combined rows.

### Word-level SST with placeholder zeros

`SST_data/sst_et_test.csv` is produced by `convert_sst_to_et.py`: NLTK `word_tokenize`, keep `[A-Za-z]+` tokens, write `nFix=FFD=GPT=TRT=GD=0`. Useful as a **schema template**, not as gaze.

---

## 3. Predicted gaze dumps and Provo

The predictor itself is out of tree. What remains:

| File | Rows | Sentence IDs | Gaze columns |
| --- | ---: | --- | --- |
| `gaze_prediction/data/prediction_test.csv` | 1,751 | 300–399 (100 sentences) | `nFix, FFD, GPT, TRT, GD` |
| `gaze_prediction/data/prediction_test_v2.csv` | 191,971 | full SST word stream | same |
| `gaze_prediction/data/provo.csv` | 2,659 | 134 Provo sentences | `nFix, FFD, GPT, TRT, fixProp` |

`convert_zuco_data.py` min-max scales ZuCo word averages to roughly 0–100 (`nFix` on its own range; `FFD/GPT/TRT/GD` on a shared range) so they sit next to predicted values.

`result/train_data_scatter_hist_plots.png`, `test_data_scatter_hist_plots.png`, and `provo_data_scatter_hist_plots.png` are pairwise scatter+histogram grids of those word-level channels (nFix vs FFD vs GPT vs TRT vs GD/fixProp).

---

## 4. What is not in git

- ZuCo MATLAB `sentenceData` structs (`ZuCo_mat_data/`)
- EEG
- Hugging Face weights / `models/*.pth`
- The neural gaze-prediction trainer
- `ZuCo_SST_data/all/{NEGATIVE,POSITIVE,NEUTRAL}/` as unpacked files (only the zip)

Public sources: see [references.md](references.md).
