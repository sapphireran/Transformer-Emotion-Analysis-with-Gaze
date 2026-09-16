# Dataset inventory

Every committed table, its schema, and who reads it. Row counts are from this checkout; re-run `examples/inspect_datasets.py` after you replace a file.

## ZuCo sentence-level eye tracking

| File | Rows (approx.) | Key columns | Produced by | Read by |
| --- | --- | --- | --- | --- |
| `ZuCo_et_csv_data/{1-12}_SR.csv` | 400 each except `3_SR.csv` (299) | `id`, `SentLen`, `omissionRate`, `nFixations`, `meanPupilSize`, `GD`, `TRT`, `FFD`, `SFD`, `GPT` | `read_ZuCo_mat.py` | averaging script |
| `ZuCo_et_csv_data/average_data.csv` | ~400 | same, subject-mean raw ms | `get_average_sentence_level.py` | analysis |
| `ZuCo_et_csv_data/min_max_scaled_average_data.csv` | ~400 | same, min-max | averaging script | optional join |
| `ZuCo_et_csv_data/standard_scaled_average_data.csv` | ~400 | same, z-score | averaging script | ZuCo SST join |

`id` is the sentence index, 0-based, aligned with ZuCo Task-1 order.

## ZuCo word-level eye tracking

| File | Rows (approx.) | Key columns | Notes |
| --- | --- | --- | --- |
| `ZuCo_et_csv_data/word/{1-12}_SR.csv` | thousands each | `id`, `Sent_ID`, `Word_ID`, `Word`, measures, `WordLen` | per subject |
| `ZuCo_et_csv_data/word/word_averages.csv` | ~7k | same | subject mean |
| `ZuCo_et_csv_data/word/word_averages_v2.csv` | ~7k | same | later average; empty `Word` → `unknown` |

`Sent_ID` format: `{sentence_index}_NR`.

## ZuCo + SST joins

| File | Rows | Columns | Used for training? |
| --- | --- | --- | --- |
| `ZuCo_SST_data/ssts_ZuCo.csv` | ~400 | `sentence_id`, `sentence`, `sentiment_label` | text only |
| `ZuCo_SST_data/combined_sst_et_standard.csv` | ~400 | text + standard gaze | **yes**, `model_ZuCo_SST.py` |
| `ZuCo_SST_data/combined_sst_et_min_max.csv` | ~400 | text + min-max gaze | optional |
| `ZuCo_SST_data/train.csv` | ~320 | same as combined standard | examples / inspection |
| `ZuCo_SST_data/valid.csv` | ~40 | same | examples / inspection |
| `ZuCo_SST_data/test.csv` | ~40 | same | examples / inspection |

`sentiment_label` ∈ `{0, 1, 2}`.

## Full SST

| File | Rows (approx.) | Columns | Notes |
| --- | --- | --- | --- |
| `SST_data/stts_all_sentence_level.csv` | ~11.8k | sentence text, `POSITIVE\|NEUTRAL\|NEGATIVE` | no header |
| `SST_data/sst_et_test.csv` | word-level | `sentence_id, word_id, word, nFix, FFD, GPT, TRT, GD` | zeros; skeleton |
| `SST_data/combined_full_sst_et.csv` | ~11.8k | `sentence_id, sentence, sentiment_label, nFix, GD, TRT, FFD, GPT` | pre-split |
| `SST_data/train_full_sst.csv` | ~9.5k | same | **yes**, `model_full_SST.py` |
| `SST_data/valid_full_sst.csv` | ~1.2k | same | **yes** |
| `SST_data/test_full_sst.csv` | ~1.2k | same | **yes** |

## Predicted / reference word-level gaze

| File | Rows (approx.) | Columns | Notes |
| --- | --- | --- | --- |
| `gaze_prediction/data/prediction_test.csv` | large | `sentence_id, word_id, word, nFix, FFD, GPT, TRT, GD` | v1 predictions |
| `gaze_prediction/data/prediction_test_v2.csv` | ~192k | same | v2 predictions |
| `gaze_prediction/data/provo.csv` | ~2.6k | `sentence_id, word_id, word, nFix, FFD, GPT, TRT, fixProp` | PROVO-style; `fixProp` not `GD` |

## Scripts that only write (no committed output from this branch)

- `convert_full_SST.py` → `ZuCo_SST_data/ssts_ZuCo.csv`
- `ZuCo_SST_data/save_SST_data.py` → `output.csv` (local)
- `SST_data/convert_sst_to_et.py` → `sst_et_test.csv`
- `gaze_prediction/data/convert_zuco_data.py` → `sst_et_train_and_vaild_v2.csv` (not committed under that name)

## Label reminder

| Integer | Class | SST folder / text tag |
| --- | --- | --- |
| 0 | negative | `NEGATIVE` |
| 1 | neutral | `NEUTRAL` |
| 2 | positive | `POSITIVE` |

`SST_data/stts_all_sentence_level.csv` still has the string tags; the combined full-SST files have already been mapped to integers.
