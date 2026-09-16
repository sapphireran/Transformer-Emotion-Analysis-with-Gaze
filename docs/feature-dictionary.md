# Feature dictionary

Eye-tracking names in this checkout are a mix of ZuCo’s MATLAB structs,
sentence-level aggregates, and a predictor’s 0–100 scaled outputs. The
same five-letter acronym is **not** on the same scale in every folder.

## Sentence-level (ZuCo readers)

From `utils_ZuCo.DataTransformer` at `level='sentence'`. Word-level
attributes are summed over fixated words and divided by the number of words
that had a non-zero feature vector (“nwords_fixated”). `SentLen` and
`omissionRate` are sentence properties.

| Column | Meaning in this repo |
| --- | --- |
| `SentLen` | Number of word tokens in the MATLAB sentence (`len(sent.word)`), not whitespace split of the SST string |
| `omissionRate` | ZuCo’s per-sentence omission rate (words never fixated / words in the sentence) |
| `nFixations` | Mean fixation count per fixated word |
| `meanPupilSize` | Mean pupil size per fixated word (arbitrary eye-tracker units; subject 12 is an outlier) |
| `GD` | Gaze duration — first-pass time on the word before leaving it |
| `TRT` | Total reading time — all fixations on the word, including regressions |
| `FFD` | First fixation duration |
| `SFD` | Single fixation duration (only defined when the word received exactly one fixation; often 0) |
| `GPT` | Go-past time / regression-path duration |

`SFD` is *not* fed to `EyeTrackingModel`. Neither is pupil size or omission
rate.

## Word-level extras

`ZuCo_et_csv_data/word/{1..12}_SR.csv` adds:

| Column | Meaning |
| --- | --- |
| `Sent_ID` | `{packed_or_original_index}_NR` for Task 1. Subject 3’s ids are packed. |
| `Word_ID` | Position inside the sentence |
| `Word` | Token after stripping punctuation; empty → later filled as `unknown` |
| `WordLen` | `len(token)` after that strip (0 for punctuation-only tokens) |

Skip rate is `nFixations == 0`. Subject 3 skips ~52.5% of tokens; the other
readers sit roughly in 0.21–0.39. See `examples/09_word_skips.py`.

## Full-SST training columns

`SST_data/train_full_sst.csv` (and valid/test/combined):

| Column | Notes |
| --- | --- |
| `sentence_id` | Integer id; unique inside the combined table |
| `sentence` | Review string (a handful of duplicate texts with different ids) |
| `sentiment_label` | 0 negative, 1 neutral, 2 positive |
| `nFix, GD, TRT, FFD, GPT` | Already **z-scored** over the combined 11.8k rows (combined mean ≈ 0, std ≈ 1) |

There is no `SFD`, no pupil, no omission rate. Combined-table mean of `nFix`
is ~10⁻¹⁶, not “about 1.6 fixations per word.” If a plot shows nFix peaking
near 20, it is not this table.

## Predictor / Provo space (raw-ish)

`gaze_prediction/data/prediction_test.csv` and `provo.csv`, and the PNG
grids in `result/`, live on a **different scale**:

- nFix roughly 0–50 (means around 20)
- FFD roughly 0–8
- GPT/TRT/GD in small millisecond-derived units after whatever min-max ×100
  `convert_zuco_data.py` applied

`SST_data/sst_et_test.csv` is the blank form: same word-level schema, every
gaze cell 0. 191,971 word rows, 11,853 sentences.

`gaze_prediction/data/convert_zuco_data.py` min-max scales nFixations on its
own min/max, and scales FFD/GPT/TRT/GD on a *shared* min/max, then multiplies
by 100. That is not z-scoring and not the ZuCo standard scaler.

## Label map

`convert_full_SST.py` / `ZuCo_SST_data/save_SST_data.py`:

```
NEGATIVE → 0
NEUTRAL  → 1
POSITIVE → 2
```

`stts_all_sentence_level.csv` still has the string names and **no header**;
pandas will eat the first review as column names unless you pass `header=None`.

## Scaling recipes already in the checkout

| File | Scaling |
| --- | --- |
| `ZuCo_et_csv_data/average_data.csv` | Unscaled mean across readers (row index!) |
| `min_max_scaled_average_data.csv` | MinMax on that mean |
| `standard_scaled_average_data.csv` | z-score on that mean |
| `combined_sst_et_standard.csv` | Standard scaled sentence gaze joined to the 400 labels |
| `combined_sst_et_min_max.csv` | Min-max sibling of the same join |
| `SST_data/*_full_sst.csv` | z-scored predicted sidecar |

`utils_ZuCo.DataTransformer` also implements `mean-norm` and `raw`, and
fillna in `{zeros, mean, min}`. `read_ZuCo_mat.py` as committed uses
`scaling='raw', fillna='zeros'` and writes `et_csv_data/` (a directory name
that is **not** the committed `ZuCo_et_csv_data/`).
