# Gaze feature dictionary

The training heads use five sentence-level eye-tracking channels. ZuCo exports
three more that stay in the CSVs. Names differ slightly between the ZuCo track
and the full-SST track.

## Names in this repo

| ZuCo column | Full-SST column | What it is |
| --- | --- | --- |
| `nFixations` | `nFix` | Number of fixations, averaged over fixated words |
| `FFD` | `FFD` | First Fixation Duration |
| `GPT` | `GPT` | Go-Past Time (regression-path duration) |
| `TRT` | `TRT` | Total Reading Time |
| `GD` | `GD` | Gaze Duration (first-pass time) |
| `SFD` | — | Single Fixation Duration |
| `omissionRate` | — | Fraction of words never fixated |
| `meanPupilSize` | — | Mean pupil size on fixated words |
| `SentLen` | — | Word count of the sentence |
| `WordLen` | — | Character length of a token (word-level tables) |

Durations in the **raw** per-subject CSVs are in milliseconds (ZuCo MATLAB
export). After `StandardScaler` they are unitless z-scores. After `MinMaxScaler`
they are in `[0, 1]` per column.

## How sentence-level values are built

`utils_ZuCo.DataTransformer` walks each ZuCo `sentenceData` record:

1. Skip known bad sentence ranges for a few subject/task pairs.
2. For every word, read the scalar fields (`nFixations`, `FFD`, …). Missing or
   array-valued fields become 0.
3. Sum those word features, then **divide by the number of words that had any
   non-zero feature** (`nwords_fixated`). Words that were never looked at do
   not dilute the mean.
4. Store `SentLen = len(sent.word)` and the sentence `omissionRate`.
5. Optionally min-max / mean-normalize / z-score **across sentences for that
   subject**, or leave `scaling='raw'` (what `read_ZuCo_mat.py` does).

Subject-level raw tables are then averaged across the 12 readers
(`get_average_sentence_level.py`) and scaled once more on the **average**
table. That second scaling is what `combined_sst_et_standard.csv` uses.

Word-level tables keep one row per token. `word/get_average.py` averages the
numeric gaze columns across subjects and fills empty `Word` with `unknown`.

## What the model actually sees

Both `EyeTrackingModel` classes do:

```text
Linear(5 → 16)  →  concat with pooler (768)  →  Dropout(0.1)  →  Linear(784 → 3)
```

So the five channels are **not** aligned to tokens inside the transformer.
They are a sentence-level side vector. There is no cross-attention and no
word-to-subword mapping at train time.

That is a deliberate simplicity: it works with predicted sentence aggregates
on the full SST, where you do not have 12 ZuCo subjects.

## Scaling you will encounter

| Stage | Method | File |
| --- | --- | --- |
| Per subject, sentence | `raw` | `ZuCo_et_csv_data/{k}_SR.csv` |
| Mean of subjects | raw mean | `average_data.csv` |
| Mean of subjects | z-score | `standard_scaled_average_data.csv` |
| Mean of subjects | min-max | `min_max_scaled_average_data.csv` |
| Word averages → predictor format | stretch each column to 0–100 | `gaze_prediction/data/convert_zuco_data.py` |

`examples/scaling_check.py` rebuilds the z-score and min-max tables from
`average_data.csv` with numpy and checks they match the checked-in files.

## Informal association with polarity

On the **z-scored ZuCo** table, class means sit close to zero (as expected
after a global z-score). Neutral sentences have a slightly higher mean
`nFixations` (+0.08) and lower `GD` (−0.14). Positive sentences have a
slightly higher `FFD` / `SFD`. These gaps are small relative to the
within-class standard deviation (~1.0). Gaze is a weak, noisy covariate here,
not a substitute for the text encoder.

On the **full SST + transferred gaze** table the class means are also small,
with the positive class slightly below zero on every channel and neutral
slightly above. Because those gaze values were predicted or copied, the
association is not a clean cognitive measurement.

`examples/gaze_by_sentiment.py` reprints these means.

## Features that are exported but unused

The current classifiers never read:

- `SFD`
- `omissionRate`
- `meanPupilSize`
- `SentLen` / `WordLen`

They are useful for diagnostics (longer sentences, more skips, larger pupils)
and for a future ablation. Do not drop them from the CSVs.

## PROVO extra column

`gaze_prediction/data/provo.csv` stores `fixProp` (fixation probability, 0–100
style) instead of `GD`. Do not concatenate PROVO rows with SST predicted-gaze
rows without renaming.
