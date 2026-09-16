# Gaze / eye-tracking features

Eye-tracking gives a time-resolved trace of where a reader looked, for how long, and whether they went back. This project reduces that trace to a handful of classic reading measures and (for the transformer) keeps five of them at sentence level.

## Measures used in this repo

Durations below are typically milliseconds in the raw ZuCo export. After averaging and scaling they are unitless.

### `nFix` / `nFixations`

Count of fixations on the word (or the mean count per fixated word when aggregated to a sentence). Higher values usually mean the region was revisited or needed more samples.

The full-SST tables use `nFix`. ZuCo tables use `nFixations`. Same idea, different name. The example library treats them as aliases.

### `FFD` — first fixation duration

Duration of the *first* fixation that lands on the word. Often used as an early processing measure: lexical access, surprise, or local difficulty.

### `SFD` — single fixation duration

Duration when the word received exactly one fixation. If the word was skipped or fixated more than once, this is often zero or missing in raw exports. Present on ZuCo tables, absent from the five-feature SST fusion set.

### `GD` — gaze duration (first-pass time)

Sum of fixation durations on the word during first pass — from first entering the word until the eyes leave it to the right, not counting later regressions back. A standard late-but-still-first-pass measure.

### `TRT` — total reading time

Sum of *all* fixation durations on the word, including regressions. `TRT >= GD` when both are defined.

### `GPT` — go-past time (regression-path duration)

Time from first entering the word until the eyes first move *past* it to the right. Includes fixations on the word and any regressions to earlier words before the reader continues. Sensitive to integration problems.

### `omissionRate`

Fraction of words in the sentence that received no fixation. High omission can mean skimming, high predictability, or tracking loss. Sentence-level only.

### `meanPupilSize`

Mean pupil size during recorded fixations. Pupil size is a rough proxy for arousal and cognitive load, but it is also affected by luminance and recording quality. Handle as an auxiliary feature, not as "emotion" by itself.

### Length covariates

- `SentLen` — number of words in the sentence (ZuCo sentence tables)
- `WordLen` — character length of the token (ZuCo word tables)

Length correlates with almost every duration measure. If you compare raw means across classes without controlling for length, you will overstate "sentiment effects."

## Sentence-level aggregation

`utils_ZuCo.DataTransformer` walks each sentence's word objects and, for sentence-level output:

1. sums word-level `nFixations`, `meanPupilSize`, `GD`, `TRT`, `FFD`, `SFD`, `GPT`
2. divides those sums by the number of words that had any reported fixation
3. stores `SentLen` and the sentence `omissionRate` as-is

Words with a missing feature become `0` before the sum. That is a modeling choice, not a claim that the reader spent zero milliseconds.

Subject-level CSVs are later averaged across the 12 readers (`get_average_sentence_level.py` and `ZuCo_et_csv_data/word/get_average.py`). Zeros can be replaced with NaN before the mean so a skipped word does not drag the subject average to zero. The sentence-level averager does that replacement; the word-level v2 script currently does not (the replacement is commented out).

## Scaling

Three scalers appear in the tree:

| Name | Formula (per column) | Files |
| --- | --- | --- |
| `raw` | unchanged | `ZuCo_et_csv_data/{k}_SR.csv` |
| `min-max` | `(x - min) / (max - min)` | `*_min_max*` |
| `standard` / z-score | `(x - mean) / std` | `*_standard*` |
| 0–100 min-max | min-max then `* 100` | `gaze_prediction/data/convert_zuco_data.py` |

`model_ZuCo_SST.py` reads the **standard** combined table. `model_full_SST.py` reads whatever scale is already stored in `SST_data/*_full_sst.csv` (values there are not raw milliseconds; many look standardized or model-projected).

Never mix a raw subject CSV with a z-scored combined CSV in the same tensor without re-scaling.

## Missing values and special cases

`DataTransformer` has hard-coded skips for known bad sentence ranges in specific ZuCo subject/task pairs (for example task1 subject 2 drops indices 150–249 and 399). Those ranges come from incomplete recordings, not from sentiment.

Fill options in the transformer: `zeros`, `mean`, `min`. The checked-in reader uses `fillna='zeros'` and `scaling='raw'`.

`check_inf` drops any sentence row that contains `+inf` or `-inf`.

## What gaze is *not* in this project

- It is not a facial-expression or video "emotion" signal.
- It is not EEG. ZuCo has EEG, but this repo's CSVs are eye-tracking only.
- Sentence-level means discard word order. A long regression on the last adjective and a long regression on the first noun look the same after averaging.
- Full-SST gaze columns are projected. Do not interpret a high `TRT` there as "this reviewer actually stared at the sentence."

## Feature sets the examples use

The example library exposes three named sets:

| Name | Columns | Typical file |
| --- | --- | --- |
| `sst5` | `nFix`, `GD`, `TRT`, `FFD`, `GPT` | `SST_data/train_full_sst.csv` |
| `zuco5` | `nFixations`, `FFD`, `GPT`, `TRT`, `GD` | combined ZuCo (matches the training scripts) |
| `zuco_full` | `omissionRate`, `nFixations`, `meanPupilSize`, `GD`, `TRT`, `FFD`, `SFD`, `GPT` | combined ZuCo |

`examples/gaze_feature_report.py` prints per-class means and a simple association score for each set.
