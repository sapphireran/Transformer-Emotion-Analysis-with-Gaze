# Gaze features

Eye-tracking during reading yields a small set of per-word (or per-sentence averaged) numbers. This project concatenates five of them onto a transformer pooled vector. Definitions below follow common psycholinguistic usage and the ZuCo documentation; the **column names in this repo are not consistent** across folders.

## Feature dictionary

| Family | ZuCo sentence/word | Full SST predicted | Provo | Fed to `EyeTrackingModel`? |
| --- | --- | --- | --- | --- |
| Fixation count | `nFixations` | `nFix` | `nFix` | yes (as the first of five) |
| First fixation duration | `FFD` | `FFD` | `FFD` | yes |
| Go-past time (regression-path) | `GPT` | `GPT` | `GPT` | yes |
| Total reading / dwell time | `TRT` | `TRT` | `TRT` | yes |
| Gaze duration (first-pass) | `GD` | `GD` | — | yes |
| Single fixation duration | `SFD` | — | — | no |
| Mean pupil size | `meanPupilSize` | — | — | no |
| Skip / omission rate | `omissionRate` | — | — | no |
| Fixation proportion | — | — | `fixProp` | no |
| Word / sentence length | `WordLen`, `SentLen` | — | — | no |

### What the timings mean

Assume a reader lands on a word, maybe refixates, maybe regresses from later words, and eventually leaves to the right.

- **FFD** — duration of the *first* fixation on the word. Early lexical processing.
- **SFD** — FFD restricted to words that received *exactly one* fixation. Often missing (stored as 0 in ZuCo dumps when the word was skipped or multiply fixated).
- **GD (gaze duration)** — sum of first-pass fixations, before the eyes leave to the right. Still “first pass”.
- **GPT (go-past / regression-path time)** — from first landing until the eyes first go *past* the word to the right. Includes regressions to earlier material.
- **TRT (total reading time)** — sum of all fixations on the word, including later rereading.
- **nFixations / nFix** — how many times the word was fixated.
- **omissionRate** — sentence-level fraction of words with no fixation (skips).
- **meanPupilSize** — pupil size averaged over recorded fixations (arbitrary camera units in the raw ZuCo CSVs, mean ≈ 797).
- **fixProp** (Provo) — percentage of readers who fixated the word.

Typical inequality on a fixated content word: `FFD ≤ GD ≤ TRT` and `GD ≤ GPT`. Predicted SST features are **not** guaranteed to obey that, because they are model outputs on a standardized scale.

## Scaling in this repo

Three regimes appear:

1. **Raw ZuCo milliseconds / counts** — `ZuCo_et_csv_data/{1–12}_SR.csv`, `average_data.csv`, word-level `*_SR.csv`.
2. **Min-max to [0, 1]** — `min_max_scaled_average_data.csv`, `combined_sst_et_min_max.csv`. `sklearn.preprocessing.MinMaxScaler` over the 400-row average.
3. **Z-score (standard)** — `standard_scaled_average_data.csv`, `combined_sst_et_standard.csv`, and the five gaze columns of `combined_full_sst_et.csv`. Full-SST values have mean 0 and std 1 on all 11,853 rows; after the 80/10/10 split the train subset is only *approximately* standard (nFix mean 0.0049, std 0.99).

`gaze_prediction/data/convert_zuco_data.py` uses a fourth mapping: min-max onto **0–100**, with `nFix` scaled independently of `{FFD,GPT,TRT,GD}`.

`DataTransformer` in `utils_ZuCo.py` also implements `mean-norm` and per-subject scaling when reading `.mat` files. The committed CSVs for training were averaged **first** (across subjects) and scaled **second** (across the 400 sentences), which is not the same as scaling per subject then averaging.

## Which five numbers enter the net

From `model_full_SST.py`:

```python
eye_tracking_features = df[['nFix', 'FFD', 'GPT', 'TRT', 'GD']]
```

From `model_ZuCo_SST.py`:

```python
eye_tracking_features = df[['nFixations', 'FFD', 'GPT', 'TRT', 'GD']]
```

Order is **not** the same as the CSV layout (`GD` is stored before `TRT` in ZuCo files but is the last concat dimension). `Linear(5, 16)` does not care about human-friendly order, but any analysis script must use this order when comparing to training.

## Correlation with the 3-class label

Pearson r vs `sentiment_label` on the joined tables:

**ZuCo standard combined (n=400)**

| Feature | r |
| --- | ---: |
| `FFD` | 0.072 |
| `TRT` | 0.051 |
| `SFD` | 0.044 |
| `meanPupilSize` | −0.041 |
| `GD` | 0.036 |
| `nFixations` | 0.033 |
| `GPT` | 0.030 |
| `omissionRate` | 0.028 |

**Full SST predicted (n=11,853)**

| Feature | r |
| --- | ---: |
| `GD` | −0.060 |
| `nFix` | −0.055 |
| `FFD` | −0.055 |
| `GPT` | −0.054 |
| `TRT` | −0.052 |

Gaze is a **weak linear** cue for this label. A nonlinear concat layer can still help, but these numbers are the right prior when reading a small CV bump on 400 sentences.

## Collinearity

Measured gaze (ZuCo 400, standard combined) has the usual reading-time cluster: `nFixations`–`TRT` r ≈ 0.96, `nFixations`–`GPT` r ≈ 0.91, `TRT`–`GPT` r ≈ 0.94. `SFD` anti-correlates with `nFixations` (r ≈ −0.59) because single-fixation words are the easy ones. `meanPupilSize` is nearly orthogonal to the timings.

Predicted full-SST gaze is **almost rank-1**:

|  | nFix | GD | TRT | FFD | GPT |
| --- | ---: | ---: | ---: | ---: | ---: |
| nFix | 1 | 0.78 | 0.99 | 1.00 | 1.00 |
| GD | 0.78 | 1 | 0.68 | 0.74 | 0.74 |
| TRT | 0.99 | 0.68 | 1 | 0.99 | 0.99 |
| FFD | 1.00 | 0.74 | 0.99 | 1 | 1.00 |
| GPT | 1.00 | 0.74 | 0.99 | 1.00 | 1 |

`Linear(5, 16)` can still run; it cannot learn five independent psycholinguistic effects from these channels. `GD` is the only predicted feature with much unique variance.

Recompute with:

```bash
PYTHONPATH=examples python3 examples/scripts/analyze_gaze_features.py
```

## Missing values

Committed joined tables have **zero NaNs**. Upstream, `DataTransformer` treats MATLAB empty fields as 0, then optionally fills remaining NaNs with zeros / mean / min (`fillna='zeros'` in `read_ZuCo_mat.py`). `get_average_sentence_level.py` replaces 0 with NaN **before** the 12-subject mean so skipped measurements do not pull the mean to zero; word-level `get_average.py` currently leaves zeros in (the `replace(0, nan)` line is commented out).

## Plot gallery already in the repo

Word-level pairwise distributions (predicted / scaled space, not ZuCo ms):

- [`result/train_data_scatter_hist_plots.png`](../result/train_data_scatter_hist_plots.png)
- [`result/test_data_scatter_hist_plots.png`](../result/test_data_scatter_hist_plots.png)
- [`result/provo_data_scatter_hist_plots.png`](../result/provo_data_scatter_hist_plots.png)

Example scripts add sentence-level heatmaps under `docs/assets/`.
