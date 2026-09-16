# Eye-tracking feature glossary

Definitions below follow the usual reading-research sense (Rayner; ZuCo documentation). Units in the *raw* subject CSVs are milliseconds for time measures and counts for fixations. After `get_average_sentence_level.py`, the joined ZuCo tables are either min-max in `[0, 1]` or z-scored. Full-SST columns are on whatever scale the gaze predictor emitted (often a 0–100 style min-max; see `gaze_prediction/data/convert_zuco_data.py`).

## Features the classifiers actually use

`model_ZuCo_SST.py` reads:

```python
df[['nFixations', 'FFD', 'GPT', 'TRT', 'GD']]
```

`model_full_SST.py` reads the same five ideas under slightly different names:

```python
df[['nFix', 'FFD', 'GPT', 'TRT', 'GD']]
```

Order in the tensor is therefore **not** the same across scripts:

| Index | ZuCo combined table | Full SST table |
| ---: | --- | --- |
| 0 | nFixations | nFix |
| 1 | FFD | FFD |
| 2 | GPT | GPT |
| 3 | TRT | TRT |
| 4 | GD | GD |

Do not dump a ZuCo row into the full-SST model by column index without reordering. Name-based joins are safe; positional copies are not.

### nFix / nFixations

Number of fixations that landed on the word (word level) or the mean number of fixations per fixated word (sentence average in `utils_ZuCo.py`). High nFix usually means the region was looked at more than once: difficulty, importance, or a regression target.

At sentence level the transformer already knows token count, so nFix is not just “longer sentence.” It is closer to “how often the eyes stopped.”

### FFD — First Fixation Duration

Duration of the *first* fixation on a word. Classic first-pass measure. Sensitive to lexical frequency, predictability, and early semantic clash. A sarcastic adjective that is orthographically normal can still show a long FFD if the sense is unexpected.

### GD — Gaze Duration

Sum of all first-pass fixations on the word *before* the eyes leave it to the right. Also called first-pass time. GD ≥ FFD. If the reader makes a second fixation before leaving, GD grows and FFD does not.

### GPT — Go-Past Time (regression-path duration)

Time from first entering the word until the eyes move *past* it to the right, including regressions to earlier words. GPT ≥ GD. A large GPT − GD gap is the usual signature of a garden path or a polarity flip (“not … bad”).

### TRT — Total Reading Time

Sum of all fixation durations on the word, including later rereads. TRT ≥ GD. High TRT with ordinary FFD means the problem showed up on the second pass.

## Features present in the CSVs but unused by the training scripts

### SFD — Single Fixation Duration

Duration when the word received exactly one fixation. Missing (stored as 0) when the word was skipped or fixated more than once. Noisy after sentence averaging, which is why it is not in the five-feature slice.

### meanPupilSize

Average pupil diameter during fixations on the region. Related to cognitive load and luminance. ZuCo was not a pupil-controlled emotion study; treat this as exploratory only.

### omissionRate

Fraction of words in the sentence that received no fixation. Fast, confident readers omit more function words. Very high omission can also mean track loss. `utils_ZuCo.py` copies `sent.omissionRate` directly from the MATLAB struct.

### SentLen / WordLen

Token count of the sentence, or character length of the word. Useful covariates for a linear baseline; redundant for a transformer that already sees the tokens.

### fixProp (PROVO only)

Proportion of readers who fixated the word. Not a duration. Not in the sentiment fusion head.

## How sentence-level numbers are computed

In `DataTransformer` (`utils_ZuCo.py`), sentence mode walks each word, sums the numeric fields, and divides by `nwords_fixated` — the count of words that had at least one non-zero feature. Words that are entirely zero do not inflate the denominator. `SentLen` and `omissionRate` are not averaged that way; they are sentence properties.

Subject 3 (zero-based index 2) in Task 1 skips MATLAB sentences `[150, 249]` and `399`. That is why `3_SR.csv` has 299 rows.

## Scaling

| Mode | Formula (per column) | Used in |
| --- | --- | --- |
| `raw` | as recorded | `read_ZuCo_mat.py` subject dumps |
| `min-max` | `(x - min) / (max - min)` | `combined_sst_et_min_max.csv` |
| `standard` | `(x - mean) / std` | `combined_sst_et_standard.csv` (default train file) |
| `mean-norm` | `(x - mean) / (max - min)` | supported in `DataTransformer`, not used for the joined tables |

`get_average_sentence_level.py` replaces `0` with NaN in every column except `id` **before** the mean, so unfixated words do not pull the subject mean to zero. After the mean, sklearn `MinMaxScaler` / `StandardScaler` fit on the 400-row table.

`gaze_prediction/data/convert_zuco_data.py` uses a different convention: nFix is min-max scaled to `[0, 100]` by itself; `{FFD, GPT, TRT, GD}` share one min and max, then also map to `[0, 100]`. Predicted SST word files follow that 0–100-ish style.

## Missing values

| Stage | Policy |
| --- | --- |
| MATLAB extract (`fillna='zeros'`) | missing attributes → 0 |
| Word averages | `Word` null → `unknown`; numeric null → 0 |
| Subject average | 0 → NaN, then mean, then scale |
| SST placeholder words | `convert_sst_to_et.py` writes five literal zeros |

A row of five zeros after scaling is *not* the same as a raw zero. On the standard-scaled ZuCo table, a true mean vector is near 0, not near the scaled image of raw 0.

## Sanity checks you can run

```bash
python3 examples/feature_stats.py
python3 examples/gaze_fusion_demo.py
```

`feature_stats.py` prints per-column min / mean / max / zero-rate for the joined tables. `gaze_fusion_demo.py` builds the same 5 → 16 → concat-with-768 toy tensor the real model uses, with NumPy if present and a tiny pure-Python fallback otherwise.
