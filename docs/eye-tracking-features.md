# Eye-tracking features

This repo uses standard reading-research duration measures. Values in the ZuCo sentence tables are **milliseconds averaged over fixated words**, then optionally z-scored or min–max scaled. Values in the predicted SST word tables are **not** milliseconds; they have been squeezed into a roughly 0–100 range (see `gaze_prediction/data/convert_zuco_data.py`).

## Glossary

| Name | Full name | What it measures | In fusion vector? |
| --- | --- | --- | --- |
| nFix / nFixations | Number of fixations | How many times the eyes landed on the word (then averaged to sentence level) | yes (index 0) |
| FFD | First fixation duration | Duration of the **first** fixation on the word during first-pass reading | yes (index 1) |
| GPT | Go-past time (regression-path duration) | Time from first entering a word until the eyes move **past** it to the right, including leftward regressions | yes (index 2) |
| TRT | Total reading time | Sum of all fixation durations on the word, including re-readings | yes (index 3) |
| GD | Gaze duration (first-pass time) | Sum of first-pass fixations on the word before leaving it | yes (index 4) |
| SFD | Single fixation duration | Duration when the word was fixated exactly once in first pass | no |
| meanPupilSize | Mean pupil size | Pupil diameter proxy; arousal / luminance confounded | no |
| omissionRate | Omission rate | Fraction of words in the sentence with no recorded fixation | no |
| SentLen | Sentence length | Token count of the stimulus | no |
| WordLen | Word length | Character length of the token | no (word tables only) |
| fixProp | Fixation proportion | Provo-only: how often the word was fixated across readers | no |

## Relationships you should expect

On raw millisecond data, roughly:

```
FFD  ≤  SFD  ≤  GD  ≤  TRT  ≤  GPT     (often, not as a hard inequality on every row)
```

- **FFD** is a single fixation.
- **GD** can include extra first-pass fixations on the same word.
- **TRT** adds later passes.
- **GPT** adds time spent on earlier words after a regression.

That is why the predicted-gaze scatter plots in `result/` show tight nFix–TRT and TRT–GD clouds: the predictor is copying those correlations.

Sentence-level ZuCo features are **means over words that had a fixation**, not sums over the whole sentence. A long sentence with many skipped function words can still have a moderate mean TRT. `omissionRate` is the place to look for skipping.

## Scaling

| Table | Scaling | Typical magnitude |
| --- | --- | --- |
| `ZuCo_et_csv_data/{1-12}_SR.csv` | raw | FFD ~ 90–130 ms, TRT ~ 110–400 ms |
| `ZuCo_et_csv_data/average_data.csv` | raw subject mean | FFD mean 116.9 ms, TRT mean 202.6 ms |
| `combined_sst_et_standard.csv` | z-score across 400 sentences | mean ≈ 0, std ≈ 1 |
| `combined_sst_et_min_max.csv` | min–max across 400 sentences | `[0, 1]` |
| `SST_data/*_full_sst.csv` | z-score across 11,853 sentences | mean ≈ 0, std ≈ 1 |
| `gaze_prediction/data/prediction_test_v2.csv` | min–max × 100 (predictor units) | nFix ~ 20, FFD ~ 4.4 |

`model_ZuCo_SST.py` is wired to the **standard** ZuCo table. `model_full_SST.py` is wired to the **z-scored** SST table. Do not mix a min–max gaze matrix with a model that was trained on z-scores without re-fitting.

Zeros in raw subject files often mean "no fixation / missing attribute", not a true zero-millisecond reading. `get_average_sentence_level.py` converts those zeros to NaN before averaging so a skipped word does not drag the subject mean to 0. `DataTransformer` with `fillna='zeros'` does the opposite on MATLAB export.

## What the fusion model actually sees

From `model_ZuCo_SST.py`:

```python
eye_tracking_features = df[['nFixations', 'FFD', 'GPT', 'TRT', 'GD']]
```

From `model_full_SST.py`:

```python
eye_tracking_features = df[['nFix', 'FFD', 'GPT', 'TRT', 'GD']]
```

Order is **not** the same as the CSV column order on SST tables (`nFix, GD, TRT, FFD, GPT` on disk vs `nFix, FFD, GPT, TRT, GD` in the tensor). That is intentional in the loaders: both models present FFD then GPT then TRT then GD after the fixation count. If you add a new script, reuse `examples/common.py` rather than taking `df.filter(like=...)` in file order.

## Label-conditioned profiles (ZuCo, z-scored)

Recomputed by `examples/gaze_feature_profiles.py` on `combined_sst_et_standard.csv`. These are observational, not causal:

- Negative and positive sentences do **not** separate cleanly on any single feature. Overlap is large.
- Extreme positive z-scores on nFixations / TRT / GPT tend to be long or re-read sentences (see sentence id 3, "Slow, silly and unintentionally hilarious.", which is labeled neutral but has very high nFixations and TRT).
- `omissionRate` (unused by the model) is worth plotting anyway: high omission with low nFixations is a different reading pattern from low omission with high TRT.

Gaze-only logistic regression on the 400 ZuCo rows is only a weak classifier. That is expected — the research bet is that gaze is a **side channel**, not a replacement for the text encoder. `examples/dummy_baseline.py` prints that number so a fusion run has something to beat besides majority class.

## Predicted vs measured

Do not evaluate a model trained on ZuCo measured gaze against SST predicted gaze and call it a replication. The predicted tables:

- come from a different unit system,
- are word-level (SST training is sentence-level),
- and inherit the predictor's bias toward linear nFix–TRT coupling visible in `result/test_data_scatter_hist_plots.png`.

If you need a fair text-only vs fusion comparison on SST, keep the transferred gaze **fixed** across `roberta` and `roberta_eye_tracking` so the only change is the extra Linear.
