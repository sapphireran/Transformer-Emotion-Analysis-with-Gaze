# Gaze-prediction track

The PNGs in `result/` are the easiest thing to over-interpret. They are
**not** plots of `SST_data/train_full_sst.csv`.

## Three spaces

| Location | Granularity | Scale | Coverage |
| --- | --- | --- | --- |
| `SST_data/*_full_sst.csv` | sentence | z-scored, mean ≈ 0, std ≈ 1 | 11,853 reviews |
| `SST_data/sst_et_test.csv` | word | **all zeros** | 191,971 words / 11,853 sentences |
| `gaze_prediction/data/prediction_test.csv` | word | raw-ish 0–50 nFix | sentences **300–399** only (100 ZuCo reviews) |
| `gaze_prediction/data/prediction_test_v2.csv` | word | same schema as the placeholder | full SST token stream |
| `gaze_prediction/data/provo.csv` | word | raw-ish + `fixProp` | 134 sentences |
| `result/*_scatter_hist_plots.png` | word | matches the raw-ish space | train / test / Provo grids |

Full-SST train `nFix` ranges about −4.1 to 3.5. The train PNG’s nFix axis
goes 0–100 with a mode near 15. Those cannot be the same column without an
inverse transform that this checkout does not store.

## `convert_zuco_data.py`

```python
scaled = ((value - min) / (max - min)) * 100
```

nFixations has its own min/max. FFD, GPT, TRT, GD share one min/max, so they
are comparable to each other after scaling but not to nFix. Input path
`training_data/word_averages_v2.csv` is not where `word_averages_v2.csv`
was committed.

## Provo

`provo.csv` adds `fixProp` (fixation proportion) and has no `GD` column.
The Provo PNG still shows a GD row — the figure was built from a slightly
different extract than the CSV next to it, or GD was derived and not saved.
Treat the PNG as a figure, the CSV as the table, and do not silently join
them.

## Why this track exists

Track B (full SST sentiment) needs a gaze sidecar for reviews ZuCo never
recorded. The predictor’s job is to fill the zero placeholder. The
sentence-level z-scored columns in `*_full_sst.csv` are the *downstream*
product of that (or an earlier) predictor after aggregation and
standardization. The exact aggregation script from word-level predictions
to those five z-scores is not in the root Python files.

## Practical rule

If a number is “about 1.6 nFixations,” you are in ZuCo reader space. If it
is “about 0 ± 1,” you are in full-SST training space. If it is “about 20
nFix,” you are in predictor/Provo space. Write the space down before
comparing to a paper table.
