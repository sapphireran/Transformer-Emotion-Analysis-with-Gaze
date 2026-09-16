# Gaze-prediction track

The `result/*.png` scatter/histogram grids are in the **raw millisecond /**
**count** space used by `gaze_prediction/data/*.csv` and Provo, *not* the
z-scored `SST_data/*_full_sst.csv` tables (those have mean ≈ 0, std ≈ 1).
Do not paste a z-score next to those plots and call it the same feature.

## `SST_data/sst_et_test.csv`

Word-tokenized SST with gaze columns filled by zeros. This is the blank
form a predictor is supposed to fill. Committed file:
191971 word rows, 11853 sentences,
nFix all zero? True.

## `gaze_prediction/data/prediction_test.csv`

1751 word rows covering **100 sentences**
with ids 300…399.
That is the last 100 ZuCo-SST reviews, not the 11.8k full SST test split.
nFix here lives on a 0–50 scale (mean 21.49), matching the train/test PNG histograms.

## `gaze_prediction/data/prediction_test_v2.csv`

Large sister table (same schema as the zero placeholder). Used as the
word-level dump of predicted gaze over the whole SST token stream.

## `gaze_prediction/data/provo.csv`

2659 word rows, 134 sentences, extra
`fixProp` column that the ZuCo-style five-tuple does not have.

### prediction_test feature means

| feature | mean | std |
| --- | --- | --- |
| nFix | 21.4945 | 6.3779 |
| FFD | 4.3715 | 0.5645 |
| GPT | 8.2804 | 5.4242 |
| TRT | 6.9224 | 2.3854 |
| GD | 5.1594 | 1.2169 |

### provo feature means

| feature | mean | std |
| --- | --- | --- |
| nFix | 15.1000 | 9.4200 |
| FFD | 3.1900 | 1.4200 |
| GPT | 6.3500 | 5.9100 |
| TRT | 5.3100 | 3.6400 |
| fixProp | 67.0600 | 26.0600 |

prediction_test sentence ids (n=100): 300…399
