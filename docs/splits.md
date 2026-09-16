# Splits

## Full SST

`SST_data/spilt.py` (typo) loads `combined_full_sst_et.csv` and runs:

```python
train, valid_test = train_test_split(df, test_size=0.2, random_state=42)
valid, test = train_test_split(valid_test, test_size=0.5, random_state=42)
```

Committed sizes: **9482 / 1185 / 1186** = 11853. Ids are disjoint and union
to the combined id set. `model_full_SST.py` uses these three files.

Text is not unique. The combined table has 11,840 unique strings / 11,853
rows. Train has 11 extra duplicate-text rows. Because the split is on rows:

- 1 string is in both train and valid
- 1 string is in both train and test
- 0 strings are in both valid and test

`examples/05_split_fingerprint.py` prints those two leaked reviews. Ids
differ, labels match in the cases we inspected, gaze values match — they are
duplicate rows of the same review, not relabelings.

No stratification: valid/test label shares are close to train but not
locked. Majority class on train is positive (2) at ~41.5%.

## ZuCo-SST

`ZuCo_SST_data/spilt.py` is the same recipe on `combined_sst_et_standard.csv`
→ 320 / 40 / 40. Those files **are not loaded** by `model_ZuCo_SST.py`.
The trainer does:

```python
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
for train_index, test_index in kf.split(df_tokenized, df_tokenized['sentiment_label']):
    ...
```

There is no held-out test outside the five folds. The printed “Validation”
metrics are fold test scores after 20 epochs, then averaged.

The leftover 40-row valid CSV is a warning label: class counts `{0: 7, 1: 14, 2: 19}`.
Do not use it as a model-selection split without noticing the 7 negatives.

Fold files are id-disjoint with each other (they come from the same
row-wise split). The k-fold is a different partition.

## Leakage that is *not* a split bug

Track B’s gaze columns are predicted. If the predictor was trained on ZuCo
including sentences that also appear in full SST, those three overlapping
strings are a theoretical leak. In practice the overlap is three rows and
the predictor’s training details are not in this checkout. Flag it; do not
overclaim.

## Order effects

`utils_ZuCo.split_data` exists to split Task 1 into first/second half
“to control for order effects.” Nothing in the trainers calls it. The
committed averages ignore presentation order.
