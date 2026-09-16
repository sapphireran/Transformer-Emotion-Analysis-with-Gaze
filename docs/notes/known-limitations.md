# Known limitations

These are properties of the original personal scripts and CSVs, not of the
docs layer. The examples call several of them out instead of silently
"fixing" the training files.

## 1. Subject 3 is reindexed, then averaged by row number

`DataTransformer` drops Task 1 sentences 150–249 and 399 for 0-based
subject 2 (file `3_SR.csv`). The remaining 299 sentences are written with
`id` `0..298`. `get_average_sentence_level.py` then does:

```python
pd.concat(dataframes).groupby(level=0).mean()
```

`level=0` is the pandas RangeIndex of each file, not a stable sentence id.

Consequences:

- Rows `0–149` of the average are 12-reader means of the same original
  sentences (aligned).
- Rows `150–298` mix 11 readers on original sentence *k* with subject 3 on
  original sentence *k + 100*.
- Rows `299–399` are 11-reader means (subject 3 absent).

A safer rebuild would keep the original ZuCo sentence index through the skip
list and `groupby('id')` (or an explicit sentence key), never the RangeIndex.

The examples do not rewrite `average_data.csv`. They document the issue and
operate on the tables as they are.

## 2. Full-SST test metrics use the last batch only

In `model_full_SST.py` the test loop assigns instead of extending:

```python
all_preds = preds.cpu().numpy()
all_labels = labels.cpu().numpy()
```

Validation in the same file uses `.extend` and is trustworthy. Test prints
are not. Any paper table that quoted this script's "Test Acc" needs a rerun
with `extend`.

## 3. Comments disagree with the code

- Best-checkpoint comment says F1; the comparison is `val_acc`.
- `get_average_sentence_level.py` comments say the folder is `et_csv_data`;
  the committed sentence CSVs sit in `ZuCo_et_csv_data/`.
- `spilt.py` is a typo of `split.py`.

## 4. Splits are not stratified

`train_test_split` is random with `random_state=42` and no
`stratify=`. ZuCo valid ended up 7 / 14 / 19. The ZuCo training script
sidesteps this by using `StratifiedKFold` on the full 400. The stored split
is still what you get if you train on `train.csv` by hand.

## 5. Gaze predictor leakage risk on Track B

Full-SST "gaze" is predicted from text. A strong predictor can smuggle
lexical frequency, length, and surprisal into `nFix` / `TRT`. Concatenating
those channels to RoBERTa is then partly a second read of the same
sentence, not an independent cognitive measurement. Track A (human ZuCo
gaze) is the only place that objection does not apply — and even there,
sentence-level averages wash out a lot of word-level structure.

## 6. Word-level tables are unused at train time

`word_averages_v2.csv` (7,129 rows) and `prediction_test_v2.csv` (191,971
rows) never enter `EyeTrackingModel`. Fusion is sentence-level only. If the
interesting signal is "readers lingered on *failing* and *decency*", the
current architecture cannot see it except as a bump in the sentence mean.

## 7. No training-time seed control

`model_*.py` never sets `torch.manual_seed`. Fold indices are deterministic;
weights are not. Quote ranges, or set seeds before claiming a 0.5-point
fused-vs-text delta.

## 8. Missing upstream inputs

ZuCo `.mat` files, the SST `all/` directory, and the gaze-predictor training
code are not in this repository. You can train and analyze from the derived
CSVs; you cannot regenerate those CSVs bit-for-bit from official dumps
without extra downloads and a few path fixes.

## 9. Hardcoded Windows path

`utils_ZuCo.get_matfiles` defaults to `\\ZuCo_mat_data\\`. On Linux this
becomes a single directory name with backslashes unless the caller passes
`subdir`.

## 10. Tokenization mismatch

ZuCo word tables keep the transformer's lightly cleaned tokens
(`presents`, `a`, `good`, …). Full-SST predicted words come from NLTK
`word_tokenize` plus an `[A-Za-z]+` filter, which drops numbers, punctuation,
and tokens like `'s`. Sentence 0 of SST becomes 28 predicted words starting
`The Rock is destined …` and loses `21st` and possessives. Do not align
ZuCo `Word_ID` with SST `word_id` by position across tracks.
