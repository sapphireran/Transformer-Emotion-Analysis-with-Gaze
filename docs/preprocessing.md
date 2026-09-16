# Preprocessing

The checked-in CSVs are the end of a small pipeline. Several stages need
data that is **not** in git (ZuCo `.mat` files, the SST `all/` text folders).
This page describes what the scripts do when those inputs are present, and
what you can still do without them.

## 1. MATLAB → per-subject sentence or word tables

`utils_ZuCo.DataTransformer` reads `ZuCo_mat_data/<task>/*.mat` (12 files).
The default path uses Windows backslashes (`\\ZuCo_mat_data\\`), so on this
Linux clone you would pass a POSIX `subdir` or edit the helper.

For each subject it walks `sentenceData`, skips a handful of known-bad
sentence indices (different per task/subject), and either:

- **sentence level:** accumulate word measures, divide by the number of
  words that had a non-zero feature vector, attach `SentLen` and
  `omissionRate`
- **word level:** one row per token with `Sent_ID`, `Word_ID`, `Word`,
  measures, `WordLen`

NaNs become zeros / mean / min (`fillna`). Inf rows are dropped on sentence
level. Scaling is `raw`, `min-max`, `mean-norm`, or `standard`.

`read_ZuCo_mat.py` instantiates the transformer for Task 1, sentence, raw,
zeros, and writes `et_csv_data/{1-12}_SR.csv`. The copies in this repo live
under `ZuCo_et_csv_data/`.

## 2. Average across readers

`get_average_sentence_level.py` (sentence) and
`ZuCo_et_csv_data/word/get_average.py` (word) concatenate the 12 subject
frames and take a mean.

The sentence script:

1. Replaces 0 with NaN on every column except the first, so skipped/missing
   values do not drag the mean to zero.
2. `concat(...).groupby(level=0).mean()` — this groups on the **RangeIndex**,
   not on a sentence-id column.
3. Writes `average_data.csv`, then sklearn `MinMaxScaler` /
   `StandardScaler` copies.

Because subject 3 only has 299 rows, ids 299–399 are 11-reader means. Worse,
subject 3's rows 150–298 are **later original sentences** that were
reindexed after the skip. Index-based averaging therefore mixes readers on
different sentences from row 150 onward. Details and a safer join key are in
[known limitations](notes/known-limitations.md).

The word script averages `nFixations, meanPupilSize, GD, TRT, FFD, SFD, GPT`
and stitches `id, Sent_ID, Word_ID, Word, WordLen` from subject 1.

## 3. SST labels for the 400 ZuCo reviews

`convert_full_SST.py` and `ZuCo_SST_data/save_SST_data.py` read
`all/NEGATIVE|POSITIVE|NEUTRAL/*.txt`, take the filename stem as
`sentence_id`, and write `ssts_ZuCo.csv`. Those `all/` folders are not in
git; `ssts_ZuCo.csv` is.

Someone then joined `ssts_ZuCo.csv` to the scaled sentence-average tables to
produce:

- `ZuCo_SST_data/combined_sst_et_standard.csv`
- `ZuCo_SST_data/combined_sst_et_min_max.csv`

There is no join script in the repo root. The join key is `sentence_id` /
`id` in `0..399` order, which matches `ssts_ZuCo.csv` after sorting.

## 4. Stored 80/10/10 splits

`ZuCo_SST_data/spilt.py` and `SST_data/spilt.py` (the filename is a typo of
"split") do `train_test_split(..., test_size=0.2, random_state=42)` and then
split the 20% in half. They do **not** stratify on `sentiment_label`.

That is why ZuCo valid is 7 / 14 / 19 (neg/neu/pos) while the parent is
123 / 137 / 140. The examples audit records this drift. The ZuCo training
script avoids the issue by using stratified 5-fold CV instead of these
files.

Full SST is large enough that the random split stays close to 39 / 19 / 42.

## 5. Full SST word skeleton and predicted gaze

`SST_data/convert_sst_to_et.py` tokenizes each SST sentence with NLTK
`word_tokenize`, keeps `[A-Za-z]+` tokens, and writes a word table with
zeroed `nFix, FFD, GPT, TRT, GD`. Empty sentences become a single `unknown`
token. The result is `sst_et_test.csv` (191,971 rows).

`gaze_prediction/data/convert_zuco_data.py` is a different converter: it
min-max scales ZuCo word averages into a 0–100 band for a training export.

`gaze_prediction/data/prediction_test_v2.csv` is the populated prediction
for every SST sentence (same 191,971 rows). Sentence-level
`combined_full_sst_et.csv` is the z-scored aggregate of those words (or of
an earlier predictor version) joined back to SST labels. The aggregation
script itself is not checked in; the outputs are.

## 6. What the training scripts do at load time

Both `model_*.py` files:

1. Read the CSV with pandas
2. Take the five gaze columns as a numpy array
3. Build a Hugging Face `Dataset` from `sentence` + `sentiment_label`
4. Tokenize with `max_length=128`, padding to that length
5. Wrap rows in `CustomDataset` so each item has `input_ids`,
   `attention_mask`, `labels`, `eye_tracking_features`

There is no further gaze normalization inside the training loop. Whatever
scaling the CSV already has is what the `Linear(5, 16)` layer sees.

## Practical path from a fresh clone

You do **not** need steps 1–3 to run examples or to train, because the
derived CSVs are in git. You need steps 1–3 only if you want to rebuild
those CSVs from official ZuCo MATLAB dumps.
