# Data pipeline

Two parallel pipelines feed the two training scripts. Nothing in the
example scripts re-runs the `.mat` extraction; they only read the
checked-in CSVs.

```
                    ZuCo .mat (not in repo)
                            │
                            ▼
                   utils_ZuCo.DataTransformer
                     (sentence or word)
                            │
            ┌───────────────┴────────────────┐
            ▼                                ▼
   ZuCo_et_csv_data/{1-12}_SR.csv   ZuCo_et_csv_data/word/{1-12}_SR.csv
            │                                │
            ▼                                ▼
   average + min-max / z-score      word_averages[_v2].csv
            │                                │
            ▼                                ▼
   join with ssts_ZuCo.csv          convert_zuco_data.py
            │                                │
            ▼                                ▼
   combined_sst_et_{standard,min_max}.csv    predictor-schema CSVs
            │                                │
            ▼                                ▼
      model_ZuCo_SST.py              gaze_prediction/data/*.csv
```

```
   SST text (stts_all_sentence_level.csv)
            │
            ├─ convert_sst_to_et.py ──► sst_et_test.csv (zeros)
            │
            ▼
   combined_full_sst_et.csv   (text + projected 5-d ET)
            │
            ▼
   SST_data/spilt.py   (80 / 10 / 10, seed 42)
            │
            ▼
   train_full_sst.csv / valid_full_sst.csv / test_full_sst.csv
            │
            ▼
      model_full_SST.py
```

## Stage A — ZuCo `.mat` → per-subject CSV

`read_ZuCo_mat.py` builds a `DataTransformer('task1', level='sentence',
scaling='raw', fillna='zeros')` and writes `et_csv_data/{i+1}_SR.csv`
for `i in 0..11`.

Caveats (see [known-quirks.md](known-quirks.md)):

- `get_matfiles()` defaults to a Windows-style subdir
  `\\ZuCo_mat_data\\`.
- The writer path is `et_csv_data/`, but the committed files live in
  `ZuCo_et_csv_data/`.
- Subject index `2` yields 299 sentences, not 400.

Word-level extraction is the same class with `level='word'`. Those CSVs
are already committed under `ZuCo_et_csv_data/word/`.

## Stage B — average and scale

`get_average_sentence_level.py`:

1. Reads `et_csv_data/{1-12}_SR.csv`.
2. Replaces `0` with `NaN` in every column except the first.
3. Means aligned rows.
4. Fits `MinMaxScaler` and `StandardScaler` on the averaged numeric
   block.
5. Writes `min_max_scaled_average_data.csv` and
   `standard_scaled_average_data.csv`.

The committed copies of those files already live in `ZuCo_et_csv_data/`.

Word-level averaging (`ZuCo_et_csv_data/word/get_average.py`) groups by
row index, not by `(Sent_ID, Word_ID)`. That is safe only because every
complete subject file is the same length and the same word order.

## Stage C — attach sentiment labels

`convert_full_SST.py` (repo root) and `ZuCo_SST_data/save_SST_data.py`
walk `all/NEGATIVE|POSITIVE|NEUTRAL/*.txt`, map folder → `{0,1,2}`, and
write `ssts_ZuCo.csv` (or `output.csv` in the folder-local copy).

The join onto averaged ET is not a committed script; the products are
`combined_sst_et_standard.csv` and `combined_sst_et_min_max.csv`. Ids
match: sentence `k` in `ssts_ZuCo.csv` is row `k` of the ET average.

## Stage D — full SST + projected gaze

`SST_data/combined_full_sst_et.csv` already has text, labels, and five
ET columns. `SST_data/spilt.py` shuffles with `random_state=42` into
the 80/10/10 files `model_full_SST.py` consumes.

`SST_data/convert_sst_to_et.py` tokenizes each SST sentence with NLTK
`word_tokenize`, keeps `[A-Za-z]+` tokens, and writes a word-level table
of zeros. Filling those zeros is the gaze-prediction project
(`gaze_prediction/data/prediction_test_v2.csv`).

## Stage E — tensors the training loop sees

Both model scripts:

1. Read the sentence CSV with pandas.
2. Pull the 5 ET columns as a float matrix.
3. Wrap `(sentence, sentiment_label)` in a Hugging Face `Dataset`.
4. Tokenize with `bert-base-uncased` or `roberta-base`,
   `max_length=128`, `padding='max_length'`.
5. Concatenate token ids / masks back onto the ET columns.
6. Serve batches from `CustomDataset`:
   `input_ids`, `attention_mask`, `labels`, `eye_tracking_features`.

`CustomDataset` stores ET as the raw numpy matrix passed in, **not** as
tensors until `__getitem__`. For the ZuCo CV loop the ET matrix is
sliced with the same `train_index` / `test_index` as the tokenized
frame, so rows stay aligned.

## Scripts that are *not* on the training path

| Script | Why it exists |
|---|---|
| `ZuCo_SST_data/spilt.py` | optional 320/40/40 split; unused by `model_ZuCo_SST.py` |
| `gaze_prediction/data/convert_zuco_data.py` | schema convert + 0–100 scale for a gaze predictor |
| `examples/*.py` | documentation-only; they do not write training CSVs |
