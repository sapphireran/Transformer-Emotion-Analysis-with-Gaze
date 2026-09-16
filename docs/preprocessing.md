# Preprocessing

This page follows a sentence from a ZuCo `.mat` file to the tensors that `EyeTrackingModel.forward` sees. Paths assume you are at the repo root.

## 1. Matlab export

`utils_ZuCo.get_matfiles(task)` builds:

```text
{cwd}\\ZuCo_mat_data\\{task}\\*.mat
```

The backslashes are a Windows leftover. On Linux you will want to change `subdir` to `/ZuCo_mat_data/` (or pass a real path) before re-running `read_ZuCo_mat.py`. The function asserts that each task folder contains **exactly 12** files, one per subject.

`DataTransformer.__call__(subject)` loads:

```python
io.loadmat(files[subject], squeeze_me=True, struct_as_record=False)['sentenceData']
```

Each `sent` has `.word` (list of word objects), `.omissionRate`, and per-word attributes `nFixations`, `meanPupilSize`, `GD`, `TRT`, `FFD`, `SFD`, `GPT`.

## 2. Known-bad trial drops

The transformer skips trial index ranges that are empty or corrupt in the public ZuCo dump. Copied from `utils_ZuCo.py`:

| Task | Subject (0-based) | Dropped `i` |
| --- | --- | --- |
| task1 | 2 | `150–249` and `399` |
| task2 | 6 | `i <= 49` |
| task2 | 11 | `50–99` |
| task3 | 3 | `178–224` |
| task3 | 7 | `i >= 359` |
| task3 | 11 | `270–313` and `362–406` |

Those ranges also change the preallocated feature matrix height. If you add a new subject, re-read this table before trusting row counts.

`split_data()` in the same file bisects Task 1 tables to "control for order effects." Nothing in the current training scripts calls it.

## 3. Word cleaning

```python
token = re.sub('[^\w\s]', '', word.content)
token = token.lower() if j == 0 else token
```

Only the first word of the sentence is lowercased. Later capitalized names stay capitalized. Punctuation-only tokens become empty strings and later `unknown`.

## 4. Sentence aggregation

See [gaze-features.md](gaze-features.md) for the running-sum / `nwords_fixated` logic. After the sentence loop:

1. `check_inf` drops any row that contains `+inf` or `-inf`.
2. Feature scaling (below) runs on the remaining matrix.
3. NaNs are filled with zeros, column min, or column mean (`fillna`).

`read_ZuCo_mat.py` uses `scaling='raw'` and `fillna='zeros'`, so the per-subject CSVs in `ZuCo_et_csv_data/` are unscaled means with zeros for missing values.

## 5. Feature scaling options

`DataTransformer` implements four modes on **columns** (features), not rows:

| Mode | Formula | Used for |
| --- | --- | --- |
| `raw` | identity | per-subject export |
| `min-max` | `(x - min) / (max - min)` | `combined_sst_et_min_max.csv` |
| `mean-norm` | `(x - mean) / (max - min)` | available, unused in checked-in tables |
| `standard` | `(x - mean) / std` | `combined_sst_et_standard.csv` (training default) |

`get_average_sentence_level.py` does **not** call `DataTransformer` scaling. It:

1. Reads the 12 raw CSVs.
2. Replaces `0` with `NaN` on every non-id column.
3. Means by row index.
4. Fits `sklearn.preprocessing.MinMaxScaler` and `StandardScaler` on the reader-mean table.

Those two sklearn scalers are column-wise as well. They are the scalers that match the files `model_ZuCo_SST.py` actually trains on (via the combined table, which was joined from the standard-scaled average).

### Why standard scaling is the training default

Linear fusion (`nn.Linear(5, 16)`) is sensitive to channel scale. Raw TRT is hundreds of milliseconds; nFixations is ~1–4. Without z-scoring, the first gradient steps are TRT-dominated. Min-max is better than raw but still compresses GPT's tail.

## 6. Joining sentiment

`ssts_ZuCo.csv` is an inner-join partner on `sentence_id`. The combined files keep:

```
sentence_id, sentence, sentiment_label,
omissionRate, nFixations, meanPupilSize, GD, TRT, FFD, SFD, GPT
```

`SentLen` is dropped at this join (it remains on the ET-only averages). If you need a length control, re-join or count tokens yourself. The example library counts whitespace tokens as a portable length proxy.

## 7. Full SST path

Different pipeline, same five names:

1. `stts_all_sentence_level.csv` — raw SST strings + string labels.
2. `convert_sst_to_et.py` — NLTK tokenize, alphabetic filter, write zeros for all ET columns.
3. A gaze predictor (not in this repo) fills `nFix, FFD, GPT, TRT, GD` at word level (`prediction_test_v2.csv`).
4. Word rows are aggregated to sentence rows → `combined_full_sst_et.csv`.
5. `SST_data/spilt.py` writes the 80/10/10 CSVs.

`model_full_SST.py` tokenizes `sentence` with BERT or RoBERTa and reads the five ET columns as `float32`.

## 8. Tokenization for the transformer

```python
tokenizer(examples['sentence'], padding='max_length', truncation=True, max_length=128)
```

- Reviews in ZuCo are short; 128 is plenty.
- Full SST has some long sentences that will truncate. Truncation is not logged.
- `padding='max_length'` makes every batch a dense 128, which is why `model_full_SST.py` can use `batch_size=256` without a dynamic pad collator.

The Hugging Face `Dataset.map` result is converted back to pandas so it can be concatenated with the ET columns. `CustomDataset` then stacks `input_ids`, `attention_mask`, `labels`, and `eye_tracking_features`.

## 9. Tensors the model sees

For each item:

| Key | Dtype | Shape |
| --- | --- | --- |
| `input_ids` | int64 | `(128,)` |
| `attention_mask` | int64 | `(128,)` |
| `labels` | int64 | `()`  (0/1/2) |
| `eye_tracking_features` | float32 | `(5,)` |

Batching adds a leading `B` dimension. Text-only model types ignore the ET tensor.

## 10. Things this pipeline does not do

- No stop-word removal, stemming, or truecasing beyond the first-word lowercasing.
- No subject-mean centering (subtracting each reader's global mean before averaging). Pupil size especially would want that.
- No leakage-safe scaler: when `model_ZuCo_SST.py` uses already-z-scored columns, the mean/std were fit on **all 400 rows**, including the fold that will be the test fold. For a 400-row set the leak is small; it is still a leak. A stricter protocol would fit `StandardScaler` inside each fold.
- No class weights. Neutral is not guaranteed to be 1/3 of each split.

`examples/scripts/04_split_sanity_check.py` reports class drift on the convenience splits. It does not rewrite them.
