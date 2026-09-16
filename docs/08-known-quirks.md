# Known quirks

Things that will waste an evening if you rediscover them. None of
these are theoretical — they are in the current Python.

## Paths and working directory

- Always run scripts from the **repository root**. Training files use
  `'SST_data/...'` and `'ZuCo_SST_data/...'`.
- `utils_ZuCo.get_matfiles` concatenates `os.getcwd()` with
  `'\\ZuCo_mat_data\\'`. On Linux that is one oddly named directory,
  not `ZuCo_mat_data/task1`.
- `read_ZuCo_mat.py` writes `et_csv_data/`; the repo has
  `ZuCo_et_csv_data/`.
- `get_average_sentence_level.py` reads `et_csv_data`.
- `gaze_prediction/data/convert_zuco_data.py` reads
  `training_data/word_averages_v2.csv` (missing).
- `SST_data/convert_sst_to_et.py` and both `spilt.py` files use
  paths relative to **their own folder**, not the repo root. `cd`
  into `SST_data/` or `ZuCo_SST_data/` before running those.

## Filenames

- `spilt.py` is a typo of `split.py` (two copies).
- `stts_all_sentence_level.csv` is a typo of SST. **No header row.**
- `get_average_sentence_level.py` does not write `average_data.csv`
  in the snippet you have — it only writes the two scaled files —
  but `ZuCo_et_csv_data/average_data.csv` exists. An older version
  of the script probably dumped the pre-scale mean; do not assume
  re-running the current file refreshes it.

## Training-script landmines

1. **Test metrics on full SST use the last batch only**
   (`all_preds = ...` instead of `.extend`). Validation is correct.
2. **Best-model comment says F1, code uses accuracy.**
3. **`batch_size` in `model_ZuCo_SST.py` is unused**; loaders use 16.
4. **`models/` is not created** before `torch.save`.
5. **No seed** except KFold’s `random_state`.
6. **Unweighted cross-entropy** on a 19% neutral class (full SST).
7. Fusion `CrossEntropyLoss` hard-codes `view(-1, 3)` instead of
   `num_labels`. Harmless today, wrong if you ever go binary.
8. `EyeTrackingModel` picks the pretrained name with
   `base_model == BertModel`. That works for the two classes passed
   in today. A future `from_pretrained` wrapper will break the
   ternary.

## Data-quality landmines

- Subject `3_SR.csv` has 299 sentences. Averages by row index are
  not a clean 12-reader mean for every id.
- ZuCo `train/valid/test` CSVs are **not** what the CV script uses,
  and they were split **without** stratification (valid is 19/14/7
  pos/neu/neg — lopsided).
- Full SST split also has no `stratify=`.
- Word-average v2 **includes zeros** in the mean (the NaN line is
  commented out).
- `convert_sst_to_et.py` writes all-zero gaze; do not train on
  `sst_et_test.csv` thinking those zeros are z-scores.
- `provo.csv`’s last numeric column is `fixProp`, not `GD`.
- `stts_all_sentence_level.csv` quoted sentences contain `\\/` and
  `-LRB-` / `-RRB-` PTB leftovers. The tokenizer still eats them;
  a length counter that splits on whitespace will count those
  tokens as words.

## Library leftovers

`model_full_SST.py` and `model_ZuCo_SST.py` import
`from datasets import Dataset` only to tokenize via
`Dataset.map`. They also import HF classification models even when
you only use the fusion class. `utils_ZuCo.py` imports `gzip`,
`math`, and `scipy` without using them.

`CustomDataset` takes a `dataframe` that already contains gaze
columns **and** a separate `eye_tracking_features` array. The extra
columns in the frame are unused at `__getitem__` time.

## Tokenizer mismatch with gaze words

ZuCo word tables use a light regex (`[^\w\s]` stripped, first word
lowercased). SST transfer uses NLTK `word_tokenize` then
`[A-Za-z]+`. BERT/RoBERTa use BPE / WordPiece. There is **no**
alignment between “word_id 3 in ZuCo” and “token id 3 in RoBERTa.”
That is fine for sentence-level fusion and fatal if you naively
add a `(len(words), 5)` tensor to hidden states.

## Chinese comments, English identifiers

Several scripts have Chinese comments from the original personal
workflow (`配置参数`, `初始化分词器`, …). Behavior is in the
English identifiers. The docs in `docs/` are English so they match
the GitHub repo language; they do not translate those comments
line-by-line.

## Example scripts vs training scripts

The `examples/` folder will not download weights, will not write
`models/`, and will not fix the test-loop bug. If an example and a
training script disagree about a column name, the training script
is what GPU runs will use; open an issue in your notes and align
them.

## “I changed one CSV cell”

Re-run `python3 examples/02_schema_check.py`. It checks headers,
row counts, label sets `{0,1,2}`, and that the five fusion columns
are parseable as floats. It does not checksum values, so a silent
rescaling will still pass.
