# Scripts as found (January 2024)

A file-by-file reading of the original Python. Paths, typos, and dead
arguments are left as they are. The atlas examples do not rewrite these
files.

## `utils_ZuCo.py`

- `get_matfiles` joins `os.getcwd() + '\\ZuCo_mat_data\\' + task` — Windows
  backslashes. On this Linux checkout the `.mat` files are not present
  anyway; only the derived CSVs are.
- `DataTransformer` supports sentence/word, scalings
  `{min-max, mean-norm, standard, raw}`, fillna `{zeros, mean, min}`.
- Hard-coded skip ranges for several (task, subject) pairs, including the
  Task-1 subject-2 hole. Word-level `n_words` accounting matches those
  skips so the DataFrame length is consistent.
- Sentence features: per-word fields summed, then divided by
  `nwords_fixated`. A word whose extracted vector is all zeros does not
  count as fixated.
- `check_inf` deletes rows containing ±inf, mutating the row alignment
  further.
- `split_data` is unused by the trainers.

## `read_ZuCo_mat.py`

Runs `DataTransformer('task1', level='sentence', scaling='raw', fillna='zeros')`
for subjects 0–11, writes `et_csv_data/{1..12}_SR.csv`. The committed
directory is `ZuCo_et_csv_data/`. Re-running the script as-is will not
overwrite the files you think it will.

## `get_average_sentence_level.py`

Looks in `et_csv_data`, replaces 0 with NaN on every column except the first,
`concat` + `groupby(level=0).mean()`, then MinMax and Standard scalers.
Writes `min_max_scaled_average_data.csv` and `standard_scaled_average_data.csv`.
See [subject-3-reindex.md](subject-3-reindex.md) for why `level=0` is the
wrong key after packed ids.

## `ZuCo_et_csv_data/word/get_average.py`

Same positional mean at word level. The zero→NaN line is commented out, so
skips (exact zeros) pull the mean down. Null `Word` → `unknown`. Output:
`word_averages_v2.csv`.

## `convert_full_SST.py` and `ZuCo_SST_data/save_SST_data.py`

Walk `NEGATIVE/POSITIVE/NEUTRAL` folders of `.txt` files, map labels, sort
by `sentence_id`. `save_SST_data.py` writes `output.csv` with string ids;
`convert_full_SST.py` writes `ZuCo_SST_data/ssts_ZuCo.csv` with integer ids.
The `all/` folders are not in this checkout.

## `SST_data/convert_sst_to_et.py`

NLTK `word_tokenize`, keep `[A-Za-z]+`, write word rows with gaze zeros.
Needs `nltk.download('punkt')`. Input/output paths are filenames, so the
working directory must be `SST_data/`.

## `SST_data/spilt.py` and `ZuCo_SST_data/spilt.py`

Identical 80/10/10 recipe, different input filenames, relative paths
assuming you `cd` into the folder.

## `gaze_prediction/data/convert_zuco_data.py`

Min-max to 0–100. nFixations scaled alone; FFD/GPT/TRT/GD share one
min/max. `Sent_ID.split('_')[0]` becomes `sentence_id`. Input path
`training_data/word_averages_v2.csv` is not the committed location.

## `model_ZuCo_SST.py`

- Loads `combined_sst_et_standard.csv`
- Tokenizes with Hugging Face `datasets.Dataset.map`
- `CustomDataset` also receives `eye_tracking_features.iloc[train_index].values`
  — the tokenized frame already concatenated the gaze columns, but
  `__getitem__` uses the separate array. Row alignment assumes
  `Dataset.from_pandas` preserves order (it does if you do not shuffle there).
- K-fold as described in [splits.md](splits.md)
- Gaze branch: `CrossEntropyLoss()(logits.view(-1, 3), labels.view(-1))`
  with `num_labels = 3` hard-coded in the view
- No checkpointing; last epoch of each fold is evaluated
- `hidden_layer_size = 16` is a global, closed over by `EyeTrackingModel`

## `model_full_SST.py`

Same fusion class. Differences that matter:

- Separate train/valid/test files
- `num_epochs = 5`, `batch_size = 256`
- Saves `models/best_{model_type}_model.pth` on best **accuracy**
  (comment: “最佳F1分数”)
- Reloads that checkpoint for test
- Test loop **assigns** `all_preds` instead of extending —
  [known-issues.md](known-issues.md)
- `torch.save` / `torch.load` without `weights_only` (fine on 2024 torch;
  noisy on newer)

Neither trainer creates `models/`. First save raises if the folder is
missing.

## Defaults

Both files default `model_type = 'roberta_eye_tracking'`. Switching to
`'bert'` / `'roberta'` still builds `CustomDataset` with gaze tensors that
the forward pass then ignores. Wasteful, not incorrect.
