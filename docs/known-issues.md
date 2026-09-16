# Known issues

Personal bug list for this checkout. None of these are "company tickets"; they are notes so a future run does not surprise you.

## Training

### Test metrics on full SST only use the last batch

In `model_full_SST.py` the test loop does:

```python
all_preds = preds.cpu().numpy()
all_labels = labels.cpu().numpy()
```

The validation loop correctly uses `.extend(...)`. The test assignment **replaces** the accumulator, so printed Test Acc/P/R/F1 describe one batch of 256 (or fewer) sentences, not the 1,186-row test set.

Fix when you next touch that file: use `extend` and keep the arrays 1-d. Until then, do not quote the test line in a write-up.

### Comment / metric mismatch on checkpointing

`best_val_acc` is described as "best F1" in comments and in the save printout (`with F1: {best_val_acc:.4f}`) but the predicate is `if val_acc > best_val_acc`. The saved file is the best **accuracy** model.

### `batch_size` constant ignored on ZuCo

`model_ZuCo_SST.py` sets `batch_size = 16` then builds loaders with a literal `16`. Harmless today; easy to desync later.

### Training files execute on import

`train_dataset = load_dataset(...)` and the epoch loops live at module top level. `import model_full_SST` would start a run. Examples therefore re-implement a tiny fusion head instead of importing the training modules.

### No `models/` directory

`torch.save(model.state_dict(), 'models/best_...pth')` assumes the folder exists. `.gitignore` ignores it. Create it before the SST script's first save.

### Unseeded torch RNG

Fold indices are seeded; Adam and dropout are not. Expect run-to-run jitter.

## Paths

### `et_csv_data` vs `ZuCo_et_csv_data`

| Script | Path it uses | Path that exists in git |
| --- | --- | --- |
| `read_ZuCo_mat.py` | `et_csv_data/{n}_SR.csv` | `ZuCo_et_csv_data/` |
| `get_average_sentence_level.py` | `et_csv_data` | `ZuCo_et_csv_data/` |
| `utils_ZuCo.get_matfiles` | `\\ZuCo_mat_data\\` (Windows) | nothing checked in |

`get_matfiles` also concatenates `os.getcwd()` with a backslash path, which will not list files on Linux until you change it to `os.path.join`.

### Sentiment source folders are missing

`convert_full_SST.py` reads `ZuCo_SST_data/all/{NEGATIVE,POSITIVE,NEUTRAL}/*.txt`. `save_SST_data.py` reads `all/...`. Those directories are not in the repo. The derived `ssts_ZuCo.csv` is.

### Typo: `spilt.py`

Both split helpers are named `spilt.py` (missing "s"). Harmless but easy to miss when searching for "split".

## Data

### Subject 3 is incomplete

`ZuCo_et_csv_data/3_SR.csv` has 299 sentence rows; word-level `word/3_SR.csv` has 5,293 rows. This matches `DataTransformer` skipping task-1 subject 2 sentences 150–249 and 399. Any average that groups by **row number** rather than `id` / `Sent_ID` will mis-align after those skips.

`word/get_average.py` groups by row index. `examples/aggregate_subjects.py` groups sentence-level files by `id` and reports how many subjects contributed to each sentence.

### `stts_all_sentence_level.csv` has no header and 11,852 rows

`combined_full_sst_et.csv` has 11,853 rows. Off-by-one if you zip the two files naively.

### Placeholder vs predicted word gaze

`SST_data/sst_et_test.csv` and `gaze_prediction/data/prediction_test_v2.csv` share a schema (191,971 rows) but the former is all zeros. Do not train on the placeholder.

### Predicted units ≠ ZuCo units

See [eye-tracking-features.md](eye-tracking-features.md). Mixing them in one DataLoader without rescaling is a silent feature-space shift.

### Word-average `Word` taken from subject 1 only

`get_average.py` copies tokens from `1_SR.csv`. If another subject has a different tokenization at the same row index, you will not see it.

## Dependencies not pinned in the original tree

The first commit had no `requirements.txt`. Versions in the new requirements files are lower bounds, not a lockfile. Training numbers can move with `transformers` releases even when seeds are fixed.

## Plots

The `result/*scatter_hist_plots.png` titles are concatenated English ("TrainDataScatterandHistogramPlotot"). They are still the right pictures: pairwise predicted-gaze diagnostics.
