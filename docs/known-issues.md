# Known issues in the original scripts

These are notes for future-me (and anyone else reading the personal
research code). The example package does **not** silently "fix" the
trainers; it documents them.

## Full-SST test metrics use only the last batch

In `model_full_SST.py` the test loop does:

```python
all_preds = preds.cpu().numpy()
all_labels = labels.cpu().numpy()
```

Validation correctly uses `extend`. Test overwrites the arrays every
batch, so Acc / P / R / F1 are computed on the final 256 (or fewer)
rows. If you quote a test score, change those two lines to `extend`
first.

## Checkpoint comment does not match the code

```python
best_val_acc = 0.0  # 初始化最佳F1分数
...
print(f"New best model saved ... with F1: {best_val_acc:.4f}")
```

The value is validation **accuracy**. The print label is wrong.

## ZuCo `batch_size` config is ignored

The file sets `batch_size = 16` at the top, then builds loaders with
a literal `16`. Changing the config does nothing. The full-SST script
does honor its `batch_size`.

## Path drift after files were renamed

| Script | Path it writes / reads | Checked-in location |
| --- | --- | --- |
| `read_ZuCo_mat.py` | `et_csv_data/{i}_SR.csv` | `ZuCo_et_csv_data/` |
| `get_average_sentence_level.py` | `et_csv_data` | `ZuCo_et_csv_data/` |
| `utils_ZuCo.get_matfiles` | `cwd + '\\ZuCo_mat_data\\' + task` | MATLAB files not in git |
| `convert_full_SST.py` | `ZuCo_SST_data/all/{LABEL}` | folder not in git |
| `ZuCo_SST_data/save_SST_data.py` | `all/{LABEL}` | folder not in git |
| `gaze_prediction/data/convert_zuco_data.py` | `training_data/word_averages_v2.csv` | not in git |

`get_matfiles` also uses Windows backslashes, which do not join cleanly
on Linux. Prefer `os.path.join(os.getcwd(), "ZuCo_mat_data", task)`.

## Subject 3 is not a bug

`3_SR.csv` having 299 sentence rows (and fewer word rows) is the
intended result of the skip ranges in `DataTransformer` for Task 1
subject index 2. Do not "repair" it by padding with zeros if you want
to match ZuCo's released quality filters.

## Averaging is by row index, not by sentence id

`get_average_sentence_level.py` does `concat(...).groupby(level=0).mean()`.
That assumes every subject file is aligned on the same trial index.
Subject 3 is shorter, so the tail of the average is a mean over 11
people. Word-level `get_average.py` has the same property and copies
identity columns from subject 1.

## Word-level zeros are kept

The zero-to-NaN line in `ZuCo_et_csv_data/word/get_average.py` is
commented out. Unfixated words stay 0 and pull the subject mean down.

## Shared duration range in `convert_zuco_data.py`

nFix is min-max scaled on its own min/max. FFD, GPT, TRT, and GD share
**one** min and **one** max across all four duration columns. That makes
the 0–100 values comparable to each other but is not the same as
column-wise sklearn scaling used on the sentence averages.

## Hold-out ZuCo splits are tiny and skewed

`valid.csv` is 40 rows with 19 POS / 7 NEG. Do not treat a single run
on that file as a model ranking. Use the 5-fold script.

## Filename typo

`spilt.py` in both `SST_data/` and `ZuCo_SST_data/` is "split" misspelled.
The scripts work; grep for `spilt` if you look for them.

## No `models/` directory in git

`model_full_SST.py` writes `models/best_{model_type}_model.pth` without
creating the folder. `mkdir -p models` before the first run.

## Unused imports

`utils_ZuCo.py` imports `gzip`, `math`, and `scipy` without using them.
Harmless, but they make the MATLAB helper look heavier than it is.
