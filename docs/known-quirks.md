# Known quirks

Things that will surprise you if you treat the original scripts as a
clean package. Documented here rather than silently “fixed”, because
this pass is docs and examples only.

## 1. Full-SST test metrics use the last batch only

In `model_full_SST.py` the test loop does:

```python
preds = torch.argmax(logits, dim=1)
all_preds = preds.cpu().numpy()
all_labels = labels.cpu().numpy()
```

The validation loop correctly `extend`s. The test loop **replaces**.
With `batch_size = 256` and 1,186 test rows that means the printed
`Test Acc / P / R / F1` is computed on the last 166 rows (or 256 if you
change the split), not the full test set.

A minimal patch is to match the validation loop:

```python
all_preds.extend(preds.cpu().numpy())
all_labels.extend(labels.cpu().numpy())
```

and initialize `all_preds, all_labels = [], []` before the loop (already
done).

## 2. Comment / metric mismatch when saving checkpoints

```python
best_val_acc = 0.0  # 初始化最佳F1分数
...
print(f"New best model saved as {best_model_path} with F1: {best_val_acc:.4f}")
```

The variable is accuracy. The comment and the print string say F1.

## 3. Directory names the writers use ≠ directory names in git

| Script | Path it writes / reads | Path of the committed files |
|---|---|---|
| `read_ZuCo_mat.py` | `et_csv_data/{n}_SR.csv` | `ZuCo_et_csv_data/{n}_SR.csv` |
| `get_average_sentence_level.py` | `et_csv_data/` | `ZuCo_et_csv_data/` |
| `utils_ZuCo.get_matfiles` | `\\ZuCo_mat_data\\` (Windows) | not committed |
| `gaze_prediction/data/convert_zuco_data.py` | `training_data/word_averages_v2.csv` | `ZuCo_et_csv_data/word/word_averages_v2.csv` |

Re-running a writer without editing paths will create a second folder
rather than overwrite the committed one.

## 4. `models/` is not created for you

`model_full_SST.py` saves to `models/best_{model_type}_model.pth`.
`torch.save` does not mkdir. An otherwise successful first epoch will
crash if you have not `mkdir -p models`.

## 5. Filename typo: `spilt.py`

Both `SST_data/spilt.py` and `ZuCo_SST_data/spilt.py` are the train /
valid / test splitters. They work; they are just misspelled. The
example scripts refer to them by the real filename.

## 6. ZuCo subject 3 is short on purpose

`3_SR.csv` (subject index 2) has 299 sentence rows and 5,293 word rows.
`DataTransformer.__call__` skips Task 1 indices `150–249` and `399` for
that subject — a known hole in the public ZuCo release, not a truncated
write. `examples/data_integrity.py` asserts this exact length.

## 7. Two different 5-feature orders

```python
# ZuCo
['nFixations', 'FFD', 'GPT', 'TRT', 'GD']

# full SST
['nFix', 'FFD', 'GPT', 'TRT', 'GD']
```

Same names up to `nFix` vs `nFixations`, but the **column order in the
tensor** is `nFix, FFD, GPT, TRT, GD` on SST and
`nFixations, FFD, GPT, TRT, GD` on ZuCo. `TRT` and `GPT` swap places
relative to each other. A 5×16 weight matrix is not portable across the
two scripts.

## 8. ZuCo `train.csv` / `valid.csv` / `test.csv` are unused

`model_ZuCo_SST.py` always opens `combined_sst_et_standard.csv` and
does 5-fold CV. The 320/40/40 split is leftover from an earlier
hold-out setup. Neutral / negative / positive counts in the 40-row
valid and test files are also jumpy (valid has only 7 negatives); that
is why stratified CV is the better default.

## 9. Word-level average aligns by row index

`ZuCo_et_csv_data/word/get_average.py` does
`pd.concat(dataframes).groupby(level=0).mean()`. That assumes every
subject file is the same length. Subject 3 is not. Concat-then-groupby
on the default integer index will still produce 7,129 rows (the longer
files dominate), but subject 3's 5,293 rows only contribute to the
leading indices. Interpret `word_averages_v2.csv` as “mean over the
subjects who have a row at this index”, not “mean over all 12 readers
for this word”.

## 10. `hash()` is not used in the toy fusion demo

Python's built-in `hash()` is randomized per process (`PYTHONHASHSEED`).
`examples/late_fusion_demo.py` hashes tokens with `hashlib.md5` so two
runs on the same CSV produce the same bag-of-words features. If you
copy the demo and switch to `hash(token)`, your accuracy will jitter.

## 11. Original comments are mixed English / Chinese

The training and averaging scripts were written with Chinese inline
comments (`配置参数`, `初始化分词器`, …). The docs and examples in this
pass are English. Behavior described here was read from the code, not
from those comments (see item 2).

## 12. PROVO schema is not the SST/ZuCo schema

`gaze_prediction/data/provo.csv` ends with `fixProp` instead of `GD`.
Any script that blindly selects the last five numeric columns will mix
units. `examples/word_level_preview.py` special-cases this file.
