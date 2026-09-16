# Known issues in the historical scripts

The trainers and converters are left as they were. This page is the
personal errata sheet. Example code does **not** copy these bugs.

---

## 1. Full-SST test metrics use only the last batch

**File**: `model_full_SST.py` (test loop, after the best checkpoint is
reloaded)

Validation does this:

```python
all_preds.extend(preds.cpu().numpy())
all_labels.extend(labels.cpu().numpy())
```

The test loop does this:

```python
all_preds = preds.cpu().numpy()
all_labels = labels.cpu().numpy()
```

Assignment replaces the list. After the last batch of 256, at most 256
of 1,186 test rows remain. Printed `Test Acc / P / R / F1` are not
full-set scores.

**Fix when you next train** (do not have to change the historical file
to use the examples):

```python
all_preds.extend(preds.cpu().numpy())
all_labels.extend(labels.cpu().numpy())
```

`examples/lib/metrics.py` always concatenates batches.

---

## 2. “Best F1” comment vs accuracy selection

Same file:

```python
best_val_acc = 0.0  # 初始化最佳F1分数
...
if val_acc > best_val_acc:
    ...
    print(f"... with F1: {best_val_acc:.4f}")
```

The number being saved is validation **accuracy**. On a 19% neutral
class this can disagree with F1. If you want F1 selection, compare
`val_f1` and say so in the print.

---

## 3. Windows path separators in `get_matfiles`

```python
def get_matfiles(task: str, subdir='\\ZuCo_mat_data\\'):
    path = os.getcwd() + subdir + task
```

On Linux this looks for a single oddly named directory, not
`ZuCo_mat_data/task1`. Prefer `os.path.join(os.getcwd(), 'ZuCo_mat_data',
task)`. Irrelevant until someone re-extracts `.mat` files.

---

## 4. Folder names drifted

| Script | Path it opens | Path that exists |
| --- | --- | --- |
| `get_average_sentence_level.py` | `et_csv_data` | `ZuCo_et_csv_data` |
| `read_ZuCo_mat.py` | `et_csv_data` | `ZuCo_et_csv_data` |
| `gaze_prediction/data/convert_zuco_data.py` | `training_data/word_averages_v2.csv` | `ZuCo_et_csv_data/word/word_averages_v2.csv` |
| `ZuCo_SST_data/save_SST_data.py` | `all/...` | not in the clone |

The example loaders only use existing paths.

---

## 5. `spilt.py` filenames and unstratified full-SST split

Both split helpers are named `spilt.py`. Cosmetic.

`SST_data/spilt.py` calls `train_test_split` **without** `stratify=`.
Neutral share and majority prior differ across train/valid/test. The
ZuCo split helper has the same pattern; the CV trainer bypasses those
files.

---

## 6. Headerless SST dump

`SST_data/stts_all_sentence_level.csv` has no header. Pandas will treat
the first review as column names unless you pass `header=None`. The
combined table is the safer source for modelling.

---

## 7. Two skip policies when averaging

Sentence-level `get_average_sentence_level.py` converts zeros to NaN
before the mean. Word-level `word/get_average.py` keeps zeros. A skip
is “missing” in one place and “zero duration” in the other. Neither is
wrong; they are inconsistent. See [gaze-features.md](gaze-features.md).

---

## 8. `fillna` on a MultiIndex-like `columns=[fields]`

`DataTransformer` builds `pd.DataFrame(..., columns=[fields])` — a
**list containing a list** — which gives pandas a MultiIndex-looking
column axis in older versions. `getattr(df, field)` then depends on
pandas treating the name as an attribute. It worked when the CSVs were
written; it is a brittle pattern. The checked-in CSVs already have
flat headers.

---

## 9. In-place `fillna` / `replace` deprecation

Several scripts use `df.fillna(0, inplace=True)` and chained
`replace(...).dropna(axis=0, inplace=True)` (the chained `inplace`
does not modify the object you think it does). Cosmetic until a pandas
upgrade starts warning loudly.

---

## 10. Hard-coded `view(-1, 3)`

```python
loss = CrossEntropyLoss()(logits.view(-1, 3), labels.view(-1))
```

`num_labels` is 3, so this matches. If you ever try binary SST it will
silently reshape wrong. Use `logits.view(-1, num_labels)`.

---

## 11. Batch size constant ignored on ZuCo

`model_ZuCo_SST.py` sets `batch_size = 16` and then constructs loaders
with `batch_size=16` as a literal. Changing the constant does nothing.
Full SST uses the constant correctly.

---

## 12. No `models/` directory creation

`torch.save(..., 'models/best_{model_type}_model.pth')` fails if
`models/` does not exist. The directory is gitignored and not created
by the script.

---

## What the examples do instead

- Load only paths that exist (`examples/lib/paths.py`).
- Treat `stts_all_sentence_level.csv` as headerless.
- Stratify when they split.
- Accumulate every evaluation row.
- Report accuracy, weighted F1, **and** macro F1.
- Refuse to pretend projected gaze is measured gaze in printed titles.
