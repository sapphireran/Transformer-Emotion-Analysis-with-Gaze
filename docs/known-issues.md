# Known issues (as committed)

Personal lab list. None of these are patched in this atlas branch; the
examples *measure* them so a later training rerun does not trust the wrong
number.

## 1. Test metrics are the last batch (`model_full_SST.py`)

Validation:

```python
all_preds.extend(preds.cpu().numpy())
all_labels.extend(labels.cpu().numpy())
```

Test:

```python
all_preds = preds.cpu().numpy()
all_labels = labels.cpu().numpy()
```

Committed test n = 1186, `batch_size = 256` → 5 batches, last size **162**.
The printed `Test Acc` covers 13.7% of the test set, and that slice does not
have the same label prior as the full split (`examples/06_last_batch_metric.py`).

`model_ZuCo_SST.py` uses `extend` in its eval loop. The bug is not
universal; it is the full-SST test block.

## 2. Checkpoint comment vs condition

```python
best_val_acc = 0.0  # 初始化最佳F1分数
...
if val_acc > best_val_acc:
    ...
    print(f"... with F1: {best_val_acc:.4f}")
```

The number is accuracy. Weighted F1 is computed and printed, then ignored
for model selection.

## 3. Subject-3 packed ids in the averages

Documented in [subject-3-reindex.md](subject-3-reindex.md). Positional
`groupby(level=0).mean()` mixes original sentence 250 with original
sentence 150 (etc.) for every reader-average row ≥ 150. Word-level
averages break at row 2594.

## 4. Column name fork

`nFix` vs `nFixations`. Full SST vs ZuCo. The two trainers cannot share a
`load_dataset` helper as written.

## 5. `models/` is not created

`torch.save(..., 'models/best_roberta_eye_tracking_model.pth')` needs the
directory. Not in `.gitignore` originally; this branch ignores `*.pth` so
accidental weights stay local.

## 6. Windows path separators for MATLAB inputs

`get_matfiles(..., subdir='\\ZuCo_mat_data\\')`. Dead on this Linux
checkout; the `.mat` files are also absent.

## 7. `read_ZuCo_mat.py` writes `et_csv_data/`, averages read `et_csv_data/`

Committed products live in `ZuCo_et_csv_data/`. Scripts and artifacts
disagree on the folder name.

## 8. ZuCo 80/10/10 CSVs are unused and the valid split is tiny

`model_ZuCo_SST.py` k-folds the combined table. `valid.csv` has 7 negatives
in 40 rows. Easy to cite the wrong protocol.

## 9. Duplicate review strings in full SST

11 extra duplicate-text rows in train; 1 string shared with valid; 1 with
test. Ids are unique. Harmless for a bag-of-rows trainer, not harmless if
you later group by text.

## 10. `CustomDataset` + `hf.map` alignment

Gaze is concatenated onto the tokenized pandas frame *and* passed as a
parallel numpy array into `CustomDataset`. It works if row order is stable.
A `reset_index` or a non-batched map that drops rows would desynchronize
labels, tokens, and gaze silently.

## 11. Hard-coded `view(-1, 3)`

Loss uses `logits.view(-1, 3)` even though `num_labels` is a variable.
Changing `num_labels` without changing the view would fail or worse.

## 12. No seed

Trainers never call `torch.manual_seed` / `numpy.random.seed`. K-fold
`random_state=42` is the only pinned shuffle. GPU reruns will not match.

## 13. Weighted metrics on a 3-class problem with a small ZuCo set

`precision_score(..., average='weighted')` can look healthy while the
7-negative leftover split is unusable. Prefer per-class scores if you
revive `valid.csv`.

## 14. Predictor placeholder vs training tables

`sst_et_test.csv` is all zeros at word level. `*_full_sst.csv` is z-scored
sentence level. `prediction_test.csv` is raw-scale word level for
sentences 300–399 only. Mixing any two without a join key is how you get
plots that cannot match the trainer inputs.
