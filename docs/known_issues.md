# Known issues in the original scripts

Personal punch list. The `examples/` suite does not depend on any of
these files being fixed. If you train, read this first.

## 1. Full-SST test metrics use only the last batch

In `model_full_SST.py` the test loop does:

```python
preds = torch.argmax(logits, dim=1)
all_preds = preds.cpu().numpy()
all_labels = labels.cpu().numpy()
```

Validation correctly `extend`s. Test **replaces**. With `batch_size=256`
and 1,186 test rows that is four full batches plus a remainder of 162;
the printed `Test Acc / P / R / F1` is computed on those last 162
labels only.

Fix when you next touch the script: use `extend` (and a list) the same
way the validation loop does. Until then, quote validation numbers or
recompute from a saved checkpoint with a patched loop.

## 2. "Best F1" is actually best accuracy

```python
best_val_acc = 0.0  # 初始化最佳F1分数
...
if val_acc > best_val_acc:
    ...
    print(f"... with F1: {best_val_acc:.4f}")
```

The checkpoint is selected on accuracy. The print labels it F1. On
full SST, accuracy and weighted F1 are close but not identical because
neutral is underrepresented.

## 3. `spilt.py` is a typo, twice

`SST_data/spilt.py` and `ZuCo_SST_data/spilt.py` both mean "split".
They work. Search for `split` will miss them.

## 4. Windows paths in `get_matfiles`

```python
def get_matfiles(task: str, subdir='\\ZuCo_mat_data\\'):
    path = os.getcwd() + subdir + task
```

On Linux this looks for `.../workspace\ZuCo_mat_data\task1`. Pass
`subdir='/ZuCo_mat_data/'` or rewrite with `os.path.join`. The function
also `assert`s exactly 12 files.

## 5. Folder name drift: `et_csv_data` vs `ZuCo_et_csv_data`

`read_ZuCo_mat.py` and `get_average_sentence_level.py` write/read
`et_csv_data`. The checked-in directory is `ZuCo_et_csv_data`. Either
symlink or edit the path before regenerating averages.

## 6. Hard-coded class count in the fusion loss

```python
loss = CrossEntropyLoss()(logits.view(-1, 3), labels.view(-1))
```

`num_labels` is ignored here. Fine while the task is 3-way; broken the
moment you try binary SST.

## 7. `batch_size` is ignored inside ZuCo CV

The header sets `batch_size = 16`, but the fold loop builds
`DataLoader(..., batch_size=16)` with a literal. Changing the header
does nothing.

## 8. ZuCo holdout CSVs are unused

`ZuCo_SST_data/train.csv` (320), `valid.csv` (40), `test.csv` (40) look
like the official split. `model_ZuCo_SST.py` reloads
`combined_sst_et_standard.csv` and runs 5-fold CV. Do not report those
40-row files as the paper's test set unless you change the script.

## 9. No `models/` directory

`torch.save(..., 'models/best_...pth')` assumes the folder exists.

## 10. Test / train scripts import heavy stacks at module level

`from transformers import ...` and `from_pretrained` (in
`get_model` / tokenizer init) run as soon as you `import` or execute
the file. That is why `examples/` never imports them.

## 11. `stts_all_sentence_level.csv` has no header

The first sentence becomes the column name if you `pd.read_csv`
without `header=None`. `examples/lib/loading.py` handles this.

## 12. `fillna` / `inplace` on possibly-chained ZuCo frames

`DataTransformer` uses `df.iloc[:, :].fillna(0, inplace=True)` and
`getattr(df, field)` on MultiIndex columns (`columns=[fields]`). This
worked in the pandas version that produced the CSVs. Newer pandas
may warn or refuse `inplace` on a copy. Prefer the checked-in CSVs
over a casual re-run.

## 13. Predicted gaze is easy to over-claim

`SST_data/combined_full_sst_et.csv` has the same column names a
measured corpus would use. The values are standardized model
outputs (or a transfer), not twelve ZuCo subjects reading 11k
sentences. Keep that sentence in any note you write.

## 14. Word scaffold drops non-letters

`convert_sst_to_et.py` keeps `^[A-Za-z]+$`. Contractions, numbers, and
SST's `-LRB-` tokens disappear. Predicted gaze therefore does not
align 1:1 with a BERT WordPiece sequence. The sentence-level fusion
sidesteps this; a future token-level fusion would have to realign.
