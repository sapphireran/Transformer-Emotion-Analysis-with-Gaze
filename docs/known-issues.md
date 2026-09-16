# Known issues

Personal punch list for this checkout. None of these are company-tracker
tickets; they are things the docs/examples had to work around.

## 1. Full-SST test metrics use only the last batch

In `model_full_SST.py` the test loop does:

```python
all_preds = preds.cpu().numpy()
all_labels = labels.cpu().numpy()
```

The validation loop correctly `extend`s. The test loop **replaces**. With
`batch_size = 256` and 1,186 test rows, the printed Test Acc / P / R / F1 are
computed on the last 162 examples, not the full split.

Fix when you next touch that file: `extend` like the validation loop, and
create `models/` before `torch.save`.

## 2. “Best F1” is actually best accuracy

```python
best_val_acc = 0.0  # comment says F1
...
if val_acc > best_val_acc:
    ...
    print(f"... with F1: {best_val_acc:.4f}")
```

The checkpoint is selected on accuracy. The log string lies.

## 3. Windows path in `get_matfiles`

```python
def get_matfiles(task: str, subdir='\\ZuCo_mat_data\\'):
    path = os.getcwd() + subdir + task
```

On Linux this becomes `.../workspace\\ZuCo_mat_data\\task1`. Pass an explicit
POSIX `subdir` or change the default before running `read_ZuCo_mat.py`.

## 4. Dump directory names drifted

| Script writes / reads | Files in git |
| --- | --- |
| `read_ZuCo_mat.py` → `et_csv_data/` | `ZuCo_et_csv_data/` |
| `get_average_sentence_level.py` ← `et_csv_data/` | `ZuCo_et_csv_data/` |
| `convert_zuco_data.py` ← `training_data/word_averages_v2.csv` | `ZuCo_et_csv_data/word/word_averages_v2.csv` |

The checked-in CSVs are the outputs. The scripts will not find them without
path edits.

## 5. `spilt.py` and unstratified cuts

Both split helpers are named `spilt.py`. Full SST is large enough that the
class mix stayed reasonable. ZuCo `valid.csv` has **7** negatives out of 40.
`model_ZuCo_SST.py` ignores those files and uses stratified CV instead — do
not switch to the hold-out without stratifying.

## 6. Headerless SST source file

`SST_data/stts_all_sentence_level.csv` has no header. `inspect_datasets.py`
treats the first row as a header if you use `DictReader` blindly. The example
script special-cases this path.

## 7. Subject 3 is short

`ZuCo_et_csv_data/3_SR.csv` has 299 sentences (ids 0–298). Cross-subject
averages for ids 299–399 are 11-subject means. `DataTransformer` documents
the dropped ranges; the CSV is consistent with that, not corrupt.

## 8. Test-set overwrite vs `CustomDataset` labels

`CustomDataset` stores `dataframe['sentiment_label'].values` as-is (strings
or ints depending on pandas). `CrossEntropyLoss` wants `int64` class indices.
This has worked when pandas infers integers; if you re-save CSVs with quoted
labels, cast explicitly.

## 9. Duplicated model code

`EyeTrackingModel`, `CustomDataset`, `calculate_metrics`, and `get_model`
are copy-pasted across the two training scripts. Behaviour can drift. The
numpy fusion example is the reference for *shapes*; it is not wired into
training.

## 10. `get_average_sentence_level.py` zeros → NaN

Replacing every `0` with NaN before averaging is reasonable for skipped
fixations, but a true zero duration (rare) would also be dropped. Word-level
`get_average.py` has that replace **commented out**.

## 11. No training seeds

Splits are seeded. `torch` is not. Do not expect bit-identical loss curves.

## 12. Zip vs working tree

`ZuCo_SST_data/ZuCo_SST_data.zip` may still contain the polarity folders that
`convert_full_SST.py` expects. The working tree does not. Unzip locally; do
not assume `all/` exists in a clean clone.
