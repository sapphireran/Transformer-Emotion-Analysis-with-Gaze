# Known issues in the original scripts

I am leaving the historical trainers alone in this docs/examples work so old
runs stay comparable. These are the traps I keep hitting.

## `model_full_SST.py` test metrics use the last batch only

The train and valid loops `extend` prediction lists. The test loop assigns:

```python
all_preds = preds.cpu().numpy()
all_labels = labels.cpu().numpy()
```

That overwrites every batch. With `batch_size=256` and 1,186 test rows the
printed test score is ~186 examples, not the test set.

## Best checkpoint is selected by accuracy, commented as F1

```python
best_val_acc = 0.0  # 初始化最佳F1分数
...
if val_acc > best_val_acc:
```

On a 3-class problem with a 42% majority, F1 and accuracy can disagree.
The filename still says `best_{model_type}_model.pth`.

## `models/` is not created

`torch.save(..., 'models/best_roberta_eye_tracking_model.pth')` needs the
directory. The repo does not ship it. `mkdir -p models` first.

## Path typos and leftover Windows paths

| Location | What it says | What exists |
| --- | --- | --- |
| `utils_ZuCo.get_matfiles` | `\\ZuCo_mat_data\\` | not in git; Linux needs `/` |
| `read_ZuCo_mat.py` / `get_average_sentence_level.py` | `et_csv_data` | `ZuCo_et_csv_data` |
| `ZuCo_SST_data/spilt.py`, `SST_data/spilt.py` | filename `spilt` | means `split` |
| `convert_full_SST.py` | `ZuCo_SST_data/all` | folder not in git |

## Subject-aware skips are hard-coded

`DataTransformer.__call__` special-cases task/subject/index ranges. Those
magic numbers match known ZuCo holes, but they make “average across 12
readers” silently become “average across whoever has this row.” Subject 3
is the visible case (299 vs 400 sentence rows).

## Word averages group by row index

`ZuCo_et_csv_data/word/get_average.py` does `concat(...).groupby(level=0).mean()`.
That is not `groupby(['Sent_ID','Word_ID'])`. Misaligned subject files
corrupt the tail of `word_averages_v2.csv`.

## Two scrapers, two outputs

`convert_full_SST.py` writes `ssts_ZuCo.csv` with integer `sentence_id`.
`ZuCo_SST_data/save_SST_data.py` writes `output.csv` and keeps the filename
stem as a string. Same label map, different destination.

## Eye-tracking unused unless `model_type` ends with `_eye_tracking`

Easy to train `roberta` by accident, see a decent F1, and think gaze helped.
Print `Model type:` is already in the script — believe that line.

## No seeding of PyTorch / CUDA

KFold and `train_test_split` use 42. Dropout and GPU kernels do not. Fusion
runs are not deterministic.

## RoBERTa + `pooler_output`

See [architecture.md](architecture.md). This is a choice, but it means the
`roberta` HF baseline and the `roberta_eye_tracking` custom model are not
the same text head plus a sidecar. They are different text heads.

## `get_average_sentence_level.py` replaces 0 with NaN

Legitimate zero fixations become missing, then the mean skips them. That may
be what you want for “ignored skips,” but it is not documented in the script.

## Example / helper package is new

`tea_gaze` and `examples/` are documentation tooling. They can be wrong
without changing a trainer. If a helper disagrees with a CSV, believe the
CSV and file a fix in the helper.
