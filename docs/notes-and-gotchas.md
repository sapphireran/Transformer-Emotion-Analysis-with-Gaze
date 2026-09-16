# Notes and gotchas

Personal debugging list for this clone. These are issues in the original scripts and tables, not a cleanup pass. The example package works around them instead of silently "fixing" training behavior.

## Path and filename leftovers

| Location | What it says | What exists |
| --- | --- | --- |
| `utils_ZuCo.get_matfiles` | `\\ZuCo_mat_data\\` | no MATLAB tree in the repo; backslashes are wrong on Linux |
| `read_ZuCo_mat.py` | writes `et_csv_data/` | checked-in dir is `ZuCo_et_csv_data/` |
| `get_average_sentence_level.py` | reads `et_csv_data` | same mismatch |
| `ZuCo_SST_data/spilt.py` | `spilt` | typo for `split` |
| `SST_data/spilt.py` | same typo | same |
| `convert_full_SST.py` | reads `ZuCo_SST_data/all/` | folder not checked in |
| `ZuCo_SST_data/save_SST_data.py` | reads `all/`, writes `output.csv` | neither is checked in |

If a script fails with `FileNotFoundError` on one of those names, you are re-running a producer, not a consumer. The consumer CSVs are already present.

## Full-SST test loop overwrites predictions

In `model_full_SST.py` the test loop does:

```python
all_preds = preds.cpu().numpy()
all_labels = labels.cpu().numpy()
```

The train/valid loops correctly `extend`. The test loop keeps only the **last batch**. With `batch_size=256` and 1186 test rows that is the final ~158 examples, not the full test set.

Until that line is changed to `extend`, do not treat the printed test accuracy as a full-set score. Validation numbers from the same script are fine.

## Checkpoint comment vs. metric

```python
best_val_acc = 0.0  # 初始化最佳F1分数
...
if val_acc > best_val_acc:
```

The file is saved on validation **accuracy**. The log line still says "with F1".

## `num_labels` is not fully parameterized

Fusion loss uses `logits.view(-1, 3)` and `CrossEntropyLoss` with three classes. `num_labels` is a variable but several sites ignore it.

## RoBERTa pooler

`EyeTrackingModel` always reads `base_output.pooler_output`. That attribute exists on the stock `RobertaModel` used here. If you swap in a RoBERTa variant without a pooler, use the first token hidden state instead.

## Scaling mix-ups

- ZuCo training reads **standard** (z-score) features.
- Full SST features are whatever was written into `combined_full_sst_et.csv` (not raw ms).
- Subject CSVs in `ZuCo_et_csv_data/{k}_SR.csv` are closer to raw means (pupil sizes in the 800–1000 range).

Feeding a raw pupil column into a model trained on z-scores will dominate the linear gaze layer.

## Tiny ZuCo valid/test files

`valid.csv` and `test.csv` are 40 rows and not stratified. The valid label counts are 7 / 14 / 19. A 5-point accuracy change there is one or two sentences. Use `model_ZuCo_SST.py`'s folds, or the demo's own stratified split.

## Word-level average alignment

`get_average.py` averages numeric columns by **row index** after `concat` + `groupby(level=0)`, then pastes `Word` / `Sent_ID` from subject 1. That is correct only if every subject file has the same words in the same order. If a subject skipped a sentence, later rows silently mis-align. The checked-in `word_averages_v2.csv` should be treated as "aligned to subject 1's token sequence."

## Hash vs. transformer in the examples

`examples/fusion_forward_demo.py` uses a hashed bag-of-words vector. It answers "does fusion arithmetic run, and do these columns have signal?" It does **not** answer "does RoBERTa + gaze beat RoBERTa." Do not paste those accuracies next to GPU runs without labeling them as the hash baseline.

## Neutral class on full SST

About 19% of full SST rows are neutral (`1`). Weighted F1 will look healthier than macro F1. Report both if you compare models.

## Chinese comments, English filenames

Several originals have Chinese comments and English identifiers. The docs and examples are English. Behavior described here was read from the code, not from those comments, and in at least one place (F1 vs. accuracy) the comment is wrong.

## Personal-only scope

This tree is a personal research repo (`sapphireran/Transformer-Emotion-Analysis-with-Gaze`). Do not copy company data, internal tokenizers, or private review text into it.
