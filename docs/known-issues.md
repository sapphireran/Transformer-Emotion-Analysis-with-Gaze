# Known issues

These are observations about the committed scripts and CSVs, written down so future personal experiments do not rediscover them. This page does not change training behavior.

## 1. Full-SST test metrics use only the last batch

In `model_full_SST.py` the validation loop does `all_preds.extend(...)`. The test loop overwrites the list:

```python
preds = torch.argmax(logits, dim=1)
all_preds = preds.cpu().numpy()
all_labels = labels.cpu().numpy()
```

With `batch_size = 256` and 1,186 test rows that is the final incomplete batch (~162 examples), not the test set. Accuracy / P / R / F1 printed after “Testing” are not test-set scores.

Fix when you next touch that file: `extend` as in the validation loop, then convert once.

## 2. Best-checkpoint comment does not match the code

```python
best_val_acc = 0.0  # 初始化最佳F1分数
...
if val_acc > best_val_acc:
    ...
    print(f"... with F1: {best_val_acc:.4f}")
```

Selection is validation **accuracy**. The log line says F1.

## 3. Subject 3 is reindexed; row-wise means mix sentences

`DataTransformer` drops Task 1 subject 2 (file `3_SR.csv`) original sentences **150–249 and 399**, then writes `id` 0–298.

| Compacted `id` in `3_SR.csv` | Original sentence | Same row in `1_SR.csv` |
| ---: | ---: | ---: |
| 0–149 | 0–149 | original 0–149 (**aligned**) |
| 150–298 | 250–398 | original 150–298 (**not aligned**) |

Check (SentLen is constant across subjects for a real sentence):

- rows 0–149: `3_SR.SentLen == 1_SR.SentLen` for every row
- rows 150–298: almost never equal
- `3_SR.SentLen[150:161] == 1_SR.SentLen[250:261]`

`get_average_sentence_level.py` does `pd.concat(...).groupby(level=0).mean()`, i.e. mean by **row number**. After row 149, subject 3’s values belong to a different movie-review sentence than the other eleven subjects. Rows 299–399 are an 11-subject mean (subject 3 has no row).

Word-level `get_average.py` copies `Sent_ID` from subject 1 and averages by row index, so the same shift applies after the skipped region.

`examples/scripts/check_subject_alignment.py` prints the first mismatched row.

A safer average would key on the **original** sentence index (skip list applied as missing data, not as a compact RangeIndex).

## 4. Hard-coded Windows path for MATLAB files

```python
def get_matfiles(task: str, subdir='\\ZuCo_mat_data\\'):
    path = os.getcwd() + subdir + task
```

On Linux this is `cwd` + `\ZuCo_mat_data\` + `task1`, which is not `cwd/ZuCo_mat_data/task1`. The `.mat` tree is not in git in any case.

## 5. Script output directories do not match git

| Script | Writes / reads | Committed location |
| --- | --- | --- |
| `read_ZuCo_mat.py` | `et_csv_data/` | `ZuCo_et_csv_data/` |
| `get_average_sentence_level.py` | `et_csv_data/` | `ZuCo_et_csv_data/` |
| `ZuCo_SST_data/save_SST_data.py` | `output.csv` | `ssts_ZuCo.csv` via `convert_full_SST.py` |

## 6. `spilt.py` filename and unstratified ZuCo split

Typo for `split.py`, two copies. `train_test_split` is not stratified. Fine for 11,853 SST rows; awkward for 40-row ZuCo valid/test (valid labels 7 / 14 / 19). The CV training path does not use those files.

## 7. Hard-coded `view(-1, 3)`

Gaze-branch loss always reshapes to 3 classes. Harmless while `num_labels = 3`, but it ignores the `num_labels` variable.

## 8. No seeds in the training loop

KFold / `train_test_split` pass `random_state=42`. Dropout, DataLoader shuffle, and torch init are unseeded.

## 9. `models/` is not created

`torch.save(..., 'models/best_{model_type}_model.pth')` fails if the folder is missing. `.gitignore` ignores `models/` and `*.pth` so checkpoints stay local.

## 10. Duplicate `EyeTrackingModel`

`model_ZuCo_SST.py` and `model_full_SST.py` each define the class. Behavior matches today; a future edit can drift. The NumPy clone in `examples/teag_examples/fusion.py` follows this shared definition.

## 11. Predicted SST gaze is nearly collinear

`nFix`, `FFD`, `GPT`, `TRT` on `combined_full_sst_et.csv` have pairwise r ≥ 0.986. Concatenating five channels is not five independent cognitive measurements. Documented with numbers in [gaze-features.md](gaze-features.md).

## 12. `sentence_id` is ordered by the original SST file

r(`sentence_id`, `sentiment_label`) ≈ −0.37 on full SST. Do not feed the id to a model. Shuffled splits still have disjoint ids (checked in examples).

## 13. RoBERTa `pooler_output`

The fusion model uses `base_output.pooler_output` for both BERT and RoBERTa. If a text-only RoBERTa run and a fused RoBERTa run disagree, part of the gap may be this pooling choice rather than gaze.

## 14. Word-level vs WordPiece mismatch

`sst_et_test.csv` / `prediction_test_v2.csv` tokenize with NLTK or the predictor’s tokenizer. Training uses BERT/RoBERTa tokenizers at length 128. There is no alignment layer.

## 15. Averaging zeros at word level

In `ZuCo_et_csv_data/word/get_average.py` the `replace(0, np.nan)` line is commented out, so unfixated words (0 ms) pull the 12-subject mean down. Sentence-level averaging does replace 0 with NaN.
