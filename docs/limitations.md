# Limitations and personal next steps

Known issues in the current tree, plus experiments that are in scope for this personal project and out of scope.

## Bugs in the original scripts

### Full SST test metrics are last-batch-only

In `model_full_SST.py`, the test loop assigns instead of extending:

```python
all_preds = preds.cpu().numpy()
all_labels = labels.cpu().numpy()
```

Validation in the same file is correct. Track A is correct. Only the final test print is wrong. Fix when you next touch that script: use `extend` like the ZuCo file. Until then, report validation numbers or recompute from saved logits.

### Checkpoint directory is not created

`torch.save(..., 'models/best_…pth')` assumes `models/` exists. `mkdir -p models` is in [reproduction.md](reproduction.md).

### Comment / metric mismatch

Best model is selected by accuracy. The log line says F1.

### Batch size constant is unused on Track A

`batch_size = 16` is duplicated as a literal. Harmless until someone edits only the constant.

### Windows paths in `get_matfiles`

`subdir='\\ZuCo_mat_data\\'` will not join correctly on Linux. The checked-in CSVs make this irrelevant until you re-export from `.mat`.

### `inplace` fillna on slices

`df.iloc[:, :].fillna(0, inplace=True)` can warn or no-op on recent pandas. Regenerated tables should use `df = df.fillna(0)`.

## Evaluation leaks and weak spots

1. **Scaler fit on all 400 ZuCo rows** before CV. Small leak of test-fold mean/std into the features.
2. **No torch seed.** Fold indices are fixed; weights are not.
3. **Weighted metrics only.** Macro-F1 can disagree, especially on 40-row slices.
4. **Convenience splits are not stratified.** `spilt.py` uses a plain `train_test_split`.
5. **Subject 3 has 299 sentences.** Reader means on some rows average 11 people, not 12, with no `n_readers` column.

## Confounds

- Sentence length, word length, frequency, and punctuation density all move gaze.
- Movie-review *style* (lists of names, parenthetical asides, `-LRB-` tokens in the full SST dump) moves both gaze predictors and tokenizers.
- Pupil size is in the CSVs but is a luminance/arousal stew. Do not interpret it as emotion.
- Predicted gaze on full SST can leak SST label information if the predictor was trained on review-like text with sentiment-correlated difficulty. Treat Track B as "does a gaze-shaped side channel help," not "do readers' eyes help."

## Architecture limits

- Late fusion of a 5-d sentence mean cannot represent "the eyes bounced on the negation."
- No activation on the 5→16 layer.
- RoBERTa still uses `pooler_output`.
- Adam instead of AdamW, no warmup, no weight decay.
- Hardcoded `view(-1, 3)` will break if you change `num_labels`.

## Data limits

- **Glued tokens** in `word_averages_v2.csv` (`murderoncampus`, `allwiseguysallthetime`, `emp11111ty`) come from stripping punctuation without inserting spaces. They inflate TRT/GPT and leak into sentence means.
- 400 labeled ZuCo sentences is the entire observed-gaze universe for this project. On this checkout, 5-fold gaze-only logistic is 36.8% vs a 35.0% majority — inside fold noise. Gains under 2–3 accuracy points are probably noise.
- Full SST labels are 3-way here; many SST papers use binary (fine/coarse) or 5-way fine-grained. Do not compare published SST-2 numbers to these scripts.
- The original `.txt` dump and `.mat` files are not in git, so a clean-room rebuild of `ssts_ZuCo.csv` is not possible from this checkout alone.

## Personal next steps (in scope)

These are small enough to do in this repo without turning it into a framework:

1. Patch the Track B test loop and add a 10-line unit test that the metric helper `extend`s.
2. Fit `StandardScaler` inside each ZuCo fold in a *new* example, not by silently changing the historical CSVs.
3. Word-length residual: regress each ET channel on `n_tokens` and reuse the residual in the logistic baseline.
4. Report fold standard deviations next to means.
5. Add a ReLU (or GELU) on the gaze projection as an optional `model_type` suffix, still in a personal branch.

## Out of scope

- Production inference service
- EEG fusion (ZuCo has it; these scripts never load it)
- Multilingual reviews
- Company datasets, internal labels, or anything that is not ZuCo / SST / PROVO
- Publishing a paper from these notes without a proper seed sweep and a nested test set

## How the example library is supposed to help

`examples/gazekit` exists so the next personal session can answer "is this CSV the one I think it is?" in under a minute: row counts, class balance, NaNs, channel correlations, majority accuracy, gaze-only logistic, split leakage. It will not replace `model_*.py`. It will keep those scripts honest.
