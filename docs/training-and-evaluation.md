# Training and evaluation

## Shared knobs

| Knob | ZuCo script | Full SST script |
| --- | --- | --- |
| File | `model_ZuCo_SST.py` | `model_full_SST.py` |
| Data | `ZuCo_SST_data/combined_sst_et_standard.csv` | `SST_data/{train,valid,test}_full_sst.csv` |
| Epochs | 20 | 5 |
| Batch size | 16 (hardcoded in the `DataLoader`, not the `batch_size` constant) | 256 |
| LR | 5e-5 Adam | 5e-5 Adam |
| Max length | 128 | 128 |
| Folds | 5-fold stratified, `random_state=42` | single split |
| Checkpoint | none (prints fold + mean) | `models/best_{model_type}_model.pth` on best val acc |
| Device | CUDA if available else CPU | same |

The ZuCo script defines `batch_size = 16` at the top and then ignores it (`DataLoader(..., batch_size=16)`). The full SST script uses the constant.

## ZuCo loop (Track A)

```python
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
for train_index, test_index in kf.split(df_tokenized, df_tokenized['sentiment_label']):
    # new model, new optimizer, 20 epochs
    # evaluate once on the held-out fold
    # store acc / P / R / F1
```

Properties:

- Each fold trains from Hugging Face initial weights, not from the previous fold.
- The held-out fold is named `test_df` but it is a CV validation fold. There is no nested test set.
- Metrics are **weighted** (`average='weighted'`) so the majority class dominates.
- No early stopping. All 20 epochs always run.
- No seed for `torch` / `numpy` / `random` beyond sklearn's fold seed. Encoder init and dropout are not reproducible as-is.

After five folds the script prints means. It does not print standard deviations; `examples/gazekit/metrics.py` can compute them if you paste the five numbers.

## Full SST loop (Track B)

One model, five epochs, evaluate on `valid_loader` after each epoch, keep the highest `val_acc` state dict, reload it, run `test_loader`.

Caveats, all observed in the current file:

1. **Test predictions are overwritten per batch.**

   ```python
   all_preds = preds.cpu().numpy()
   all_labels = labels.cpu().numpy()
   ```

   The ZuCo script correctly `extend`s. The full SST test loop does not. With `batch_size=256` and 1,186 test rows that means the printed test metrics are computed on the **last batch only** (~178 rows). Validation metrics in the same file are fine (`extend`).

2. The save-message says `"with F1"` but the comparison is accuracy.

3. `torch.load` is called without `map_location`. A CPU-only machine cannot reload a GPU checkpoint.

4. `models/` is not created by the script. `torch.save` will fail if the directory is missing.

## Metrics

`calculate_metrics` in both scripts:

```python
accuracy_score
precision_score(..., average='weighted')
recall_score(..., average='weighted')
f1_score(..., average='weighted')
```

Weighted precision/recall on a 3-class problem with imbalance will look optimistic next to macro-F1. Neutral movie-review sentences are the usual minority. When you compare a gaze-only logistic model to these numbers, use the same averaging. The example baseline reports **both** weighted and macro.

`zero_division` is left at sklearn's default (warn + 0.0). A fold that never predicts a class will warn.

## Class balance (expected picture)

Exact counts change if you reshuffle, but on this checkout:

- ZuCo 400 is small enough that a 40-row test split can swing several points of accuracy by chance. That is why Track A uses stratified CV.
- Full SST is large enough that a non-stratified 80/10/10 is acceptable, but you should still print the three histograms before claiming a 0.5-point gain.

`examples/scripts/01_explore_zuco_sst.py` and `04_split_sanity_check.py` print the histograms.

## Baselines that should be beaten

Before treating a fusion run as interesting, compare it to:

| Baseline | How | Typical role |
| --- | --- | --- |
| Majority class | `examples/gazekit/baselines.fit_majority` | Accuracy floor |
| Length-only logistic | `n_tokens` (+ maybe `omissionRate`) | Difficulty confound |
| Gaze-only logistic | five ET channels | "does gaze even correlate?" |
| Text-only BERT/RoBERTa | `model_type='roberta'` | Encoder floor |
| Fusion | `model_type='roberta_eye_tracking'` | The actual hypothesis |

If gaze-only logistic is at majority, fusion is unlikely to help unless the encoder uses gaze as a tiny residual. If gaze-only logistic is well above majority, fusion *should* be able to harvest it — and if it does not, look at scaling or the missing activation on the 5→16 layer.

## Compute notes

- Track A: 5 folds × 20 epochs × ~320 rows / 16 ≈ 2,000 optimizer steps per fold, full RoBERTa. A single 16 GB GPU is enough.
- Track B: 5 epochs × ~9,482 / 256 ≈ 185 steps/epoch. Memory is dominated by `batch_size=256` × 128 tokens × RoBERTa. Lower the batch size if you OOM; the script has no grad accumulation.
- CPU-only: Track A is hours-to-overnight; Track B is not fun. Use the example scripts instead.

## Logging

There is no TensorBoard, no CSV logger, and no seed dump. The `tqdm` postfix shows running mean train loss. Validation prints one line per fold (Track A) or per epoch (Track B).

If you add logging later, write it next to the example reports under `examples/output/` (gitignored) rather than into `result/`, which already holds the legacy scatter plots.
