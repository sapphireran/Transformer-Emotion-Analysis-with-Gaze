# Training, splits, and metrics

Side-by-side of the two scripts, then the measurement bugs I do not
want to rediscover.

## Hyperparameters as they are in the files

| Knob | `model_ZuCo_SST.py` | `model_full_SST.py` |
| --- | --- | --- |
| `num_eye_tracking_features` | 5 | 5 |
| `hidden_layer_size` | 16 | 16 |
| `num_labels` | 3 | 3 |
| `num_epochs` | **20** | **5** |
| `learning_rate` | 5e-5 | 5e-5 |
| `batch_size` | 16 (also hard-coded in the loaders) | **256** |
| default `model_type` | `roberta_eye_tracking` | `roberta_eye_tracking` |
| optimizer | Adam (not AdamW) | Adam (not AdamW) |
| schedule / warmup | none | none |
| dropout | 0.1 on the concat | 0.1 on the concat |
| seed | KFold `random_state=42` only | split seed 42; training unseeded |
| checkpointing | none | `models/best_{model_type}_model.pth` on val acc |

Adam vs AdamW matters more on the full-SST run (weight decay would
regularize the 125M text weights). I left the optimizer as written.

## Splits

### ZuCo — 5-fold stratified CV

```python
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
for train_index, test_index in kf.split(df_tokenized, df_tokenized['sentiment_label']):
    ...
```

400 rows → 320 / 80 per fold, class ratios preserved. Each fold
**re-initializes** `get_model(...)`, so you get five independent
fine-tunes, not one model that saw every split.

There is no inner validation set. The 80-row slice is used once, after
20 epochs, as the fold score. Early stopping is therefore impossible
in this script. 20 epochs on 320 RoBERTa-finetune sentences is
aggressive; the fold score is "accuracy after a long train," not
"accuracy at the best epoch."

The unused 80/10/10 files (`train.csv` 320, `valid.csv` 40,
`test.csv` 40) come from `ZuCo_SST_data/spilt.py`. Valid label counts
are 7 / 14 / 19 — do not quote those 40 rows as a balanced test.

### Full SST — hold-out

`SST_data/spilt.py`:

```
train_test_split(df, test_size=0.2, random_state=42)
train_test_split(valid_test_df, test_size=0.5, random_state=42)
```

That is 80 / 10 / 10, **not stratified**. Realized counts:

| Split | N | neg | neu | pos |
| --- | --- | --- | --- | --- |
| train | 9,482 | 3,710 | 1,833 | 3,939 |
| valid | 1,185 | 476 | 209 | 500 |
| test | 1,186 | 463 | 199 | 524 |
| all | 11,853 | 4,649 | 2,241 | 4,963 |

Neutral is ~19% everywhere, which is close enough that I do not
panic, but a stratified split would have been the default if I
rewrote `spilt.py`.

## Metrics

```python
accuracy_score
precision_score(..., average='weighted')
recall_score(..., average='weighted')
f1_score(..., average='weighted')
```

Weighted-average P/R/F1 on a 3-class problem with a minority
neutral class will look similar to accuracy. If I want to know whether
gaze helps the *neutral* reviews (the ones where fusion is most
plausible), I need per-class F1. The scripts do not print that.

`zero_division` is left at sklearn's default. A fold that never
predicts neutral will warn and put 0.0 in that class's precision.

## The full-SST test loop bug

In `model_full_SST.py`, training and validation **extend** the
prediction lists:

```python
all_preds.extend(preds.cpu().numpy())
all_labels.extend(labels.cpu().numpy())
```

The test loop **assigns**:

```python
all_preds = preds.cpu().numpy()
all_labels = labels.cpu().numpy()
```

So the printed `Test Acc / P / R / F1` are computed on **the last
batch only** (up to 256 of 1,186 rows). I have not patched the script
in this documentation pass; I am writing it down so I do not quote
that test line as a result. Validation numbers in the same file are
fine. The ZuCo script's fold loop uses `extend` and is fine.

Related comment rot in the same file:

```python
best_val_acc = 0.0  # 初始化最佳F1分数
...
print(f"... with F1: {best_val_acc:.4f}")
```

The number is accuracy. The comment and the f-string say F1.

## Checkpointing

Only the full-SST script saves weights, and only when validation
accuracy improves. The path is `models/best_{model_type}_model.pth`.
The directory is not created in-repo; `torch.save` will fail if
`models/` does not exist. After the loop it reloads that file and
runs the (buggy) test pass.

The ZuCo script never saves. Five fold models die at the end of the
process. That is acceptable for a mean-score experiment and annoying
if I want to probe sentence 4 with a trained fusion head.

## Optimization details worth remembering

- **No `model.zero_grad()` vs `optimizer.zero_grad()` issue** —
  `optimizer.zero_grad()` is what they call.
- **No clipping.** A bad batch on full SST can spike the RoBERTa
  gradients. I have not seen it, but I have also not kept logs.
- **Learning rate 5e-5** is the HF classification default. It is
  applied to *all* parameters, including the tiny gaze `Linear(5, 16)`.
  That layer could take a larger LR; I have not tried.
- **`tqdm` `leave=`** is set so only the last epoch's bar stays.
  Cosmetic.
- **CUDA vs CPU** is printed at start. There is no `torch.manual_seed`.
  Full-SST runs are not reproducible even with the split seed.

## What I would report (if I re-ran)

For Track A, one table:

```
model_type                mean Acc   mean weighted-F1   (5 folds)
bert
bert_eye_tracking
roberta
roberta_eye_tracking
```

plus per-class F1 for `roberta` vs `roberta_eye_tracking` on the
pooled out-of-fold predictions.

For Track B, validation curves per epoch (the script already prints
a val line each epoch) and a **fixed** test evaluation that extends
lists. Until that fix, I would not put a test number in a README.

## Runtime intuition (no GPU in this environment)

This cloud environment does not have the training stack installed
(`pandas` / `torch` / `transformers` are absent). I am not quoting
a wall-clock. Order-of-magnitude from memory of similar runs:

- ZuCo, 5 folds × 20 epochs × RoBERTa, batch 16: tens of minutes on
  one consumer GPU.
- Full SST, 5 epochs, batch 256: also tens of minutes; the tokenizer
  map over 11k rows is the slow CPU part on first launch.

The example scripts are seconds on CPU and are what this documentation
pass actually executes.
