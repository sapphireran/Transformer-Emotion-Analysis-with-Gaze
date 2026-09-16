# Training loops

The two scripts look similar but they answer different questions and
they do **not** share a library. Copy-paste drift is real: if you
fix a bug in one, fix the other.

## `model_ZuCo_SST.py` — measured gaze, 5-fold CV

**Data:** `ZuCo_SST_data/combined_sst_et_standard.csv` (400 rows),
tokenized once up front.

**Split:** `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`
on `sentiment_label`. Each fold trains on 320 sentences and evaluates
on 80. The committed `train.csv` / `valid.csv` / `test.csv` are unused.

**Loop, per fold:**

1. Fresh `get_model(...)` — **no** weight sharing across folds.
2. `Adam(lr=5e-5)`, `batch_size=16` (the header `batch_size` variable
   is ignored; the `DataLoader` hard-codes 16).
3. Train `num_epochs=20` with a tqdm bar. No validation **inside**
   the epoch loop.
4. After the last epoch, one eval pass on the fold’s test loader.
5. Append accuracy, weighted precision, recall, F1.

After five folds, print the **mean** of each metric. There is no
standard deviation printout, no per-fold model save, and no
confusion matrix.

**Implication:** you cannot pick a “best epoch.” Twenty epochs on 320
RoBERTa examples will often overfit. If you add early stopping, you
need a fold-internal val split (e.g. 4/1 of the 320) so the 80-row
test fold stays clean.

## `model_full_SST.py` — transferred gaze, fixed split

**Data:**

- `SST_data/train_full_sst.csv` (9,482)
- `SST_data/valid_full_sst.csv` (1,185)
- `SST_data/test_full_sst.csv` (1,186)

**Loop:**

1. One model, `Adam(lr=5e-5)`, `batch_size=256`, `num_epochs=5`.
2. Each epoch: train on all batches, then evaluate the **validation**
   set.
3. If `val_acc` improves, `torch.save(state_dict, models/best_{model_type}_model.pth)`.
4. After training, reload that checkpoint and evaluate the test set.

The comment next to `best_val_acc` says “best F1” but the code
compares **accuracy**. The save message also prints the accuracy
value with an `F1:` label. Believe the variable, not the string.

### Test-set metric bug

```python
preds = torch.argmax(logits, dim=1)
all_preds = preds.cpu().numpy()      # assignment, not extend
all_labels = labels.cpu().numpy()
```

The training and validation loops use `all_preds.extend(...)`. The
**test** loop **overwrites** the arrays every batch. Reported test
Acc / P / R / F1 are computed on the **last batch only** (up to 256
rows), not on 1,186 rows.

Until that is fixed, do not quote the printed test line. Validation
metrics in the same script are fine.

`examples/06_toy_late_fusion.py` uses `extend` on every split so the
lightweight baseline is not carrying this bug.

## Shared metric helper

```python
accuracy_score
precision_score(..., average='weighted')
recall_score(..., average='weighted')
f1_score(..., average='weighted')
```

Weighted averages follow class support. On full SST that means
neutral errors hurt the headline F1 less than they should if you
care about the minority class. Print a per-class report when you
compare fusion vs text-only.

`zero_division` is left at sklearn’s default (warn + 0). A fold that
never predicts neutral will warn.

## Batch contents

`CustomDataset.__getitem__` returns:

```
input_ids          long tensor (128,)
attention_mask     long tensor (128,)
labels             scalar (sentiment 0/1/2)
eye_tracking_features   float32 (5,)
```

Text-only branches still **load** gaze tensors (they are in every
batch) and then ignore them. That wastes a little RAM and keeps the
dataset class identical across `model_type`s.

## Checkpointing

Only `model_full_SST.py` saves weights. The directory `models/` is
not created by the script — `torch.save` will fail if it does not
exist. The README and `examples` do not create it either; mkdir
before the full-SST run.

ZuCo CV never writes a `.pth`. If you want a single deployable
model from the 400, train on all 400 after you have finished
comparing folds (and accept that you no longer have an unbiased
test number from that run).

## Randomness

| Source | Seeded? |
| --- | --- |
| `StratifiedKFold` | `random_state=42` |
| `DataLoader(shuffle=True)` | **no** `generator` |
| `torch` / CUDA / numpy global | **no** |
| Dropout | unseeded |
| HF weight init | HF default (pretrained, so encoder init is fixed; classifier head is not seeded) |

Fold **assignments** are reproducible. Training dynamics are not,
unless you add a seed block.

## Runtime ballpark (qualitative)

- ZuCo fusion, 5 folds × 20 epochs × 320 rows × batch 16: on the
  order of one GPU-hour for RoBERTa, more if the disk cache for
  `roberta-base` is cold.
- Full SST, 5 epochs × 9.5k × batch 256: usually faster per epoch
  than it looks, but 128-token padding makes each step heavy.

CPU-only fine-tuning is possible and unpleasant. Use the `examples/`
scripts when you only need to know whether the CSVs are healthy.
