# Training and evaluation

## Shared hyperparameters

| Knob | ZuCo script | Full-SST script |
| --- | --- | --- |
| `num_eye_tracking_features` | 5 | 5 |
| `hidden_layer_size` | 16 | 16 |
| `num_labels` | 3 | 3 |
| `num_epochs` | 20 | 5 |
| `learning_rate` | 5e-5 | 5e-5 |
| `batch_size` | 16 | 256 |
| default `model_type` | `roberta_eye_tracking` | `roberta_eye_tracking` |
| tokenizer max length | 128 | 128 |

ZuCo's smaller batch and longer schedule match the 400-row table. Full SST
uses a large batch (256) and only five epochs.

## ZuCo protocol

`model_ZuCo_SST.py` runs `StratifiedKFold(n_splits=5, shuffle=True,
random_state=42)` over `combined_sst_et_standard.csv`. Each fold:

1. Builds `CustomDataset` views of the tokenized frame
2. Trains a **fresh** encoder + fusion head for 20 epochs
3. Evaluates once on that fold's hold-out (no inner validation, no
   early stopping, no `best_*.pth`)

It then prints the mean of accuracy / weighted precision / weighted recall /
weighted F1 across the five folds.

Because there is no model checkpoint, you cannot recover a particular fold's
weights after the script exits. The stored `ZuCo_SST_data/{train,valid,test}.csv`
files are unused here.

## Full-SST protocol

`model_full_SST.py` loads the three split files, trains for 5 epochs, and
after every epoch computes validation accuracy / P / R / F1. It writes
`models/best_{model_type}_model.pth` whenever validation **accuracy**
improves (the comment says F1; the comparison is `val_acc`). After training
it reloads that checkpoint and runs the test loader.

### Test-loop bug

The test loop **replaces** `all_preds` / `all_labels` on every batch instead
of extending them:

```python
preds = torch.argmax(logits, dim=1)
all_preds = preds.cpu().numpy()
all_labels = labels.cpu().numpy()
```

Reported test metrics are therefore the last batch only (up to 256 rows),
not the 1,186-row test set. Validation metrics in the same file use
`.extend` and are fine. See [known limitations](notes/known-limitations.md).

## Metrics

`calculate_metrics` always uses `average='weighted'` for precision, recall,
and F1. On ZuCo that is close to macro because the labels are balanced. On
full SST weighted F1 can look healthy while the rare neutral class is
ignored.

The examples library reports **macro F1** as well for the sklearn baselines,
precisely because full SST is 19% neutral and the ZuCo stored valid split is
unbalanced.

## Majority baselines

| Table | Majority class | Accuracy if you always predict it |
| --- | --- | ---: |
| ZuCo ∩ SST (400) | positive (140) | 0.350 |
| Full SST (11,853) | positive (4,963) | 0.419 |

Any fused model has to beat the matching text-only run, not just these
dummies. The interesting delta is:

```
fused − text-only    on the same seed / fold / split
```

not `fused − majority`.

## Linear baselines (no GPU)

`examples/06_text_vs_gaze_baselines.py` fits three logistic models with the
same 5-fold stratification the ZuCo script uses:

1. `gaze_only` — the five (or chosen) gaze columns, standard-scaled
2. `text_only` — TF-IDF unigrams+bigrams, 4,000 features, `min_df=2`
3. `text_plus_gaze` — FeatureUnion of (2) and (1)

These numbers will not match RoBERTa. They answer a cheaper question: does
sentence-level gaze move a linear text model at all?

A run on this clone (seed 42, five gaze channels) landed at:

| model | mean acc | mean macro F1 |
| --- | ---: | ---: |
| `gaze_only` | 0.348 | 0.336 |
| `text_only` | 0.497 | 0.491 |
| `text_plus_gaze` | 0.472 | 0.465 |

Gaze-only matches the 0.350 majority baseline. Fusion **lost** 2.5
accuracy points to text-only. Full fold table:
[notes/linear-baselines.md](notes/linear-baselines.md).


## Device

Both scripts pick `cuda` if `torch.cuda.is_available()` else `cpu`. There is
no multi-GPU, mixed precision, or gradient accumulation.

## Checkpoints

Only the full-SST script saves weights, and only under `models/` (gitignored).
The fused `state_dict` includes `base_model.*`, `eye_tracking_layer.*`, and
`classifier.*`. Loading requires constructing the same `EyeTrackingModel`
class; you cannot `from_pretrained` those files.
