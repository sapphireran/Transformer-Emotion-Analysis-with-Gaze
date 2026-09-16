# Models and training

## Choosing a model type

Set `model_type` at the top of the training script. Allowed values:

| `model_type` | Encoder | Gaze branch | Hugging Face class or custom |
| --- | --- | --- | --- |
| `bert` | `bert-base-uncased` | no | `BertForSequenceClassification` |
| `roberta` | `roberta-base` | no | `RobertaForSequenceClassification` |
| `bert_eye_tracking` | `bert-base-uncased` | yes | `EyeTrackingModel(BertModel, …)` |
| `roberta_eye_tracking` | `roberta-base` | yes | `EyeTrackingModel(RobertaModel, …)` |

Default in both files is `roberta_eye_tracking`.

## Shared hyperparameters

| Knob | Value | Notes |
| --- | --- | --- |
| Max token length | 128 | Padded; short movie-review sentences fit |
| Optimizer | Adam | No weight decay, no schedule |
| Learning rate | 5e-5 | Typical fine-tune LR |
| Dropout after concat | 0.1 | Fusion models only |
| Gaze hidden size | 16 | Single Linear, no activation |
| Labels | 3 | Weighted P/R/F1 via sklearn |

## ZuCo protocol (`model_ZuCo_SST.py`)

- **Data:** `ZuCo_SST_data/combined_sst_et_standard.csv` (400 rows)
- **Split:** `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`
- **Epochs:** 20 per fold
- **Batch size:** 16 (hard-coded again when building DataLoaders; the `batch_size` constant is unused for the loaders)
- **Checkpoint:** none
- **Reported number:** mean of five holdout scores after the last epoch

This is the run to use when you want **measured** gaze. 400 rows is small enough that a 16-d gaze projection can overfit a fold if you are not careful. The text encoder is still ~110M parameters; treat fold scores as noisy.

Suggested personal log line after a run:

```
model_type=...  fold_acc=[...]  mean_acc=...  mean_f1=...  seed=42  epochs=20
```

## Full SST protocol (`model_full_SST.py`)

- **Data:** `SST_data/train_full_sst.csv` (9482), `valid_full_sst.csv` (1185), `test_full_sst.csv` (1186)
- **Epochs:** 5
- **Batch size:** 256
- **Checkpoint:** `models/best_{model_type}_model.pth` on best **validation accuracy**
- **Test:** reload checkpoint, score test loader

Create `models/` first:

```bash
mkdir -p models
python3 model_full_SST.py
```

Read [known-issues.md](known-issues.md) before trusting the printed test line. The test loop currently keeps only the last batch.

Neutral reviews are ~19% of SST. Weighted F1 will look healthier than macro F1. If you add a metric, log both.

## Metrics helper

```python
accuracy_score(labels, preds)
precision_score(..., average='weighted')
recall_score(..., average='weighted')
f1_score(..., average='weighted')
```

No confusion matrix is printed. `examples/dummy_baseline.py` writes one for the gaze-only and majority baselines so you have a template.

## What "better with gaze" would look like

A fair comparison is four runs that differ only in `model_type`:

1. `roberta` on ZuCo CV
2. `roberta_eye_tracking` on ZuCo CV
3. `roberta` on SST holdout
4. `roberta_eye_tracking` on SST holdout

Keep tokenizer, max length, epochs, batch size, and seed matched. If fusion wins on ZuCo and loses on SST, that is still a useful result: it suggests the transferred gaze is the weak link, not the concat head.

`examples/dummy_baseline.py` gives the floor (recorded in [baselines.md](baselines.md)):

- Majority class on ZuCo (always 2 / positive): 140/400 = **0.350** accuracy
- Gaze-only logreg on ZuCo, same 5 folds: **0.3675** mean accuracy
- Majority class on SST test (always 2): **0.4418** accuracy (524/1186)
- Gaze-only logreg on SST test: **0.4368** accuracy, **0.387** weighted F1, **never predicts neutral**

## Memory and runtime (order of magnitude)

These are planning numbers, not benchmarks from this machine.

| Run | Sequences / epoch | Tokens padded | Rough GPU need |
| --- | --- | --- | --- |
| ZuCo fusion, batch 16 | 320 train rows / fold | 128 | 8 GB is comfortable |
| SST fusion, batch 256 | 9482 | 128 | 16 GB preferred; cut batch to 32 on smaller cards |

CPU training of `roberta-base` on SST is possible but slow. The example suite is designed so you can still inspect data without that cost.

## Saving more than a checkpoint

The training scripts print metrics to stdout only. A lightweight personal convention:

```
result/logs/{date}_{script}_{model_type}.txt
```

Redirect:

```bash
python3 -u model_ZuCo_SST.py | tee result/logs/$(date +%F)_zuco_roberta_et.txt
```

Do not commit large `.pth` files; `.gitignore` already excludes `models/` and `*.pth`.
