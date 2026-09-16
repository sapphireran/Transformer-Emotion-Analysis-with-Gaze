# Architecture

Both training scripts implement the same four model types. The interesting one is `EyeTrackingModel`, a late-fusion classifier that never mixes gaze into the transformer layers. Gaze is projected independently and concatenated with the pooled text vector.

## Class: `EyeTrackingModel`

Defined in `model_ZuCo_SST.py` and duplicated in `model_full_SST.py`.

```
base_model: BertModel or RobertaModel
    input_ids, attention_mask
    → last hidden states
    → pooler_output          shape [B, 768]

eye_tracking_layer: Linear(5 → 16)
    gaze features            shape [B, 5]
    → projection             shape [B, 16]

classifier: Linear(768 + 16 → 3)
    concat(pooler, gaze_h)   shape [B, 784]
    → Dropout(0.1)
    → logits                 shape [B, 3]
```

Constants at the top of each script:

| Name | ZuCo script | Full SST script | Role |
| --- | --- | --- | --- |
| `num_eye_tracking_features` | 5 | 5 | Input width of the gaze Linear |
| `hidden_layer_size` | 16 | 16 | Gaze projection width |
| `num_labels` | 3 | 3 | Negative / neutral / positive |
| `num_epochs` | 20 | 5 | Full passes over the train loader |
| `learning_rate` | 5e-5 | 5e-5 | Adam |
| `batch_size` | 16 | 256 | SST uses a much larger batch |
| `model_type` | `roberta_eye_tracking` | `roberta_eye_tracking` | Switch |

There is no nonlinearity on the gaze branch. The projection is a single `nn.Linear`. Dropout is applied only after concatenation.

## Text-only path

When `model_type` is `bert` or `roberta`, `get_model()` returns `BertForSequenceClassification` or `RobertaForSequenceClassification` with `num_labels=3`. The training loop then uses the Hugging Face `labels=` argument and `outputs.loss`. Gaze tensors are still present on the `CustomDataset` items; they are simply not read.

## Tokenization

Both scripts tokenize with `max_length=128` and `padding='max_length'`. BERT uses `bert-base-uncased`; RoBERTa uses `roberta-base`. The tokenizer is chosen by `model_type.startswith('bert')`.

`CustomDataset` stores:

- `input_ids` and `attention_mask` as Python lists of lists (later wrapped with `torch.tensor`)
- `labels` from `sentiment_label`
- `eye_tracking_features` as a float32 matrix with one row per sentence

## Training loop differences

### ZuCo (`model_ZuCo_SST.py`)

- Loads `ZuCo_SST_data/combined_sst_et_standard.csv` once.
- Reads gaze from `nFixations, FFD, GPT, TRT, GD`.
- `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)` on `sentiment_label`.
- Each fold trains a **fresh** model for 20 epochs, then evaluates once on the fold's holdout.
- No validation split inside a fold and no checkpointing.
- Prints per-fold accuracy / weighted P / R / F1, then the mean across folds.

Because there is no early stopping, the reported fold score is the model after epoch 20, not the best epoch.

### Full SST (`model_full_SST.py`)

- Loads three files: `SST_data/train_full_sst.csv`, `valid_full_sst.csv`, `test_full_sst.csv`.
- Reads gaze from `nFix, FFD, GPT, TRT, GD` (same five quantities, different column names).
- Trains 5 epochs on the train loader, evaluates the valid loader every epoch, and saves `models/best_{model_type}_model.pth` when **validation accuracy** improves.
- Reloads that checkpoint and scores the test loader.

The comment next to `best_val_acc` says "best F1" but the comparison is on accuracy. The test loop also has a real bug: it **assigns** `all_preds` / `all_labels` instead of extending them, so only the last test batch is scored. See [known-issues.md](known-issues.md).

## Loss

Fusion models use `CrossEntropyLoss` on logits reshaped to `(-1, 3)`. Hugging Face text-only models use their built-in loss. There is no class-weighting despite the SST label skew.

## Why late fusion

The implementation is intentionally shallow:

1. The transformer is not asked to attend over gaze values.
2. Word-level gaze is collapsed to a sentence vector **before** the model (ZuCo: mean over fixated words in `DataTransformer`; SST: a single five-vector already on the row).
3. A 16-unit projection is small enough that, on 400 ZuCo sentences, it is unlikely to dominate the 768-d text vector unless the optimizer drives those weights up.

`examples/fusion_architecture_demo.py` rebuilds this geometry with NumPy so you can see the shapes without loading `roberta-base`.

## Device

Both scripts pick CUDA when `torch.cuda.is_available()` else CPU. The ZuCo run (400 rows, batch 16, 20 epochs × 5 folds) is the cheaper of the two. The SST run (batch 256, 5 epochs, ~9.5k train rows, 128-token padding) is memory-heavy on CPU.

## Duplicated code

`EyeTrackingModel`, `CustomDataset`, `calculate_metrics`, and `get_model` are copy-pasted across the two training files. The examples do not import those classes (the training files execute I/O at import time). If you refactor later, extract a `src/` module first so importing the model does not start a training run.
