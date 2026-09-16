# Architecture

Both trainers implement the same idea:

```
sentence text ──► BERT or RoBERTa ──► pooler_output (768)
                                              │
                                              ├─ concat ─► dropout 0.1 ─► Linear ─► 3 logits
                                              │
5 gaze numbers ──► Linear(5, 16) ─────────────┘
```

The class is `EyeTrackingModel` in `model_ZuCo_SST.py` and `model_full_SST.py`.
The two files are copies, not a shared module.

## Text side

`model_type` picks the tokenizer and the backbone:

| `model_type` | Tokenizer | Backbone | Head |
| --- | --- | --- | --- |
| `bert` | `BertTokenizer` `bert-base-uncased` | `BertForSequenceClassification` | Hugging Face classifier |
| `roberta` | `RobertaTokenizer` `roberta-base` | `RobertaForSequenceClassification` | Hugging Face classifier |
| `bert_eye_tracking` | BERT tokenizer | `BertModel` + custom concat | `EyeTrackingModel` |
| `roberta_eye_tracking` | RoBERTa tokenizer | `RobertaModel` + custom concat | `EyeTrackingModel` |

Default in both scripts is `roberta_eye_tracking`.

Tokenization is always:

```python
tokenizer(sentence, padding="max_length", truncation=True, max_length=128)
```

That is enough for this corpus (ZuCo max 43 words, full SST max 56 words) but
it wastes a lot of pad tokens. The Hugging Face `Dataset.map` path materializes
`input_ids` and `attention_mask` into a pandas frame before `CustomDataset`
wraps them as tensors.

## Gaze side

Five floats go through one linear layer (`hidden_layer_size = 16`) with **no
nonlinearity** in the original `EyeTrackingModel`. The example toy model in
`tea_gaze.models` adds `tanh` only so the demo stays bounded; the training
scripts do not.

The five floats are already scaled in the CSVs the trainers load. There is no
`StandardScaler` inside the `DataLoader`.

## Fusion head

```
combined = dropout(concat(pooler_output, Linear(et)))
logits   = Linear(768 + 16, 3)
```

Loss for the custom models is `CrossEntropyLoss` on those logits. The plain
`bert` / `roberta` variants use the built-in HF loss instead and **ignore**
the eye-tracking tensor even though `CustomDataset` still loads it.

Dropout is 0.1. Optimizer is `Adam` at `5e-5` with no weight decay and no
warmup. There is no learning-rate schedule.

## What “pooler_output” is

For BERT, `pooler_output` is `tanh(W * h_[CLS])`. For RoBERTa, Hugging Face
still exposes a pooler, but a lot of RoBERTa classification code uses the raw
first token hidden state instead. These scripts always take `.pooler_output`.
That is a modeling choice, not a bug, but it is one reason a “RoBERTa + gaze”
run is not an exact replica of a typical `RobertaForSequenceClassification`
head.

## Training loops (they are not the same)

### `model_ZuCo_SST.py` — small real-gaze set

- Data: `ZuCo_SST_data/combined_sst_et_standard.csv` (400 rows)
- Split: `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`
- Epochs: 20 per fold, **no validation checkpoint inside the fold**
- Batch size: 16
- Metrics: weighted accuracy / P / R / F1 on the fold test slice after epoch 20
- Prints the mean of those five fold scores
- Does not write a `models/*.pth` file

### `model_full_SST.py` — large predicted-gaze set

- Data: `SST_data/train_full_sst.csv`, `valid_full_sst.csv`, `test_full_sst.csv`
- Epochs: 5
- Batch size: 256
- After each epoch, scores the valid loader and writes
  `models/best_{model_type}_model.pth` when **validation accuracy** improves
- Reloads that checkpoint and scores the test loader

The comment next to `best_val_acc` says “F1” but the comparison uses accuracy.
The test loop currently **overwrites** `all_preds` / `all_labels` on every
batch instead of extending them, so the printed test score is the last batch
only. See [known-issues.md](known-issues.md).

## Geometry cheat sheet

| Piece | Size |
| --- | ---: |
| Tokenizer max length | 128 |
| Transformer hidden size | 768 |
| ET input | 5 |
| ET hidden | 16 |
| Concat | 784 |
| Labels | 3 |
| Dropout | 0.1 |

`tea_gaze.models.ToyEyeTrackingFusion` keeps the 5→16→concat story and
replaces the 768-d encoder with a 48-d hashed bag-of-words so
`examples/04_toy_fusion_forward.py` can run offline.
