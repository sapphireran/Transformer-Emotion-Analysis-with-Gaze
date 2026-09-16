# Model architecture

Both `model_ZuCo_SST.py` and `model_full_SST.py` define the same two classes and the same factory. They are copy-pasted, not imported from a shared module. If you change one, change the other or extract a `models/` package in a later personal cleanup.

## Factory

```python
def get_model(model_type, num_eye_tracking_features, num_labels):
    if model_type == 'bert':
        return BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=num_labels)
    elif model_type == 'roberta':
        return RobertaForSequenceClassification.from_pretrained('roberta-base', num_labels=num_labels)
    elif model_type == 'bert_eye_tracking':
        return EyeTrackingModel(BertModel, num_eye_tracking_features, num_labels)
    elif model_type == 'roberta_eye_tracking':
        return EyeTrackingModel(RobertaModel, num_eye_tracking_features, num_labels)
```

Tokenizer choice is a prefix check: anything starting with `bert` uses `BertTokenizer`; otherwise `RobertaTokenizer`. That means a typo like `robert` would still load RoBERTa.

Checkpoints:

| Encoder | Hugging Face id | Hidden size |
| --- | --- | ---: |
| BERT | `bert-base-uncased` | 768 |
| RoBERTa | `roberta-base` | 768 |

## EyeTrackingModel

```text
EyeTrackingModel
├── base_model          BertModel or RobertaModel (pretrained)
├── eye_tracking_layer  Linear(num_eye_tracking_features → hidden_layer_size)
├── dropout             Dropout(0.1)
└── classifier          Linear(hidden_size + hidden_layer_size → num_labels)
```

Defaults from the scripts:

| Hyperparameter | ZuCo script | Full SST script |
| --- | ---: | ---: |
| `num_eye_tracking_features` | 5 | 5 |
| `hidden_layer_size` | 16 | 16 |
| `num_labels` | 3 | 3 |
| `num_epochs` | 20 | 5 |
| `learning_rate` | 5e-5 | 5e-5 |
| `batch_size` | 16 | 256 |
| default `model_type` | `roberta_eye_tracking` | `roberta_eye_tracking` |

Forward (gaze variants):

1. `base_output = base_model(input_ids, attention_mask)`
2. `pooled = base_output.pooler_output` — BERT: `tanh(W · h_[CLS])`. RoBERTa: the same `pooler_output` field exists on `RobertaModel` and is a linear+tanh on the first token (`<s>`).
3. `gaze_h = Linear(5, 16)(eye_tracking_features)` — **no activation**.
4. `h = dropout(concat(pooled, gaze_h))` → shape `[batch, 784]`
5. `logits = Linear(784, 3)(h)`

There is no LayerNorm on the gaze branch and no learned gate. The classifier can ignore gaze by driving those 16 incoming weights toward zero.

## Loss

- Text-only: Hugging Face sequence-classification loss (`CrossEntropyLoss` inside the HF head).
- Gaze: `CrossEntropyLoss()(logits.view(-1, 3), labels.view(-1))` computed in the training loop.

Labels are integers `{0, 1, 2}`. No class weights, despite SST being ~2:1:2 and ZuCo being almost balanced.

## Tokenization

```python
tokenizer(examples['sentence'], padding='max_length', truncation=True, max_length=128)
```

Applied through Hugging Face `datasets.Dataset.map(..., batched=True)`, then converted back to pandas so the gaze columns can be concatenated by row index. That implicit alignment assumes `eye_tracking_features` and `text_data` came from the same `read_csv` and were never reordered independently.

`CustomDataset` stores:

- `input_ids`, `attention_mask` as Python lists of lists (one more memory copy than a tensor dataset)
- `labels` from `sentiment_label`
- `eye_tracking_features` as a NumPy array of shape `[N, 5]`

`__getitem__` wraps each field in `torch.tensor` on the fly.

## Optimizer

Plain `torch.optim.Adam` on **all** parameters at `5e-5`. No weight decay, no layer-wise decay, no freeze-then-unfreeze schedule. For 400 ZuCo sentences this is aggressive: 20 epochs × 5 folds will happily memorize.

## Evaluation

`sklearn.metrics` with `average='weighted'` for precision, recall, and F1, plus accuracy.

| Script | Validation | Selection | Test |
| --- | --- | --- | --- |
| `model_ZuCo_SST.py` | each fold’s hold-out after 20 epochs | none (no early stopping) | mean of 5 fold metrics |
| `model_full_SST.py` | `valid_full_sst.csv` every epoch | best **accuracy** → `models/best_{type}_model.pth` | reload best, run `test_full_sst.csv` |

`model_full_SST.py` comments say “best F1” in two places; the code compares `val_acc`. The print after save also labels the number as F1. See `docs/experiments.md`.

## Why 16 gaze units

16 is small enough that the 768-d text vector dominates capacity, which is the point of the ablation: if gaze helps, it is because the five numbers are informative, not because we added a large extra network. `examples/gaze_fusion_demo.py` reconstructs the tensor shapes without downloading weights.

## What a later personal refactor would change

These are notes, not work in this pass:

- Share `EyeTrackingModel` / `CustomDataset` / `calculate_metrics` in one module.
- Add a ReLU or GELU on the gaze projection.
- Optional FiLM / gated fusion instead of concat.
- Word-level gaze attention aligned to tokenizer word pieces.
- Class-weighted loss on full SST.
- Fix the test-loop overwrite in `model_full_SST.py` (documented under known issues).
