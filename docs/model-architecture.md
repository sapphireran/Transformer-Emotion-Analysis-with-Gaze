# Model architecture

Both `model_full_SST.py` and `model_ZuCo_SST.py` implement the same four
`model_type` values and the same `EyeTrackingModel`. They differ only in
data source, batch size, epoch count, and how they validate
(hold-out vs 5-fold CV).

## Variants

```
model_type ∈ { bert, roberta, bert_eye_tracking, roberta_eye_tracking }
```

Text-only variants are stock Hugging Face sequence classifiers:

- `BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=3)`
- `RobertaForSequenceClassification.from_pretrained('roberta-base', num_labels=3)`

They ignore the ET tensor even though `CustomDataset` still loads it.

Gaze-aware variants wrap a bare encoder (`BertModel` / `RobertaModel`)
and add a late-fusion head.

## Late fusion (`EyeTrackingModel`)

```
                    input_ids, attention_mask
                                │
                                ▼
                     BERT / RoBERTa encoder
                                │
                          pooler_output
                            (B, 768)
                                │
                                │     eye_tracking_features (B, 5)
                                │                 │
                                │                 ▼
                                │        Linear(5 → 16)
                                │             (B, 16)
                                │                 │
                                └────────┬────────┘
                                         ▼
                                   concat dim=1
                                     (B, 784)
                                         │
                                    Dropout(0.1)
                                         │
                                   Linear(784 → 3)
                                         │
                                      logits
```

Source (shared by both training files):

```python
class EyeTrackingModel(nn.Module):
    def __init__(self, base_model, num_eye_tracking_features, num_labels):
        super().__init__()
        self.base_model = base_model.from_pretrained(
            'bert-base-uncased' if base_model == BertModel else 'roberta-base'
        )
        self.eye_tracking_layer = nn.Linear(num_eye_tracking_features, hidden_layer_size)
        self.classifier = nn.Linear(
            self.base_model.config.hidden_size + hidden_layer_size, num_labels
        )
        self.dropout = nn.Dropout(0.1)

    def forward(self, input_ids, attention_mask, eye_tracking_features):
        base_output = self.base_model(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = base_output.pooler_output
        eye_tracking_output = self.eye_tracking_layer(eye_tracking_features)
        combined_output = torch.cat((pooled_output, eye_tracking_output), dim=1)
        combined_output = self.dropout(combined_output)
        return self.classifier(combined_output)
```

Constants: `num_eye_tracking_features = 5`, `hidden_layer_size = 16`,
`num_labels = 3`. The gaze branch is a **single linear layer with no
nonlinearity**. That is late fusion, not a deep multimodal encoder:
gradients can still flow into the transformer through `pooler_output`,
and into the 5×16 matrix through the concatenated classifier.

### Shapes

| Tensor | Shape | Notes |
|---|---|---|
| `input_ids` | `(B, 128)` | padded / truncated |
| `attention_mask` | `(B, 128)` | |
| `eye_tracking_features` | `(B, 5)` | float32 |
| `pooler_output` | `(B, 768)` | BERT/RoBERTa `hidden_size` |
| `eye_tracking_output` | `(B, 16)` | |
| `combined_output` | `(B, 784)` | after concat and dropout |
| `logits` | `(B, 3)` | unnormalized class scores |

`B` is 256 on full SST and 16 on ZuCo.

## Loss and optimization

- Gaze-aware path: `CrossEntropyLoss()(logits.view(-1, 3), labels.view(-1))`.
- Text-only path: the HF model's built-in `outputs.loss` (also
  cross-entropy over 3 labels).
- Optimizer: `Adam` at `5e-5` on **all** parameters (encoder + head).
  No layer-wise decay, no frozen embeddings.

## Metrics

`calculate_metrics` reports accuracy plus weighted precision / recall /
F1 (`sklearn`, `average='weighted'`). Weighted averages matter because
full SST is unbalanced (neutral ≈ 19%).

`model_full_SST.py` picks the checkpoint with the best **validation
accuracy**, despite a comment that says “best F1”. The saved path is
`models/best_{model_type}_model.pth`. After the epoch loop it reloads
that checkpoint and runs the test loader.

`model_ZuCo_SST.py` has no checkpointing. Each fold trains 20 epochs,
evaluates once on the fold's hold-out, and the script prints the mean
of the five fold scores.

## Tokenization

```python
tokenizer(examples['sentence'], padding='max_length', truncation=True, max_length=128)
```

BERT uses `BertTokenizer` / `bert-base-uncased`; RoBERTa uses
`RobertaTokenizer` / `roberta-base`. Movie-review sentences in both
corpora are short; 128 tokens is enough for essentially every row.

## What a “gaze-only” or “toy fusion” run looks like

The original scripts cannot run without `transformers` and a download of
`roberta-base`. Two examples stand in for that when you only want to
see whether the ET vector carries signal:

| Script | Stand-in for |
|---|---|
| `examples/gaze_only_baseline.py` | a linear classifier on the 5-d ET vector (no text) |
| `examples/late_fusion_demo.py` | hashed bag-of-words (64-d) + Linear(5→16) + concat + softmax, trained with numpy SGD |

Those are **not** substitutes for the BERT numbers. They answer a
narrower question: is there any linear (or weakly nonlinear) association
between the committed ET columns and the 3-way label?

## Implementation notes worth knowing

1. `EyeTrackingModel` is copy-pasted in both training files. Changing
   one does not change the other.
2. The gaze linear layer has **no activation**. `late_fusion_demo.py`
   uses ReLU after the 5→16 map so the toy model can actually bend;
   that is a documented difference, not a hidden one.
3. The full-SST test loop currently **overwrites** `all_preds` /
   `all_labels` on every batch instead of extending them, so the printed
   test metrics are the last batch only (256 rows, or fewer). See
   [known-quirks.md](known-quirks.md).
4. `CustomDataset` calls `torch.tensor` on every `__getitem__`. Fine at
   these sizes; not a `TensorDataset`.
