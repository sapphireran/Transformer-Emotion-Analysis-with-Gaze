# Model architecture

Both training scripts define the same two families:

- **Text-only:** Hugging Face `BertForSequenceClassification` or
  `RobertaForSequenceClassification`, `num_labels=3`.
- **Late fusion:** custom `EyeTrackingModel` wrapping `BertModel` or
  `RobertaModel`.

`model_type` is a string at the top of the file:

```
bert | roberta | bert_eye_tracking | roberta_eye_tracking
```

The checked-in default is `roberta_eye_tracking`.

## Late-fusion diagram

```
sentence tokens ──► BERT / RoBERTa ──► pooler_output     (B, 768)
                                              │
gaze (B, 5) ──► Linear(5, 16) ──► (B, 16)     │
                                              ▼
                                    concat → (B, 784)
                                              │
                                         Dropout(0.1)
                                              │
                                      Linear(784, 3) ──► logits
```

There is **no** nonlinearity on the gaze projection (no ReLU). The
only nonlinearity on that path is whatever the classifier + softmax
(in the loss) provide. Dropout is applied **after** concatenation, so
it can drop both text and gaze coordinates.

## Code map (`EyeTrackingModel`)

```python
self.base_model = base_model.from_pretrained(
    'bert-base-uncased' if base_model == BertModel else 'roberta-base'
)
self.eye_tracking_layer = nn.Linear(num_eye_tracking_features, hidden_layer_size)
self.classifier = nn.Linear(
    self.base_model.config.hidden_size + hidden_layer_size,  # 768 + 16
    num_labels,
)
self.dropout = nn.Dropout(0.1)
```

Forward:

1. `base_output = self.base_model(input_ids, attention_mask)`
2. `pooled_output = base_output.pooler_output`
3. `eye_tracking_output = self.eye_tracking_layer(eye_tracking_features)`
4. `torch.cat(..., dim=1)` → dropout → classifier

`pooler_output` is the dense-tanh projection of the first token state
(BERT CLS, RoBERTa equivalent). It is **not** a mean over tokens, so
a long review and a short review are the same width, and gaze never
interacts with individual token states.

## Shapes and hyperparameters

| Name | Default | Where |
| --- | ---: | --- |
| `num_eye_tracking_features` | 5 | both scripts |
| `hidden_layer_size` | 16 | both |
| `num_labels` | 3 | both |
| `max_length` | 128 | tokenizer |
| dropout | 0.1 | fusion only (HF heads use their own) |

BERT-base / RoBERTa-base hidden size is 768. Changing to a large
encoder requires no code change in the classifier line because it
reads `config.hidden_size`, but `from_pretrained` is hard-coded to
the base checkpoints.

## Loss

- Text-only: whatever the HF model returns as `outputs.loss`
  (standard token classification-style cross-entropy on 3 logits).
- Fusion: `CrossEntropyLoss()(logits.view(-1, 3), labels.view(-1))`

No class weights. On full SST, class 1 is ~19% of rows, so the
unweighted loss favors negative/positive. The ZuCo 400 is close to
balanced; the issue is milder there.

Labels are stored as integers in the CSV and become
`torch.tensor(self.labels[idx])` without an explicit `dtype=torch.long`.
PyTorch usually casts NumPy integer arrays to `int64`; if a CSV is
ever read as floats you will get a CE loss error. The schema checker
flags non-integer labels.

## Optimizer

Plain `torch.optim.Adam` on **all** parameters, `lr=5e-5`.

There is no:

- layer-wise decay
- freeze-then-unfreeze
- different LR for `eye_tracking_layer` vs the encoder
- weight decay (AdamW)
- learning-rate schedule
- gradient clipping

That is acceptable for a first personal experiment and easy to
overfit on 320-ish ZuCo training rows per fold (20 epochs).

## What is *not* in the architecture

- **Word-level gaze → token alignment.** Would need a tokenizer that
  emits the same words as ZuCo, plus a `(B, T, 5)` tensor and either
  addition to token embeddings or a gated fusion.
- **Subject ID or random effect.** Readers were averaged away.
- **EEG.** ZuCo has it; these scripts never load it.
- **Multi-task** (sentiment + readability).
- **Attention visualization** over gaze.

`examples/06_toy_late_fusion.py` replaces the 768-d encoder with
either nothing (gaze-only) or a handful of cheap text statistics
(length, punctuation, a tiny polarity lexicon). That is **not** a
substitute for BERT; it exists so you can see whether the 5-d gaze
vector has any linear signal before you pay for fine-tuning.

## Parameter count (order of magnitude)

| Piece | Params |
| --- | ---: |
| BERT/RoBERTa base | ~110M / ~125M |
| `Linear(5, 16)` + bias | 96 |
| `Linear(784, 3)` + bias | 2,355 |
| HF classification head `Linear(768, 3)` | 2,307 |

The gaze path is a rounding error. If fusion helps, it is because
those 16 numbers are **informative**, not because you added capacity.

## Device

`cuda` if `torch.cuda.is_available()` else `cpu`. No `DataParallel`,
no mixed precision. Full-SST batch 256 × 128 tokens will OOM on
small GPUs; drop `batch_size` in the script header.
