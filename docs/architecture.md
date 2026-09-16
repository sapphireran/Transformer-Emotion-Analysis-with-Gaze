# Architecture

Both training scripts define the same `EyeTrackingModel` and the same
`get_model` switch.

## Variants

| `model_type` | Encoder | Gaze | Classifier |
| --- | --- | --- | --- |
| `bert` | `BertForSequenceClassification` | ignored | Hugging Face head |
| `roberta` | `RobertaForSequenceClassification` | ignored | Hugging Face head |
| `bert_eye_tracking` | `BertModel` pooler | `Linear(5, 16)` | `Linear(784, 3)` |
| `roberta_eye_tracking` | `RobertaModel` pooler | `Linear(5, 16)` | `Linear(784, 3)` |

Default in both files is `roberta_eye_tracking`. Tokenizers follow the prefix:
`bert-base-uncased` vs `roberta-base`. Max length is 128, which is enough for
these review snippets (ZuCo max 42 words, SST max about 51 predicted tokens).

## Late fusion

```
                     input_ids + attention_mask
                                │
                                ▼
                     BertModel / RobertaModel
                                │
                          pooler_output
                             (B, 768)
                                │
                                │              eye_tracking_features
                                │                       (B, 5)
                                │                        │
                                │                        ▼
                                │                 Linear(5 → 16)
                                │                    (B, 16)
                                │                        │
                                └──────────┬─────────────┘
                                           ▼
                                    concat (B, 784)
                                           │
                                      Dropout(0.1)
                                           │
                                      Linear(784 → 3)
                                           │
                                        logits
```

The gaze branch has no nonlinearity. It is a learned 5→16 stretch so the
classifier can reweight channels; it cannot build interactions among gaze
features except through the final linear layer (which also sees the 768-d
text vector).

Dropout is applied **after** concatenation, so it can drop text or gaze
coordinates. There is no separate gaze dropout.

## Loss and optimization

- Fused models: `CrossEntropyLoss` on `logits.view(-1, 3)` vs `labels`
- Text-only models: the Hugging Face sequence-classification loss
- Optimizer: `Adam` (not AdamW), `lr = 5e-5`
- No weight decay, no learning-rate schedule, no warmup
- No gradient clipping

## Why a numpy stand-in exists

`examples/gaze_emotion_examples/fusion.py` mirrors the linear part:

```python
gaze_hidden = gaze @ W_gaze + b_gaze          # (B, 5) → (B, 16)
logits      = concat(text, gaze_hidden) @ W_cls + b_cls
```

Weights there are random. The class exists so docs and tests can lock
shapes without downloading `roberta-base`. The same file also fits sklearn
logistic baselines (TF-IDF ± gaze) for a GPU-free comparison on the 400
ZuCo rows.

## What this architecture cannot do

- It does **not** attend over word-level gaze. A 22-word ZuCo sentence
  becomes one 5-d vector before it ever meets RoBERTa.
- It does **not** gate gaze by transformer uncertainty.
- `pooler_output` for BERT is `tanh(W cls_token)`. For RoBERTa, Hugging Face
  still exposes a pooler when `add_pooling_layer=True` (the default), but it
  is a different trained projection than BERT's NSP pooler.
- Token-level features in `word_averages_v2.csv` and
  `prediction_test_v2.csv` are unused at train time. They are there for
  analysis and for the gaze predictor, not for the sentiment model.

A natural next architecture — not implemented here — would add a gaze
vector to each token embedding, or use gaze as a per-token bias on
attention scores.
