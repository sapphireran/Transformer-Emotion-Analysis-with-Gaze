# Models

Two training scripts, four `model_type` strings, one fusion idea.

| `model_type` | Encoder | Gaze branch | Hugging Face class |
| --- | --- | --- | --- |
| `bert` | `bert-base-uncased` | no | `BertForSequenceClassification` |
| `roberta` | `roberta-base` | no | `RobertaForSequenceClassification` |
| `bert_eye_tracking` | `BertModel` | yes | custom `EyeTrackingModel` |
| `roberta_eye_tracking` | `RobertaModel` | yes | custom `EyeTrackingModel` |

The default in both scripts is `roberta_eye_tracking`.

## Tokenization

```python
tokenizer(examples['sentence'], padding='max_length', truncation=True, max_length=128)
```

BERT uses `BertTokenizer.from_pretrained('bert-base-uncased')`.
RoBERTa uses `RobertaTokenizer.from_pretrained('roberta-base')`. The
choice is `model_type.startswith('bert')`. Sentences longer than 128
WordPieces / BPE tokens are truncated; SST review sentences are usually
well under that.

The Hugging Face `datasets.Dataset.map` path is only used as a
tokenizer. The training loop itself is a plain `torch.utils.data.Dataset`
(`CustomDataset`) that stores:

- `input_ids`, `attention_mask` (lists of ints)
- `labels` (`sentiment_label`)
- `eye_tracking_features` (float32 vector of length 5)

Text-only modes still *store* the gaze vector; they just never read it.

## `EyeTrackingModel`

Copied almost verbatim in `model_full_SST.py` and `model_ZuCo_SST.py`:

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
        pooled_output = base_output.pooler_output          # [B, 768]
        eye_tracking_output = self.eye_tracking_layer(eye_tracking_features)  # [B, 16]
        combined_output = torch.cat((pooled_output, eye_tracking_output), dim=1)  # [B, 784]
        combined_output = self.dropout(combined_output)
        return self.classifier(combined_output)            # [B, 3]
```

`hidden_layer_size = 16` in both scripts. There is **no** nonlinearity
on the gaze projection — it is a linear change of basis. Dropout is
applied after concat, not inside the encoder.

`pooler_output` for BERT is `tanh(W · h_[CLS])`. For RoBERTa the
pooler is trained during pretraining of `RobertaModel` the same way;
it is not the same as "mean of last hidden states."

## Loss and optimizer

- Text-only: Hugging Face's built-in loss (`outputs.loss`) —
  `CrossEntropyLoss` over `num_labels=3`.
- Fusion: `CrossEntropyLoss()(logits.view(-1, 3), labels.view(-1))`.
  The `3` is hard-coded, not `num_labels`.
- Optimizer: `torch.optim.Adam` at `5e-5` on **all** parameters,
  including the full encoder. No layer-wise decay, no frozen
  embeddings, no warmup.

## Metrics

```python
accuracy_score
precision_score(..., average='weighted')
recall_score(..., average='weighted')
f1_score(..., average='weighted')
```

Weighted averages matter because full SST is unbalanced (neutral is
~19%). ZuCo 400 is close to balanced, so weighted vs macro will not
move as much.

`model_full_SST.py` checkpoints on **validation accuracy**, even though
the log line says "best F1". `model_ZuCo_SST.py` does not checkpoint;
it reports the mean of the five fold metrics after the last epoch.

## What is *not* in the model

- No token-level gaze (the word-level CSVs never enter the classifier).
- No cross-attention from gaze to hidden states — concat only.
- No subject identity embedding (ZuCo features are already averaged).
- No class weights for the minority neutral class on full SST.

## Shapes cheat sheet

| Tensor | Shape |
| --- | --- |
| `input_ids` | `[B, 128]` |
| `attention_mask` | `[B, 128]` |
| `eye_tracking_features` | `[B, 5]` |
| encoder `pooler_output` | `[B, 768]` |
| gaze hidden | `[B, 16]` |
| concat | `[B, 784]` |
| logits | `[B, 3]` |

`B` is 256 on full SST and 16 on ZuCo.

A numpy reimplementation of the fusion arithmetic (random weights, no
pretrained encoder) is `examples/06_feature_fusion_walkthrough.py`.
Use it to sanity-check a new feature width before editing the training
scripts.

## Duplication note

`EyeTrackingModel`, `CustomDataset`, `calculate_metrics`, and
`get_model` are copy-pasted across the two training files. If you
change one, change the other — or, next time this repo is refactored,
lift them into a shared module. The examples library does **not**
import those scripts (they load tokenizers at import time).
