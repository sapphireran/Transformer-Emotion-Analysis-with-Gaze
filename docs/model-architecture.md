# Model architecture

Both `model_ZuCo_SST.py` and `model_full_SST.py` define the same two
building blocks: `EyeTrackingModel` and `CustomDataset`. They differ in
**how they loop** (k-fold vs train/valid/test) and **which CSV columns**
they bind to gaze.

## Tokenization

```python
if model_type.startswith('bert'):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
else:
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
```

Sentences are padded / truncated to **128** WordPiece / BPE tokens. The
Hugging Face `datasets.Dataset.map` tokenizer is used only as a convenience
to get `input_ids` and `attention_mask` back into a pandas frame.

RoBERTa has no `token_type_ids` in this pipeline; they are not stored on
`CustomDataset`.

## `CustomDataset`

Each item is:

| Key | Tensor |
| --- | --- |
| `input_ids` | `LongTensor [128]` |
| `attention_mask` | `LongTensor [128]` |
| `labels` | scalar label `0/1/2` |
| `eye_tracking_features` | `FloatTensor [5]` |

Gaze is **not** aligned to tokens. There is no per-word feature in the
forward pass, even though word-level CSVs exist. The five numbers are a
sentence embedding of gaze, analogous to a second `[CLS]`.

## Text-only heads

`get_model` for `bert` / `roberta`:

```python
BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=3)
RobertaForSequenceClassification.from_pretrained('roberta-base', num_labels=3)
```

Forward:

```python
outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
loss = outputs.loss
logits = outputs.logits
```

The pretrained classification head is a linear layer on the pooled
representation (BERT: `tanh` pooler; RoBERTa: same pooler module in the
`transformers` implementation).

## Fusion head (`EyeTrackingModel`)

```python
class EyeTrackingModel(nn.Module):
    def __init__(self, base_model, num_eye_tracking_features, num_labels):
        self.base_model = base_model.from_pretrained(...)  # BertModel or RobertaModel
        self.eye_tracking_layer = nn.Linear(num_eye_tracking_features, hidden_layer_size)  # 5 → 16
        self.classifier = nn.Linear(self.base_model.config.hidden_size + hidden_layer_size, num_labels)  # 784 → 3
        self.dropout = nn.Dropout(0.1)
```

Forward:

1. `base_output = encoder(input_ids, attention_mask)`
2. `pooled = base_output.pooler_output`  # (B, 768)
3. `gaze_h = Linear_5_16(gaze)`          # (B, 16), **no activation**
4. `h = dropout(concat(pooled, gaze_h))` # (B, 784)
5. `logits = Linear_784_3(h)`            # (B, 3)

Loss for fusion:

```python
CrossEntropyLoss()(logits.view(-1, 3), labels.view(-1))
```

There is no temperature, label smoothing, or class-weighting.

### Design choices that are easy to miss

- **No nonlinearity on the gaze projection.** `eye_tracking_layer` is a
  plain affine map. A ReLU / GELU would make the 16-D branch strictly
  richer; as written it is a learned rescaling + rotation of the five
  features, then a linear mix with the 768-D text vector.
- **`pooler_output`, not mean pooling.** BERT's pooler is `tanh(W h_CLS)`.
  RoBERTa still exposes `pooler_output` in `RobertaModel`. Mean-over-tokens
  is not used.
- **Gaze never sees attention.** You cannot ask “which token got the long
  GPT” inside this head. Word-level files are for the **predictor** and for
  analysis, not for this classifier.
- **Dropout only after concat.** The gaze affine layer is not dropped on
  its own.
- **`hidden_layer_size = 16`** is tiny vs 768. Gaze can only move the
  decision if the classifier puts weight on those 16 dimensions. If fusion
  matches text-only, it may be capacity, not “gaze is useless.”

## Shapes

Assume batch `B`, `bert-base` / `roberta-base`:

```
input_ids          (B, 128)
attention_mask     (B, 128)
encoder hidden     (B, 128, 768)
pooler_output      (B, 768)
gaze               (B, 5)
gaze_h             (B, 16)
concat             (B, 784)
logits             (B, 3)
```

`examples/toy_fusion_forward.py` rebuilds this with numpy / random weights
so you can print shapes without downloading BERT.

## Optimization

Both scripts use **Adam** (not AdamW) at `lr = 5e-5` on **all** parameters,
including the pretrained encoder. There is no layer-wise decay, no freeze
of the bottom blocks, no warmup, no gradient clipping.

Implications:

- The 5→16 gaze layer and the 784→3 classifier learn at the same nominal
  LR as BERT, but their gradients live on a different scale. A slightly
  higher LR on the new layers is a common tweak not implemented here.
- Full fine-tuning on 400 sentences (Track A) overfits easily; 20 epochs
  × 5 folds is a lot of passes over the same reviews.

## Metrics

`sklearn.metrics` with `average='weighted'` for precision, recall, and F1,
plus accuracy. Weighted F1 is accuracy-like when classes are balanced; on
SST-3 they are not, so **per-class** recall is worth printing (see
`examples/metrics_example.py`).

Track A reports **mean over 5 folds**. Track B reports valid metrics every
epoch and test metrics once, after reloading `models/best_{model_type}_model.pth`
selected by **validation accuracy** (the log line still says “F1” — naming
bug, see [known-pitfalls.md](known-pitfalls.md)).

## What a stronger fusion might look like (not implemented)

Documented as future personal notes, not as code in this pass:

1. **Word-aligned gaze:** project per-token `(nFix, FFD, …)` and add to
   token embeddings before the encoder (early fusion).
2. **Gated concat:** `h = text + gate(text) * W_gaze(gaze)` so the model
   can ignore noisy predicted gaze.
3. **Auxiliary gaze loss:** predict the five features from `h_CLS` and
   train text-only models with that auxiliary head (ZuCo as multi-task).
4. **Class-weighted CE** for SST-3 imbalance.

`examples/gaze_only_baseline.py` is the opposite extreme: **no** transformer,
just a linear softmax on the five z-scored features. If that baseline is
near chance, fusion should not be expected to help much either.
