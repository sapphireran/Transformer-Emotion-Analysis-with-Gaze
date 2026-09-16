# Model architecture commentary

Both training scripts define the same two pieces:

1. A factory (`get_model`) that returns either a stock Hugging Face
   sequence classifier or the custom `EyeTrackingModel`.
2. A `CustomDataset` that yields `input_ids`, `attention_mask`,
   `labels`, and a float vector of five gaze features.

This page is a walk through `EyeTrackingModel` with the actual tensor
shapes I expect at batch 16 (ZuCo) or batch 256 (full SST).

## Block diagram

```
sentence text
    │
    ▼
tokenizer  →  input_ids [B, 128], attention_mask [B, 128]
    │
    ▼
BertModel or RobertaModel   (frozen? no — everything is trained)
    │
    ▼
pooler_output               [B, 768]
    │
    │   gaze vector         [B, 5]
    │       │
    │       ▼
    │   Linear(5 → 16)
    │       │
    │       ▼
    │   gaze_hidden         [B, 16]
    │
    ▼
concat(pooler, gaze_hidden) [B, 784]
    │
    ▼
Dropout(0.1)
    │
    ▼
Linear(784 → 3)             logits [B, 3]
    │
    ▼
CrossEntropyLoss vs labels  {0, 1, 2}
```

`examples/fusion_sketch.py` reprints a numpy version of the concat and
the parameter count of the two linear layers so the 784 figure is not
just a comment.

## Why late fusion

The gaze tower never sees tokens. The text tower never sees milliseconds.
They meet after both have been reduced to one vector per sentence. That
is **late fusion**.

Late fusion is the honest match for the data I actually have on Track A:
one 5-D sentence summary, one SST label. It is also the weakest possible
test of the scientific story in `01-research-notes.md` (that *which*
word was re-read matters). If late fusion already helps, a token-aligned
model might help more. If late fusion does nothing, I should not jump
to a 128-step gaze sequence without first checking whether the 5-D
vector is just length in disguise.

## The gaze tower is tiny on purpose

```
Linear(5 → 16)   = 5*16 + 16 = 96 parameters
Linear(784 → 3)  = 784*3 + 3 = 2,355 parameters
```

RoBERTa-base is ~125M. The gaze path cannot overpower the text path by
parameter count. If the fusion model beats text-only, it is because the
16 numbers are *informative*, not because we added capacity. That is
the point. A 512-D gaze MLP would muddy the comparison.

There is **no activation** between `Linear(5 → 16)` and the concat.
The 16-D vector is an affine remix of the five z-scores. Dropout sits
*after* the concat, so it can drop text dimensions and gaze dimensions
together. I would rather dropout only the 16-D tower (so the text
baseline stays intact inside the same network). Noted as a next-pass
change; not edited here.

## `pooler_output` on BERT vs RoBERTa

```python
base_output = self.base_model(input_ids=input_ids, attention_mask=attention_mask)
pooled_output = base_output.pooler_output
```

For `BertModel`, `pooler_output` is `tanh(W · h_[CLS] + b)`, pretrained
with next-sentence prediction. For `RobertaModel`, Hugging Face still
exposes a `pooler` (dense + tanh on the first token `<s>`), but RoBERTa
was not pretrained with NSP. People often use `last_hidden_state[:, 0, :]`
or a mean of non-padding tokens for RoBERTa classification.

This repo uses the stock pooler for both. I am documenting that choice,
not defending it. A fair RoBERTa ablation would try:

- `pooler_output` (current)
- `last_hidden_state[:, 0, :]` without the extra tanh
- attention-masked mean pool

Any of those changes belongs in both the text-only and the fusion
run, or the comparison is garbage.

## Text-only branch

When `model_type` is `'bert'` or `'roberta'`, `get_model` returns
`BertForSequenceClassification` / `RobertaForSequenceClassification`
with `num_labels=3`. The training loop then uses the HF loss:

```python
outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
loss = outputs.loss
```

Gaze is still *loaded* into `CustomDataset` and still moved… no, it is
not moved, because that branch never reads `batch['eye_tracking_features']`.
The CSV columns are wasted I/O. Harmless.

When `model_type` is `'*_eye_tracking'`, the loop builds
`CrossEntropyLoss()` **inside the batch loop**. That constructs a new
criterion object every step. Functionally fine, stylistically noisy.
The `view(-1, 3)` matches `num_labels`.

## Tokenization

```python
tokenizer(examples['sentence'], padding='max_length', truncation=True, max_length=128)
```

- BERT: `bert-base-uncased` WordPiece, lowercased.
- RoBERTa: `roberta-base` BPE, case-preserving.

The same sentence therefore has different `input_ids` in the two
`model_type` families. Gaze does not change. A BERT-vs-RoBERTa gap is
not a gaze finding.

`CustomDataset` stores `input_ids` and `attention_mask` as Python
lists and wraps them in `torch.tensor` per item. At batch 256 that is
a lot of tiny allocations. Fine for 5 epochs; I would pad in a collate
function if this were a longer run.

## Device and precision

```python
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
```

No AMP, no gradient accumulation, no `max_grad_norm`. Full SST at
batch 256 × 128 tokens × RoBERTa is the setting that actually wants a
GPU. ZuCo at batch 16 is runnable on CPU, just slow (20 epochs × 5
folds × ~320 sentences).

## What is *not* in the graph

- No layer-wise decay.
- No freezing of embeddings or of the bottom *k* layers.
- No auxiliary loss on gaze (we do not predict gaze; we consume it).
- No subject ID embedding. Averaging already destroyed the subject
  axis on Track A.
- No length feature. Length is only present implicitly (transformers
  see more tokens; gaze correlates with length).

If I ever add a control, the first one is concatenating
`log(sent_len)` instead of gaze. If *that* matches the fusion gain,
the five eye-tracking names were a long way to write "this review is
long / hard."

## Forward pass, one ZuCo row

Sentence 0, z-scored gaze (from `combined_sst_et_standard.csv`):

```
nFixations    1.5209
FFD          -0.0784
GPT           1.5875
TRT           1.2480
GD            0.0758
```

After `Linear(5 → 16)` those become a 16-vector whose first few
coordinates depend on the random init (or on a trained checkpoint we
do not ship). Concatenated with a 768-D RoBERTa pooler, the classifier
sees "this sentence's language-model summary, plus a slightly
above-average fixation count and a clearly above-average go-past
time." That is exactly the "readers lingered / went back" signal the
glossary called out for sentence 0's late words (`beyond`, `decency`).

Whether those 16 numbers move the argmax from class 1 (neutral, the
true label) is the empirical question the training scripts exist to
answer. This clone does not contain a trained `state_dict`, so I do
not pretend to know the answer from the repo alone.
