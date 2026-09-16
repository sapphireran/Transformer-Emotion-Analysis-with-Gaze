# Architecture

The interesting model is not a custom transformer. It is a stock BERT or
RoBERTa encoder plus a **late-fusion** sentiment head that sees a 5-D gaze
vector.

Both `model_ZuCo_SST.py` and `model_full_SST.py` define the same class
(duplicated, not imported):

```python
class EyeTrackingModel(nn.Module):
    def __init__(self, base_model, num_eye_tracking_features, num_labels):
        self.base_model = base_model.from_pretrained(...)
        self.eye_tracking_layer = nn.Linear(num_eye_tracking_features, hidden_layer_size)
        self.classifier = nn.Linear(self.base_model.config.hidden_size + hidden_layer_size, num_labels)
        self.dropout = nn.Dropout(0.1)

    def forward(self, input_ids, attention_mask, eye_tracking_features):
        pooled = self.base_model(...).pooler_output
        gaze = self.eye_tracking_layer(eye_tracking_features)
        return self.classifier(self.dropout(torch.cat((pooled, gaze), dim=1)))
```

## Dimensions

| Piece | ZuCo script | Full-SST script |
| --- | ---: | ---: |
| Gaze input | 5 | 5 |
| Gaze hidden (`hidden_layer_size`) | 16 | 16 |
| Encoder hidden | 768 (`bert-base` / `roberta-base`) | 768 |
| Concat | 784 | 784 |
| Labels | 3 | 3 |
| Dropout | 0.1 | 0.1 |

`nn.Linear(5, 16)` is `y = x @ W.T + b` with `W` shaped `(16, 5)`. The numpy
clone in `examples/fusion_forward.py` uses that layout so a random batch has
the same shapes as a real forward.

## Four `model_type` values

`get_model()`:

| `model_type` | Object | Gaze used? |
| --- | --- | --- |
| `bert` | `BertForSequenceClassification` | no |
| `roberta` | `RobertaForSequenceClassification` | no |
| `bert_eye_tracking` | `EyeTrackingModel(BertModel, …)` | yes |
| `roberta_eye_tracking` | `EyeTrackingModel(RobertaModel, …)` | yes |

Text-only runs still *load* gaze columns into `CustomDataset`. They just never
pass `eye_tracking_features` into the module. That makes ablations easy: flip
the string, keep the CSV.

Default in both scripts is `'roberta_eye_tracking'`.

## Tokenizer and max length

```python
tokenizer(..., padding='max_length', truncation=True, max_length=128)
```

BERT uses `bert-base-uncased`, RoBERTa uses `roberta-base`. The switch is
`model_type.startswith('bert')`.

128 subwords is enough for this data: SST sentences here top out around 56
whitespace tokens, ZuCo around 43. Long reviews are not in these tables.

## Pooler, not token-level gaze

`base_output.pooler_output` is the tanh-projected `[CLS]` vector (BERT) or
RoBERTa's pooler. Gaze never enters the encoder layers and never attends to
individual tokens.

Consequences:

- You cannot ask “which word's FFD mattered?” without adding a new head.
- Predicted sentence-level gaze on the full SST is the same *kind* of object
  as ZuCo's subject-averaged sentence gaze, so the same module works on both.
- A word-level predictor (`prediction_test_v2.csv`) has to be pooled *before*
  training. That pooling is not in the training scripts.

## Loss and metrics

Fusion path:

```python
CrossEntropyLoss()(logits.view(-1, 3), labels.view(-1))
```

Text-only path uses the Hugging Face sequence-classification loss
(`outputs.loss`), which is also cross-entropy over 3 classes.

Reported metrics (sklearn, `average='weighted'`):

- accuracy
- precision
- recall
- F1

`model_full_SST.py` checkpoints on **accuracy** even though the log line says
“F1”. Details in [known-issues.md](known-issues.md).

## Optimiser

Plain `torch.optim.Adam` at `5e-5` on **all** parameters, including the
pretrained encoder. There is no layer-wise decay, no frozen bottom layers, and
no Hugging Face `AdamW`. For 400 ZuCo sentences that is aggressive; the 20-epoch
CV loop can overfit a fold if you are not watching the unused hold-out files.

## What this is not

- Not a two-stream transformer with gaze embeddings per token.
- Not EEG fusion (ZuCo also records EEG; this repo never loads it).
- Not a gaze *predictor*. The files under `gaze_prediction/data/` are
  **outputs** (or converted gold) for a model that lives elsewhere.

If you add token-level fusion later, you will need a word-to-subword aligner.
`sst_et_test.csv` already has `(sentence_id, word_id, word)` which is the
right place to start; BPE alignment is the missing piece.

## Tiny forward you can run

```bash
python3 examples/fusion_forward.py
```

That script builds random `pooled` `(B, 768)` and `gaze` `(B, 5)` tensors,
applies the Linear → concat → Linear graph, and checks output shape
`(B, 3)`. It also compares a text-only head (`Linear(768, 3)`) on the same
`pooled` vector so you can see that the extra 16 units actually change the
logits.
