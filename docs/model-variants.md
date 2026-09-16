# Model variants

`get_model()` in both trainers is a four-way switch. Only the
`*_eye_tracking` variants use gaze.

## Plain text classifiers

```python
BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=3)
RobertaForSequenceClassification.from_pretrained("roberta-base", num_labels=3)
```

These are the controls: same tokenizer, same 3-class head, no concat. The
`CustomDataset` still ships `eye_tracking_features` in every batch; the loop
just never reads that key.

Use these when asking “did gaze do anything, or did RoBERTa already know the
review?”

## Concat fusion (`EyeTrackingModel`)

```python
self.base_model = BertModel / RobertaModel   # no classification head
self.eye_tracking_layer = Linear(5, 16)
self.classifier = Linear(768 + 16, 3)
self.dropout = Dropout(0.1)
```

Forward:

1. `base_output = encoder(input_ids, attention_mask)`
2. `pooled = base_output.pooler_output`
3. `et_h = Linear(et_features)`
4. `logits = classifier(dropout(cat(pooled, et_h)))`

There is no gate, no attention over words, no per-token gaze alignment.
Sentence-level gaze is a single 5-d sidecar.

That is a deliberate personal-project simplification. Word-level files exist
(`ZuCo_et_csv_data/word/`, `gaze_prediction/data/prediction_test_v2.csv`) but
nothing in the trainers attends over them.

## Hyperparameters that actually sit in the scripts

| Knob | ZuCo script | Full SST script |
| --- | ---: | ---: |
| `num_eye_tracking_features` | 5 | 5 |
| `hidden_layer_size` | 16 | 16 |
| `num_labels` | 3 | 3 |
| `num_epochs` | 20 | 5 |
| `learning_rate` | 5e-5 | 5e-5 |
| `batch_size` | 16 | 256 |
| default `model_type` | `roberta_eye_tracking` | `roberta_eye_tracking` |

Switching `model_type` is a one-line edit at the top of the file. There is no
CLI.

## What I compare when I rerun things

1. `roberta` vs `roberta_eye_tracking` on the same split seed.
2. Gaze-only logistic regression (`examples/05_gaze_only_baseline.py`) as a
   floor that does not get to read the sentence.
3. Majority class (`tea_gaze.baselines.majority_cv`) as a dumber floor.
4. The toy fusion in `examples/04_toy_fusion_forward.py` only to see that
   concat-and-classify is wired the way I think it is. It is not a paper
   result.

## Things these variants do not do

- Per-word gaze aligned to BPE tokens
- EEG (ZuCo recorded it; this repo never loads it)
- Binary positive/negative (neutral stays a real class)
- Multi-task reading-time prediction
- Layer-wise fusion or cross-attention

If I add those later, they belong in new scripts under `examples/` or a new
trainer, not as silent edits to the historical `model_*_SST.py` files.
