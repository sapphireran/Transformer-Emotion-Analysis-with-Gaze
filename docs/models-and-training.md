# Models and training

The two training scripts share one idea and almost the same code. This page
is a reading guide, not a second implementation.

## Architectures

`get_model(model_type, ...)` understands four strings:

| `model_type` | Class | Gaze used? |
| --- | --- | --- |
| `bert` | `BertForSequenceClassification` | no |
| `roberta` | `RobertaForSequenceClassification` | no |
| `bert_eye_tracking` | `EyeTrackingModel(BertModel, ...)` | yes |
| `roberta_eye_tracking` | `EyeTrackingModel(RobertaModel, ...)` | yes |

Both scripts default to `roberta_eye_tracking`.

### EyeTrackingModel

```
base_model = BertModel / RobertaModel.from_pretrained(...)
eye_tracking_layer = Linear(5, 16)
classifier = Linear(hidden_size + 16, 3)
dropout = Dropout(0.1)

forward(input_ids, attention_mask, eye_tracking_features):
    pooled = base_model(...).pooler_output          # [B, 768]
    gaze   = eye_tracking_layer(eye_tracking_features)  # [B, 16]
    logits = classifier(dropout(cat(pooled, gaze)))     # [B, 3]
```

Notes:

- **Pooler output**, not the CLS hidden state before the pooler. For BERT
  that is `tanh(W cls)`. RoBERTa still exposes a pooler in the Hugging Face
  port this script uses.
- Gaze is **sentence-level**. There is no token-wise feature alignment.
- The encoder is **not** frozen. Adam sees every parameter.
- Loss for fusion models is `CrossEntropyLoss` on the raw logits.
- Text-only models use the HF head's built-in loss.

`examples/fusion_architecture_demo.py` reprints these shapes with a tiny
NumPy stand-in (`GazeFusionClassifier`).

## Tokenization

```
tokenizer(sentence, padding='max_length', truncation=True, max_length=128)
```

BERT uses `bert-base-uncased`, RoBERTa uses `roberta-base`. Long reviews
are truncated; ZuCo sentences almost always fit.

## ZuCo track (`model_ZuCo_SST.py`)

| Setting | Value |
| --- | --- |
| Data | `ZuCo_SST_data/combined_sst_et_standard.csv` |
| Protocol | `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)` |
| Epochs | 20 |
| Batch size | 16 (hard-coded in the fold loop; the `batch_size` config is unused) |
| Optimizer | Adam `5e-5` |
| Checkpointing | none — last epoch of each fold is evaluated |
| Reported numbers | mean accuracy / P / R / F1 over the 5 folds |

Because the fold loop rebuilds the model each time, you get five independent
fine-tunes. There is no nested validation split inside a fold.

## Full-SST track (`model_full_SST.py`)

| Setting | Value |
| --- | --- |
| Data | `SST_data/{train,valid,test}_full_sst.csv` |
| Epochs | 5 |
| Batch size | 256 |
| Optimizer | Adam `5e-5` |
| Checkpointing | `models/best_{model_type}_model.pth` on **validation accuracy** |
| Test | reload the best checkpoint, score the test loader once |

The comment next to `best_val_acc` says "best F1" but the comparison uses
accuracy. The save message also prints the accuracy under the name F1.
See [known issues](known-issues.md).

The test loop currently **assigns** `all_preds = preds.cpu().numpy()` instead
of extending the list. Metrics then reflect **only the last batch**. Fix
that before quoting a test score.

## Metrics

Both scripts call sklearn with `average='weighted'`:

- accuracy
- precision
- recall
- F1

`gaze_emotion.metrics.weighted_scores` reproduces those four numbers so the
examples do not import sklearn at runtime. `tests/test_metrics.py` compares
the two implementations.

On full SST, always-positive accuracy is ~0.41. On ZuCo-SST combined data it
is 140/400 = 0.35. Weighted F1 is the number to compare across model types
because the classes are not even.

## Suggested comparison grid

For a write-up, run the same seed and data for:

1. `roberta` vs `roberta_eye_tracking` on ZuCo 5-fold
2. `roberta` vs `roberta_eye_tracking` on full SST hold-out
3. Optional: `bert` / `bert_eye_tracking` as a lower-capacity control

Keep the five gaze channels identical. If you add SFD or pupil size, treat
that as a new ablation, not a drop-in replacement.

## What this is not

- Not a token-level gaze transformer (no per-BPE feature).
- Not multimodal EEG (ZuCo EEG is unused).
- Not a gaze *predictor* — `gaze_prediction/data` is input, not trained here.
- Not production inference. There is no `predict.py` in the original tree.
