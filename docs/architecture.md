# Model architecture

This note describes the models in `model_ZuCo_SST.py` and `model_full_SST.py`. Both scripts implement the same four variants and the same fusion idea. They differ in data scale, evaluation protocol, and a few training hyperparameters.

## Variants

| `model_type` | Encoder | Gaze input | Head |
| --- | --- | --- | --- |
| `bert` | `bert-base-uncased` | none | Hugging Face sequence-classification head |
| `roberta` | `roberta-base` | none | Hugging Face sequence-classification head |
| `bert_eye_tracking` | `BertModel` | 5 numeric features | custom fusion classifier |
| `roberta_eye_tracking` | `RobertaModel` | 5 numeric features | custom fusion classifier |

The default in both scripts is `roberta_eye_tracking`.

## Text branch

Sentences are tokenized with the matching Hugging Face tokenizer (`BertTokenizer` or `RobertaTokenizer`). Padding and truncation use `max_length=128`. The encoder returns a pooled sentence vector:

- BERT: `[CLS]` pooler output, size `768`
- RoBERTa: the model's `pooler_output`, also size `768`

Text-only runs (`bert`, `roberta`) pass `input_ids`, `attention_mask`, and `labels` into `BertForSequenceClassification` / `RobertaForSequenceClassification` and use the library loss.

## Gaze branch

Five sentence-level eye-tracking features are projected by a linear layer:

```
eye_hidden = Linear(5 -> 16)(gaze_features)
```

Feature names are not identical across datasets:

| Slot | Full SST columns | ZuCo combined columns |
| --- | --- | --- |
| 1 | `nFix` | `nFixations` |
| 2 | `FFD` | `FFD` |
| 3 | `GPT` | `GPT` |
| 4 | `TRT` | `TRT` |
| 5 | `GD` | `GD` |

ZuCo also stores `omissionRate`, `meanPupilSize`, and `SFD`. Those extra columns are **not** fed to `EyeTrackingModel` in the current training scripts.

The projection is an affine map with no activation. It exists so the numeric gaze vector is not concatenated at raw scale against a 768-d transformer embedding.

## Fusion head

```
h = concat(pooler_output, eye_hidden)     # 768 + 16 = 784
h = Dropout(0.1)(h)
logits = Linear(784 -> 3)(h)
```

Loss is `CrossEntropyLoss` over three sentiment classes:

| Label | Meaning |
| --- | ---: |
| `0` | negative |
| `1` | neutral |
| `2` | positive |

This is late fusion: the transformer never sees gaze tokens. Gaze only enters after pooling. That keeps the example small and avoids aligning word-level fixations to subword tokens.

## Why this fusion is a baseline, not a ceiling

Word-level ZuCo files exist under `ZuCo_et_csv_data/word/`. A tighter model could:

1. align each word's gaze vector to the tokenizer's first subword
2. add the projected gaze vector to that token's hidden state
3. let attention mix text and reading behavior before pooling

The current code does none of that. It asks a simpler question: *does a sentence-level gaze summary still help a frozen-width classifier after the text has already been pooled?*

## Training loops

### ZuCo script (`model_ZuCo_SST.py`)

- Data: `ZuCo_SST_data/combined_sst_et_standard.csv` (400 rows)
- Protocol: `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`
- Epochs: 20 per fold
- Batch size: 16
- Optimizer: Adam, `lr=5e-5`
- Reports mean accuracy, precision, recall, and weighted F1 across folds
- Does not write a checkpoint

This protocol is the right default for 400 labeled sentences. A single 80/10/10 split is noisy at that size (the checked-in ZuCo `valid.csv` / `test.csv` files are only 40 rows each).

### Full SST script (`model_full_SST.py`)

- Data: `SST_data/train_full_sst.csv`, `valid_full_sst.csv`, `test_full_sst.csv`
- Protocol: fixed split (approximately 80/10/10)
- Epochs: 5
- Batch size: 256
- Optimizer: Adam, `lr=5e-5`
- Writes `models/best_{model_type}_model.pth` when validation accuracy improves
- Reloads that checkpoint for the test pass

The comment next to the checkpoint says "best F1" but the comparison uses validation accuracy.

## Shapes

For a batch of size `B`:

| Tensor | Shape |
| --- | --- |
| `input_ids` | `[B, 128]` |
| `attention_mask` | `[B, 128]` |
| `eye_tracking_features` | `[B, 5]` |
| `pooler_output` | `[B, 768]` |
| `eye_hidden` | `[B, 16]` |
| `logits` | `[B, 3]` |

## Worked numeric example

The runnable demo in `examples/fusion_forward_demo.py` replaces the transformer with a hashed bag-of-words vector so the same fusion arithmetic can run without downloading weights:

```
text_vec   = hash_embed(sentence)          # 32-d
gaze_vec   = Linear(5 -> 16)(gaze)         # 16-d
hidden     = concat(text_vec, gaze_vec)    # 48-d
logits     = Linear(48 -> 3)(hidden)
```

That script compares three heads on the real ZuCo rows: text only, gaze only, and fused. It is a teaching model, not a substitute for BERT/RoBERTa.

## Implementation notes

- `CustomDataset` stores token ids as Python lists and converts them to tensors in `__getitem__`.
- Gaze features are cast to `float32` tensors.
- Text-only and fusion models share the same `DataLoader`, so gaze columns are always present even when unused.
- Both scripts hard-code `logits.view(-1, 3)` in the fusion loss. Changing `num_labels` without changing that view will break.
- `model_full_SST.py` currently **replaces** `all_preds` / `all_labels` on each test batch instead of extending them. See `docs/notes-and-gotchas.md`.
