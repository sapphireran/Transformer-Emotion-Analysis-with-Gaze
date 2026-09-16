# Model architecture

Both `model_ZuCo_SST.py` and `model_full_SST.py` define the same two families. The classes are copy-pasted, not imported.

## Families

| `model_type` | Encoder | Gaze | Classifier |
| --- | --- | --- | --- |
| `bert` | `BertForSequenceClassification` (`bert-base-uncased`, 3 labels) | none | HF head |
| `roberta` | `RobertaForSequenceClassification` (`roberta-base`, 3 labels) | none | HF head |
| `bert_eye_tracking` | `BertModel` pooler | 5 → 16 linear | `Linear(768+16, 3)` |
| `roberta_eye_tracking` | `RobertaModel` pooler | 5 → 16 linear | `Linear(768+16, 3)` |

Default in both scripts: `model_type = 'roberta_eye_tracking'`.

## EyeTrackingModel

Constants (module-level, not constructor args except where noted):

| Name | Value | Where it applies |
| --- | --- | --- |
| `num_eye_tracking_features` | 5 | constructor |
| `hidden_layer_size` | 16 | gaze projection width |
| `num_labels` | 3 | constructor |
| dropout | 0.1 | after concat |
| encoder hidden | 768 | BERT-base / RoBERTa-base |

Forward pass, in order:

```
input_ids, attention_mask
        |
        v
  BertModel / RobertaModel
        |
        v
  pooler_output          eye_tracking_features (B, 5)
   (B, 768)                       |
        |                         v
        |                  Linear(5, 16) + (no activation)
        |                         |
        +---------- concat -------+
                    (B, 784)
                      |
                   Dropout(0.1)
                      |
                 Linear(784, 3)
                      |
                    logits
```

There is **no nonlinearity** on the gaze projection. The 16-d vector is an affine remix of the five channels. The only nonlinearity in the gaze path is whatever the classifier and the encoder already have.

Loss for fusion types is `CrossEntropyLoss` on `logits.view(-1, 3)` vs `labels.view(-1)`. Text-only types use the loss returned by the HF model (also CE).

## Why pooler_output

`pooler_output` is the `[CLS]` hidden state after a tanh dense layer (BERT) or the equivalent RoBERTa pooler. It is the standard sentence embedding these checkpoints were trained to expose. Alternatives not implemented here:

- Mean of last hidden state over `attention_mask`
- First-token state without the pooler
- Word-aligned gaze attention over tokens

Mean pooling is often better for RoBERTa; the current code does not special-case that.

## Tokenizer pairing

| Encoder | Tokenizer | Checkpoint |
| --- | --- | --- |
| BERT family | `BertTokenizer` | `bert-base-uncased` |
| RoBERTa family | `RobertaTokenizer` | `roberta-base` |

The script picks the tokenizer with `model_type.startswith('bert')`. That is correct for `bert` and `bert_eye_tracking`, and everything else falls through to RoBERTa — including unknown strings. Typos silently load RoBERTa.

## Parameter count (order of magnitude)

- BERT-base / RoBERTa-base: ~110M / ~125M
- Gaze linear: `5*16 + 16 = 96`
- Classifier: `784*3 + 3 = 2,355`

Fusion adds a rounding error of parameters. Any win over text-only is from the **features**, not from capacity. That is why a sklearn logistic baseline on the same five numbers is a useful sanity check: if logistic already separates classes and the transformer fusion does not beat text-only, the extra channel is redundant with the encoder.

## Dummy forward (no weights download)

`examples/gazekit/fusion.py` implements the same affine math on NumPy arrays:

- `project_gaze(et) -> (B, 16)`
- `fuse(pooled, et) -> (B, 3)` logits
- `shapes_report()` for docs / tests

It uses random orthonormal-ish initializations so tests can check ranks and concat width without `torch`. It is not a trained model.

## Optimization (architecture-adjacent)

Both scripts use `torch.optim.Adam` (not `AdamW`) at `lr=5e-5` on **all** parameters, including the full encoder. There is no layer-wise decay, no freeze of lower BERT layers, and no separate, larger LR for the 16-d gaze layer. On 400 rows that means the encoder can overfit freely; 5-fold CV is the only regularizer besides dropout 0.1.

`model_ZuCo_SST.py` trains 20 epochs per fold with batch size 16.
`model_full_SST.py` trains 5 epochs with batch size 256 and keeps the best validation-accuracy checkpoint.

## What a word-aligned sequel would look like

Not implemented. Sketch only:

1. Keep word-level `word_averages_v2.csv`.
2. Align each ZuCo word to one or more WordPiece ids (first-subword convention).
3. Replace the 5-d sentence vector with a `(seq, 5)` tensor.
4. Either concat onto each token hidden state before the pool, or use a second attention over tokens keyed by gaze.

The current late-fusion head is the right first experiment because the CSVs already have sentence means and the alignment problem is messy (three tokenizers, see [data-dictionary.md](data-dictionary.md)).
