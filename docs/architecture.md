# Architecture

This note describes the models in `model_ZuCo_SST.py` and `model_full_SST.py` as they are written, including a few pitfalls that matter when you re-run them.

## Four variants, one switch

Both training scripts pick a variant with a string:

```python
model_type = 'roberta_eye_tracking'
# also: 'bert', 'bert_eye_tracking', 'roberta'
```

| `model_type` | Encoder | Gaze used? | Classification head |
| --- | --- | --- | --- |
| `bert` | `bert-base-uncased` | no | Hugging Face sequence-classification |
| `roberta` | `roberta-base` | no | Hugging Face sequence-classification |
| `bert_eye_tracking` | `BertModel` | yes | custom concat head |
| `roberta_eye_tracking` | `RobertaModel` | yes | custom concat head |

Tokenizer choice follows the prefix: BERT types use `BertTokenizer`, everything else uses `RobertaTokenizer`. Sentences are padded / truncated to **128** tokens.

## Fusion head

`EyeTrackingModel` is a late-fusion wrapper:

```text
input_ids, attention_mask ──► BERT / RoBERTa ──► pooler_output (768)
                                                         │
eye_tracking_features (5) ──► Linear(5 → 16) ────────────┤
                                                         ▼
                                              concat (784)
                                                         │
                                              Dropout(0.1)
                                                         │
                                              Linear(784 → 3)
```

Details that are easy to miss:

- The five gaze features are **already sentence-level**. There is no word-to-token alignment inside the model. Word-level tables in `ZuCo_et_csv_data/word/` and `gaze_prediction/` are used only during data preparation.
- The linear gaze layer has **no activation**. The 16-d vector is a raw projection.
- Dropout sits on the concatenated vector, not inside the gaze branch.
- Loss for fusion models is `CrossEntropyLoss` on reshaped logits. Text-only models use the loss returned by the Hugging Face classifier.
- Hidden size `16` and `num_eye_tracking_features = 5` are module-level constants, not constructor arguments for the gaze width.

`get_model()` constructs a fresh encoder every call. In the ZuCo 5-fold loop that means five independent pretrained checkpoints.

## Dataset object

`CustomDataset` stores:

- `input_ids` and `attention_mask` as Python lists of lists, then wraps each row in `torch.tensor` at `__getitem__`
- integer `sentiment_label`
- a float32 gaze vector of length 5

Full-SST columns are `nFix, FFD, GPT, TRT, GD`. ZuCo-SST columns are `nFixations, FFD, GPT, TRT, GD`. Same five measures, different `nFix` name.

The tokenizer is applied once up front through Hugging Face `datasets.Dataset.map`, then converted back to pandas so the gaze columns can be concatenated by row index. That only works if the gaze frame and the text frame share row order, which they do because both are sliced from the same CSV.

## Training loops

### ZuCo SST (`model_ZuCo_SST.py`)

- Reads `ZuCo_SST_data/combined_sst_et_standard.csv` (~400 rows).
- `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)` on `sentiment_label`.
- 20 epochs per fold, batch size 16, Adam `5e-5`.
- No validation split inside a fold: the held-out 20% is evaluated once after the last epoch.
- Fold metrics are accuracy, weighted precision, weighted recall, weighted F1. Means across folds are printed at the end.
- Checkpoints are **not** saved.

Because the fold test set is also the only evaluation set, the reported numbers are closer to cross-validated train-until-the-end scores than to early-stopped validation scores.

### Full SST (`model_full_SST.py`)

- Reads `SST_data/train_full_sst.csv`, `valid_full_sst.csv`, `test_full_sst.csv`.
- 5 epochs, batch size 256, Adam `5e-5`.
- After each epoch, computes validation accuracy / P / R / F1.
- Keeps the checkpoint with the best **validation accuracy** at `models/best_{model_type}_model.pth`.
- Reloads that checkpoint and evaluates the test loader.

The comment next to `best_val_acc` says "best F1" and the save message also prints the value as F1. The tracked quantity is accuracy.

## Known issues in the training scripts

These are documented so later personal runs do not treat a bad test number as a real result.

### Test-loop overwrite (full SST)

The test loop assigns instead of extending:

```python
all_preds = preds.cpu().numpy()
all_labels = labels.cpu().numpy()
```

Only the **last batch** is scored. With `batch_size=256` that can be a large slice of the test set, but it is not the full test set. The validation loop uses `.extend()` correctly.

### Hardcoded three-class reshape

Fusion loss does `logits.view(-1, 3)`. If `num_labels` ever changes, this stays at 3.

### Path and filename leftovers

- `utils_ZuCo.get_matfiles()` joins `os.getcwd()` with `\\ZuCo_mat_data\\`, a Windows path. On Linux the MATLAB export step needs that string changed or the files copied into the expected folder.
- Several helper scripts are named `spilt.py` (typo for split).
- `get_average_sentence_level.py` reads `et_csv_data/` while the committed tables live in `ZuCo_et_csv_data/`.

### Device and determinism

Device is `cuda` if available, else `cpu`. There is no seed on the full-SST run, no `torch.backends.cudnn.deterministic` flag, and the ZuCo script only seeds the sklearn splitter.

## What the examples approximate

`examples/fusion_toy.py` is a CPU stand-in for the concat head: a bag-of-words (or TF-IDF) text vector plus the five gaze features, fed to logistic regression. It is not a transformer, but it answers the same question: does adding sentence-level gaze change a linear decision surface on these CSVs?
