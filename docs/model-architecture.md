# Model architecture

The personal trainers are two self-contained scripts. They duplicate the `EyeTrackingModel` class on purpose (no shared package when they were written). `tea_gaze` does **not** re-implement the transformer; it only documents the layout and offers sklearn stand-ins.

## Shared fusion head

```text
input_ids, attention_mask
        │
        ▼
 BERT or RoBERTa encoder
        │
        ▼
  pooler_output          eye_tracking_features  (batch, 5)
   (batch, 768)                    │
        │                          ▼
        │                 Linear(5 → 16)
        │                          │
        └────────► concat (batch, 784)
                          │
                          ▼
                       Dropout(0.1)
                          │
                          ▼
                  Linear(784 → 3 logits)
```

Forward (from either script):

```python
base_output = self.base_model(input_ids=input_ids, attention_mask=attention_mask)
pooled_output = base_output.pooler_output
eye_tracking_output = self.eye_tracking_layer(eye_tracking_features)
combined_output = torch.cat((pooled_output, eye_tracking_output), dim=1)
combined_output = self.dropout(combined_output)
return self.classifier(combined_output)
```

Loss for the fusion models is `CrossEntropyLoss` on the 3-class logits. Text-only `*ForSequenceClassification` models use the built-in `outputs.loss`.

Tokenizer max length is 128. BERT uses `bert-base-uncased`; RoBERTa uses `roberta-base`.

## Track A — `model_ZuCo_SST.py`

| Knob | Value |
|---|---|
| Data | `ZuCo_SST_data/combined_sst_et_standard.csv` |
| Gaze columns | `nFixations, FFD, GPT, TRT, GD` |
| Split | `StratifiedKFold(5, shuffle=True, random_state=42)` |
| Epochs | 20 (no early stopping, no per-fold checkpoint) |
| Batch | 16 |
| Optimizer | Adam `5e-5` |
| Hidden gaze layer | 16 |
| Metrics | weighted P/R/F1 + accuracy, then mean over folds |

The script tokenizes the whole CSV up front with Hugging Face `datasets`, concatenates the gaze columns back on, and rebuilds `CustomDataset` per fold.

## Track B — `model_full_SST.py`

| Knob | Value |
|---|---|
| Data | `SST_data/{train,valid,test}_full_sst.csv` |
| Gaze columns | `nFix, FFD, GPT, TRT, GD` |
| Split | the pre-cut 80/10/10 files |
| Epochs | 5 |
| Batch | 256 |
| Checkpoint | `models/best_{model_type}_model.pth` on **validation accuracy** |
| Test | reload that checkpoint |

The printed message says “best F1” when it actually tracks accuracy. The test loop assigns `all_preds = preds.cpu().numpy()` instead of extending the list, so the published test line is the **last batch only**. Details in [known-issues.md](known-issues.md).

## Custom dataset

`CustomDataset.__getitem__` returns:

- `input_ids`, `attention_mask` as tensors
- `labels` as the integer sentiment
- `eye_tracking_features` as `float32` length-5

Text-only branches ignore the gaze tensor.

## Sklearn companion (not a substitute)

`tea_gaze.baselines` answers “did gaze do anything on these 400 rows?” without a GPU:

| Name | Features |
|---|---|
| `text` | TF-IDF (1–2 grams, 4000 cols, `min_df=2`) |
| `gaze` | the same five fusion columns, `StandardScaler` |
| `fusion` | `hstack([tfidf, scaled_gaze])` |

All three use `LogisticRegression(class_weight='balanced')` and the same 5-fold seed as Track A. Run `examples/03_text_vs_gaze_baselines.py`.

If fusion F1 is indistinguishable from text F1 on this tiny set, that is a useful personal result: the linear head in the transformer scripts may also be starved for signal, or the 400 sentences may already be lexically easy.

## Suggested personal log

When you do run the transformer scripts, write down at least:

- `model_type`
- which CSV (standard vs min-max vs full SST)
- fold metrics *and* the mean
- whether you patched the full-SST test loop
- GPU / epoch time

There is no `result/` logger in the original scripts; they print to stdout.
