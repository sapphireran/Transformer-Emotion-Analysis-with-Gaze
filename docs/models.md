# Models

Personal notes on the two training scripts. Architecture is the same
idea in both files; the loop is not.

## Variants

Set `model_type` at the top of the script:

| `model_type` | backbone | gaze |
| --- | --- | --- |
| `bert` | `BertForSequenceClassification` | no |
| `roberta` | `RobertaForSequenceClassification` | no |
| `bert_eye_tracking` | `BertModel` + concat head | yes |
| `roberta_eye_tracking` | `RobertaModel` + concat head | yes |

The default in both files is `roberta_eye_tracking`.

## Fusion head

```python
class EyeTrackingModel(nn.Module):
    # pooler (768) ──┐
    # Linear(5→16) ──┴─► dropout 0.1 ─► Linear(784→3)
```

- Gaze features are **not** layered per token. One 5-d vector per
  sentence.
- The script always reads `base_output.pooler_output`. Stock
  `RobertaModel` ships a pooler. A pooler-free checkpoint would need
  the first-token state instead.
- Fusion loss is `CrossEntropyLoss()(logits.view(-1, 3), labels.view(-1))`.
  `num_labels` is not used at that site.

## ZuCo script (`model_ZuCo_SST.py`)

| knob | value |
| --- | --- |
| data | `combined_sst_et_standard.csv` (400 rows) |
| epochs | 20 |
| batch size | 16 |
| lr | 5e-5, Adam |
| protocol | `StratifiedKFold(5, shuffle=True, random_state=42)` |
| checkpoint | none; prints per-fold and mean Acc/P/R/F1 |

The 320/40/40 files are unused. Metrics are weighted. With 400 rows
and 20 epochs this is the cheaper of the two runs, but it still
downloads `roberta-base` and wants a GPU to be pleasant.

## Full-SST script (`model_full_SST.py`)

| knob | value |
| --- | --- |
| data | `train/valid/test_full_sst.csv` |
| epochs | 5 |
| batch size | 256 |
| lr | 5e-5, Adam |
| protocol | train → valid each epoch → save best Acc → test |
| checkpoint | `models/best_{model_type}_model.pth` |

Create `models/` before running. The save log says "with F1" but the
predicate is `val_acc > best_val_acc`.

The **test** loop overwrites `all_preds` / `all_labels` each batch.
At `batch_size=256` and 1,186 test rows that is the last **162**
examples. Validation numbers from the same script are fine. See
[known bugs](known-bugs.md) and example 07.

## What a CPU demo is allowed to claim

`examples/08_cpu_fusion.py` keeps the concat story and replaces the
transformer with a hashed bag-of-words of width 32. It is a wiring
check. It is not a RoBERTa score.

`examples/09_gaze_baselines.py` fits a stdlib one-vs-rest least
squares on the five fusion features. On a stratified 20% holdout of
the 400-row table it lands around **0.46** accuracy against a
majority floor of **0.35**. That is a weak standalone signal, not a
reason to skip the text encoder.

## Hardware

Both scripts pick `cuda` if `torch.cuda.is_available()` else `cpu`.
Full SST at batch 256 is a GPU job. The personal examples do not
import torch.
