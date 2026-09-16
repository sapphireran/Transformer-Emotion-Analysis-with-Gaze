# Experiments and known issues

Personal runbook for the two training scripts. Numbers below are **file facts and code behavior**, not claimed paper scores. This snapshot has an empty `result/` of plots only — no `*.json` metric dumps.

## Setting A — Native ZuCo (real gaze)

**Script:** `model_ZuCo_SST.py`  
**Table:** `ZuCo_SST_data/combined_sst_et_standard.csv` (400 rows)  
**Protocol:** `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`  
**Epochs / batch / lr:** 20 / 16 / 5e-5  
**Metrics:** accuracy and weighted P / R / F1 on each fold, then the arithmetic mean

### Why this setting exists

This is the only setting where the five gaze numbers are *measurements of people reading the labeled sentence*. If fusion does nothing here, predicted-gaze SST results are not evidence for the hypothesis.

### What to log

For each of `{bert, roberta, bert_eye_tracking, roberta_eye_tracking}`:

- 5 fold scores
- mean and standard deviation
- wall time and whether CUDA was used

Compare **paired** folds (same `random_state`, same indices) so a 1-point gap is not just a different shuffle.

### Risks

- **Memorization.** 320 train sentences, 110M parameters, 20 epochs, no weight decay, no early stopping.
- **Gaze is only weakly related to the label.** On `combined_sst_et_standard.csv`, Pearson r between each of the five fusion features and `sentiment_label` is in roughly 0.03–0.07 (`examples/feature_stats.py`). TRT/nFixations/GPT are highly collinear with each other (r > 0.9). Fusion is asking the classifier to use a small residual, not a second copy of the label.
- **Leak via length.** Even without gaze, sentence length correlates weakly with the ZuCo review set. Gaze features also correlate with length. A linear probe on `{nFixations, TRT}` alone is a useful personal baseline (`examples/feature_stats.py` prints the raw moments; a sklearn probe can be added later).
- **Subject 3 hole.** If you regenerate averages from the CSVs in this clone with the current `get_average_sentence_level.py`, positional mean + a 299-row file will corrupt the last 101 rows. Prefer the checked-in averaged tables unless you re-align on `id`.

## Setting B — Full SST (predicted gaze)

**Script:** `model_full_SST.py`  
**Tables:** `SST_data/{train,valid,test}_full_sst.csv` (9482 / 1185 / 1186)  
**Protocol:** one split, checkpoint on validation **accuracy**, evaluate test once  
**Epochs / batch / lr:** 5 / 256 / 5e-5  
**Checkpoint path:** `models/best_{model_type}_model.pth`

### Why this setting exists

RoBERTa fine-tuning is more stable at 9k+ sentences. The cost is that `nFix, FFD, GPT, TRT, GD` are **not** ZuCo recordings of those SST sentences. They are outputs of a side gaze-prediction experiment (`gaze_prediction/data/`).

### What would make a gain untrustworthy

1. The gaze predictor was trained with sentiment as an input feature.
2. The predictor was trained on ZuCo *sentiment* sentences and those sentences also appear in full SST (they do — ZuCo Task 1 *is* a movie-review subset). Predicted gaze on overlapping text can leak ZuCo-specific reading patterns that correlate with the label.
3. Sentence-level pooling used the label (it should not).

Until the missing predictor script is restored, treat Setting B as “does a five-dimensional side channel change SST numbers?”, not as “human gaze improves SST.”

## Shared implementation notes

### Metrics are weighted, not macro

```python
precision_score(..., average='weighted')
```

On full SST, weighted F1 sits close to accuracy because negative and positive dominate. If you care about neutral, compute `average='macro'` yourself.

### Best-model comment does not match the code

`model_full_SST.py`:

```python
best_val_acc = 0.0  # 初始化最佳F1分数
...
if val_acc > best_val_acc:
    ...
    print(f"... with F1: {best_val_acc:.4f}")
```

Selection is accuracy. The printed word “F1” is wrong.

### Test loop overwrites the prediction list

In `model_full_SST.py` the test loop does:

```python
all_preds = preds.cpu().numpy()
all_labels = labels.cpu().numpy()
```

instead of `.extend(...)`. With `batch_size=256` and 1186 test rows that is **five** batches; reported test metrics are the **last batch only** (~118 rows). Validation uses `extend` and is fine.

Personal rule: do not quote a test number from this script until that assignment is `extend`. The example suite does not patch the training script; it only documents the bug.

### Token alignment

`load_dataset` / the ZuCo prelude concatenates tokenized pandas with the gaze frame on **axis=1**. If any row is dropped during tokenization (none should be; `map` is 1:1), gaze would silently mis-align. `examples/schema_validate.py` checks that the source CSVs have no null sentences and that gaze columns are finite.

### Pooler vs first token

Fusion uses `pooler_output`, not `last_hidden_state[:, 0]`. For BERT these differ by a trained pooler. For RoBERTa the pooler is typically unused in the original pretraining objective; it still exists on `RobertaModel` and *is* used here. A personal ablation would try CLS / `<s>` directly.

### Device and reproducibility

- No `torch.manual_seed` in either script (ZuCo KFold is seeded; SGD is not).
- `num_workers` is default 0.
- `torch.load` in the full-SST reload does not set `weights_only=True` (fine on a personal machine; noisy on newer PyTorch).

## Suggested personal log format

When you do run a GPU job, drop a JSON line next to `result/` rather than overwriting plots:

```json
{
  "setting": "zuco_cv",
  "model_type": "roberta_eye_tracking",
  "seed_kfold": 42,
  "epochs": 20,
  "fold_acc": [0, 0, 0, 0, 0],
  "mean_acc": 0,
  "mean_f1_weighted": 0,
  "notes": ""
}
```

No such files are in this commit on purpose: empty templates invite fake numbers.

## Ablations worth doing (not implemented)

1. Text-only vs gaze, both encoders, both settings.
2. Shuffle gaze rows (destroy alignment) — fused model should collapse to text-only. If it does *better*, there is a bug.
3. Length-only instead of five features.
4. Standard vs min-max ZuCo tables.
5. Macro-F1 in addition to weighted.

`examples/gaze_fusion_demo.py` includes a tiny shuffle-vs-aligned toy so the *idea* of ablation 2 is easy to explain without a GPU.
