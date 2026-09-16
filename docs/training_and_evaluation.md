# Training and evaluation

This page is the "how do I actually run the original scripts" note.
For architecture see [models.md](models.md). For bugs that affect
reported numbers see [known_issues.md](known_issues.md).

## Shared knobs

| Knob | Full SST (`model_full_SST.py`) | ZuCo (`model_ZuCo_SST.py`) |
| --- | --- | --- |
| `num_eye_tracking_features` | 5 | 5 |
| `hidden_layer_size` | 16 | 16 |
| `num_labels` | 3 | 3 |
| `num_epochs` | 5 | 20 |
| `learning_rate` | `5e-5` | `5e-5` |
| `batch_size` | 256 | 16 (the CV loop also hard-codes 16) |
| Default `model_type` | `roberta_eye_tracking` | `roberta_eye_tracking` |
| Split | 9,482 / 1,185 / 1,186 holdout | `StratifiedKFold(5, shuffle=True, random_state=42)` |
| Checkpoint | `models/best_{model_type}_model.pth` on best val **accuracy** | none (prints fold + mean metrics) |

Edit the knobs in the script header. There is no CLI.

## Environment

```bash
python3 -m pip install -r requirements-train.txt
# GPU: install the CUDA wheel of torch that matches the machine
```

On first run Hugging Face will download `bert-base-uncased` or
`roberta-base` (tokenizer + encoder). That is several hundred MB and
needs network. The scripts always `from_pretrained(...)` — they do not
look for a local cache flag.

Device selection:

```python
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
```

Batch size 256 on full SST will not fit on CPU in any comfortable way.
ZuCo's batch size 16 × 400 sentences × 5 folds × 20 epochs is the more
realistic CPU experiment, and it is still slow because the whole
encoder is trained.

Create `models/` before a full-SST run. `torch.save` will not create
the directory.

## Full SST procedure

1. Set `model_type` to one of `bert`, `roberta`, `bert_eye_tracking`,
   `roberta_eye_tracking`.
2. `python3 model_full_SST.py`
3. Each epoch: train on the 9,482-row loader, then evaluate the 1,185-row
   validation loader with weighted P / R / F1 and accuracy.
4. If `val_acc` improves, overwrite `models/best_{model_type}_model.pth`.
5. After the last epoch, reload that checkpoint and run the test loader.

**Do not trust the printed test line until you have read
[known_issues.md](known_issues.md).** The test loop assigns
`all_preds = preds.cpu().numpy()` instead of `extend`, so the number is
the last batch only (up to 256 rows), not the full 1,186.

Validation is fine: it uses `extend`.

Compare four `model_type` values on the **same** split if you want the
gaze ablation. Because the split is not stratified, write down the
label counts (`examples/02_label_and_length_profile.py`) next to any
accuracy you report.

## ZuCo 400 procedure

1. Set `model_type`.
2. `python3 model_ZuCo_SST.py`
3. For each of 5 folds: train 20 epochs on 80% of the 400, evaluate
   once on the held-out 20% **after the last epoch** (no per-epoch
   validation, no early stopping, no checkpoint).
4. Print the five fold scores and their means.

`StratifiedKFold` keeps class balance in each fold, which is why this
protocol is a better default on 400 rows than the unused 320/40/40
CSVs.

There is no test set separate from the folds. If you need an untouched
holdout, use `ZuCo_SST_data/test.csv` and change the script — do not
pretend the fold mean is a holdout number.

## What to log

Minimum useful record for a personal experiment:

- `model_type`
- script name (full SST vs ZuCo)
- epoch / batch / lr
- device
- git commit
- validation (or fold) accuracy, weighted F1
- whether you applied the test-loop fix
- note that full-SST gaze is predicted, ZuCo gaze is measured

## Ablations worth running

1. `roberta` vs `roberta_eye_tracking` on ZuCo 400 — this is the
   cleanest "does measured gaze help?" comparison.
2. Same pair on full SST — "does *predicted* gaze help?"
3. `bert` vs `roberta` text-only, to see if a gaze gain is just "RoBERTa
   is better."
4. Shuffle the gaze columns as a negative control (not implemented;
   `examples/03_gaze_feature_stats.py` at least tells you whether the
   features correlate with the label at all before you train).

## Metrics code

Both scripts share:

```python
def calculate_metrics(preds, labels):
    accuracy = accuracy_score(labels, preds)
    precision = precision_score(labels, preds, average='weighted')
    recall = recall_score(labels, preds, average='weighted')
    f1 = f1_score(labels, preds, average='weighted')
    return accuracy, precision, recall, f1
```

`sklearn` will warn if a fold never predicts a class. That is common
on 80-row ZuCo test folds if the model collapses to majority. The
weighted average still produces a number; look at the confusion
pattern before celebrating.

`examples/lib/metrics.py` reimplements the same four scores so the
docs/examples path does not import the training scripts.
