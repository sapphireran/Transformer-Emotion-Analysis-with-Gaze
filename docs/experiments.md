# Experiments and hyperparameters

Two scripts, one architecture, two evaluation protocols. Constants are hardcoded
at the top of each file — there is no CLI.

## Shared constants

| Name | Value | Meaning |
| --- | --- | --- |
| `num_eye_tracking_features` | 5 | `nFix(ations)`, `FFD`, `GPT`, `TRT`, `GD` |
| `hidden_layer_size` | 16 | Gaze projection width |
| `num_labels` | 3 | Negative / neutral / positive |
| `learning_rate` | `5e-5` | Adam on the full network |
| Encoder | `bert-base-uncased` or `roberta-base` | Chosen by `model_type` |
| Max length | 128 | Tokenizer pad / truncate |

## Track 1 — ZuCo-grounded SST

File: `model_ZuCo_SST.py`  
Data: `ZuCo_SST_data/combined_sst_et_standard.csv` (400 rows)

| Knob | Value |
| --- | --- |
| Epochs | 20 |
| Batch size | 16 |
| Split | `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)` |
| Checkpointing | none (last epoch of each fold is evaluated) |
| Printed summary | mean accuracy / P / R / F1 over the 5 folds |

Each fold:

1. Rebuild `get_model(...)` (fresh pretrained weights).
2. Train 20 epochs on 320 sentences.
3. Evaluate once on the remaining 80.
4. Append fold metrics; no early stopping.

Because `random_state=42` is fixed, the five test index sets are reproducible.

This track answers: *given real subject-averaged gaze on the 400 ZuCo movie
reviews, does the fusion head beat text-only BERT/RoBERTa under CV?*

Switch `model_type` to `'roberta'` or `'bert'` for the control.

## Track 2 — Full SST with transferred gaze

File: `model_full_SST.py`  
Data: `SST_data/{train,valid,test}_full_sst.csv`

| Knob | Value |
| --- | --- |
| Epochs | 5 |
| Batch size | 256 |
| Split | 9,482 / 1,185 / 1,186 |
| Checkpoint | `models/best_{model_type}_model.pth` on best **validation accuracy** |
| Test | reload that checkpoint, run `test_loader` |

This track answers: *can predicted or transferred sentence gaze still help when
the text set is the full SST?* Five epochs and a large batch are a “does it
run” setup more than a paper-ready schedule.

Create `models/` before the first run or `torch.save` will fail.

## Suggested personal ablations (not automated)

Keep the CSV and the dataset class; only change one thing.

1. **Text-only vs fusion** — `roberta` vs `roberta_eye_tracking` on both tracks.
2. **BERT vs RoBERTa** — four-way grid on ZuCo CV (cheap enough at 400 rows).
3. **Gaze subset** — drop `GPT` or `TRT` by slicing the feature matrix.
4. **Scaling** — point the ZuCo script at `combined_sst_et_min_max.csv`
   (rename columns or edit the feature list; min-max uses the same names).
5. **Unused channels** — add `SFD`, `omissionRate`, `meanPupilSize` and bump
   `num_eye_tracking_features` to 8.
6. **Hold-out vs CV on ZuCo** — train on `train.csv`, pick on `valid.csv`.
   Expect noisy valid numbers (7 negatives).
7. **Frozen encoder** — `param.requires_grad = False` on `base_model`.

Log fold-level arrays (`val_accs`, …) if you add anything; they are already
in memory at the bottom of `model_ZuCo_SST.py`.

## What I would not treat as a finished number

- `model_full_SST.py` test metrics currently use only the **last test batch**
  because `all_preds` is assigned rather than extended. Fix that before
  quoting a test score. See [known-issues.md](known-issues.md).
- ZuCo CV reports the last epoch, not the best epoch. A high fold score can be
  memorization.
- Full-SST gaze is not 12-subject ZuCo gold. Compare fusion vs text-only on
  *that* table, not against the ZuCo CV number.

## Hardware notes

- ZuCo track: 5 folds × 20 epochs × 320/16 batches. Fits a single consumer GPU
  easily; CPU is slow but possible.
- Full-SST track: 5 epochs × 9,482/256 batches of RoBERTa. Wants a GPU.
- `examples/` never loads transformers and is fine on CPU.

## Baseline you can compute without a model

`examples/gaze_by_sentiment.py` prints class-conditional gaze means. If those
means barely separate, any fusion gain has to come from interactions with the
text vector, not from a linear gaze-only classifier. You can also train a
tiny `Linear(5, 3)` on gaze alone inside `fusion_forward.py` (the script
already reports that gaze-only accuracy on a random toy batch — it will be
chance, which is the point of the demo).
