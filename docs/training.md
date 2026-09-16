# Training

Constants copied from the two scripts as they exist on `main`. There is no config file or argparse.

## Shared

| Knob | Value |
| --- | --- |
| Labels | 3 (`CrossEntropyLoss`) |
| Tokenizer max length | 128, `padding='max_length'` |
| Gaze hidden size | 16 |
| Dropout on concat | 0.1 |
| Learning rate | `5e-5` |
| Optimizer | Adam (PyTorch, not AdamW) |
| Default `model_type` | `roberta_eye_tracking` |
| Device | CUDA if `torch.cuda.is_available()` else CPU |
| Metrics | accuracy, weighted precision / recall / F1 (`sklearn`) |
| `zero_division` | sklearn default (warnings on missing class in a tiny fold) |

`calculate_metrics` is identical in both files.

## ZuCo-SST — `model_ZuCo_SST.py`

| Knob | Value |
| --- | --- |
| Table | `ZuCo_SST_data/combined_sst_et_standard.csv` (400 rows) |
| Gaze columns | `nFixations, FFD, GPT, TRT, GD` |
| Epochs | 20 per fold |
| Batch size | 16 (constant also duplicated as `16` in `DataLoader`) |
| Protocol | `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)` |
| Checkpoint | none; last epoch of each fold is evaluated |
| Printed summary | mean accuracy / P / R / F1 over the five held-out folds |

The script does **not** evaluate a nested test set. The committed `train.csv` / `valid.csv` / `test.csv` are a separate 80/10/10 split from `spilt.py` and are unused here.

On 400 sentences, 20 epochs of full RoBERTa is enough to overfit a fold. Treat a large CV number as a hypothesis, not a leaderboard score.

## Full SST — `model_full_SST.py`

| Knob | Value |
| --- | --- |
| Train / valid / test | `SST_data/{train,valid,test}_full_sst.csv` |
| Gaze columns | `nFix, FFD, GPT, TRT, GD` |
| Epochs | 5 |
| Batch size | 256 |
| Selection | highest **validation accuracy** (variable `best_val_acc`) |
| Checkpoint | `models/best_{model_type}_model.pth` |
| Test | reload best state, one pass over `test_loader` |

The comment next to `best_val_acc` says “best F1”; the condition is `if val_acc > best_val_acc`. The save log string also says `with F1:` while printing accuracy.

**Test-loop bug:** the test `for batch` assigns `all_preds = preds.cpu().numpy()` instead of `extend`. Metrics are computed on the **last batch only** (up to 256 rows of the 1,186-row test set). Validation uses `extend` and is fine. Details in [known-issues.md](known-issues.md).

Create `models/` before running; `torch.save` will not create the directory.

## Batch / memory ballpark

| Script | Sequence length | Batch | Activations |
| --- | --- | --- | --- |
| ZuCo | 128 | 16 | comfortable on a single 8–12 GB GPU |
| Full SST | 128 | 256 | needs more VRAM; drop `batch_size` if it OOMs |

CPU training of `roberta-base` on 9,482 rows is possible but slow; the example package does not call Hugging Face.

## Metrics reporting

Weighted P/R/F1 matches a label distribution that is not uniform (full SST ≈ 39% neg / 19% neu / 42% pos; ZuCo is closer to balanced). If you add a confusion matrix later, keep the `{0,1,2}` mapping from [datasets.md](datasets.md).

`examples/teag_examples/metrics.py` mirrors `calculate_metrics` so docs examples can unit-test the same four numbers without importing torch.

## Reproducibility gaps

Neither training script sets `torch.manual_seed`, `numpy.random.seed`, or `random.seed`. KFold / `train_test_split` use `random_state=42`, but dropout, shuffling, and GPU kernels will still move. Hugging Face download of `roberta-base` must succeed at runtime.
