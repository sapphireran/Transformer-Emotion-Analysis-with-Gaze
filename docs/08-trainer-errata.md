# Trainer errata

The original scripts are left as-is so an old GPU log still matches this
clone. This page is the punch list, not a patch.

## Full-SST test loop keeps the last batch only

`model_full_SST.py` validation does `all_preds.extend(...)`. The test
loop does:

```python
all_preds = preds.cpu().numpy()
all_labels = labels.cpu().numpy()
```

`test_full_sst.csv` has **1,186** rows. `batch_size` is **256**.
`1186 = 4 × 256 + 162`. The printed `Test Acc / P / R / F1` is computed
on **162 rows** (13.7% of the test set), whichever sentences happened to
land in the last DataLoader batch (no shuffle on the test loader, so it
is the last 162 rows of that CSV, not a random subset).

Do not quote that line as a 1,186-row result. Changing `extend` vs `=`
is a one-line fix when someone is ready to break comparability.

## Checkpoint comment vs metric

The same file comments “初始化最佳F1分数” and prints `with F1:` while
comparing `val_acc`. The checkpoint is accuracy-selected.

## ZuCo 80/10/10 is unused and unstratified

`model_ZuCo_SST.py` never opens `ZuCo_SST_data/train.csv`. It folds the
400-row combined table. The stored split is still a useful audit: 320 /
40 / 40, disjoint, covers 0–399, produced by `train_test_split(..., 0.2)`
then 50/50 on the hold-out, `random_state=42`, **no** `stratify=`.

The 40-row valid file is 7 / 14 / 19 (neg / neu / pos). The 400-row mix
is 123 / 137 / 140. A 40-row draw should be near 12 / 14 / 14. The
stored valid set is a 13-point negative-class miss — fine as a folder
artifact, disastrous as a model-selection split, and luckily unused.

## Hard-coded 3-class view

Both trainers do `logits.view(-1, 3)` in the fusion branch even though
`num_labels` exists. Harmless while SST-3 stays 3-way.

## `best_model_path` requires `models/`

`model_full_SST.py` writes `models/best_{model_type}_model.pth` and never
`os.makedirs`. A first run on a clean clone dies at `torch.save` unless
that directory is created.

## Device and weights

Both scripts download `bert-base-uncased` or `roberta-base` at import
time. This environment does not run those trainers. Examples never
import them.

## Metrics

Trainers use sklearn `average='weighted'` for P / R / F1. The personal
`gazebook.metrics.weighted_scores` helper matches that definition so
CPU probes are comparable in *kind*, not in *model*.
