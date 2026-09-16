# Last-batch test metric (as committed)

In `model_full_SST.py` the test loop does:

```python
preds = torch.argmax(logits, dim=1)
all_preds = preds.cpu().numpy()   # assignment, not extend
all_labels = labels.cpu().numpy()
```

The validation loop correctly `extend`s. The test loop therefore reports
metrics on the **last DataLoader batch only**. Committed test size 1186,
`batch_size = 256`, so that is rows `1024:1186`
(162 examples, 13.7% of the test set). The printed
`Test Acc` is not a test-set score.

The comment above the checkpoint also says 'best F1' while the `if`
condition is `val_acc > best_val_acc`. Two naming bugs in the same file.

Number of batches: 5 (the first 4 are size 256).

## Label prior: full test vs last batch

| label | full_n | full_share | last_n | last_share |
| --- | --- | --- | --- | --- |
| 0 | 463 | 0.3904 | 66 | 0.4074 |
| 1 | 199 | 0.1678 | 28 | 0.1728 |
| 2 | 524 | 0.4418 | 68 | 0.4198 |

## Toy predictors: full vs last-batch scores

These are not model results. They show that the *reporting procedure*
moves the number even when the predictor is trivial.

| predictor | eval | accuracy | precision_weighted | recall_weighted | f1_weighted | n |
| --- | --- | --- | --- | --- | --- | --- |
| majority | full_test | 0.4418 | 0.1952 | 0.4418 | 0.2708 | 1186 |
| majority | last_batch_only | 0.4198 | 0.1762 | 0.4198 | 0.2482 | 162 |
| 70pct_correct_toy | full_test | 0.7816 | 0.7970 | 0.7816 | 0.7861 | 1186 |
| 70pct_correct_toy | last_batch_only | 0.7963 | 0.8135 | 0.7963 | 0.8011 | 162 |
