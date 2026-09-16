# Known bugs

Personal debugging list. The example package works around these
instead of editing the original scripts.

## High: full-SST test loop keeps the last batch

`model_full_SST.py` validation:

```python
all_preds.extend(preds.cpu().numpy())
```

`model_full_SST.py` test:

```python
all_preds = preds.cpu().numpy()
all_labels = labels.cpu().numpy()
```

With `batch_size=256` and 1,186 test rows the printed test Acc/P/R/F1
cover **162** sentences. Example 07 reproduces the coverage math with
a dummy model. Until that assignment becomes `extend`, do not treat
the printed test line as a full-set score.

## High: subject 3 is remapped, then averaged by index

See [subject alignment](subject-alignment.md). File `3_SR.csv` has
299 compacted ids. `get_average_sentence_level.py` and the word-level
averager group by row index. 246 sentence means are slightly wrong.
`SentLen` at id 150 is the smoking gun (9.5 vs 9.0).

## Medium: producer paths that do not exist here

| location | it asks for | what exists |
| --- | --- | --- |
| `utils_ZuCo.get_matfiles` | `\\ZuCo_mat_data\\` | no MATLAB tree; backslashes |
| `read_ZuCo_mat.py` | `et_csv_data/` | `ZuCo_et_csv_data/` |
| `get_average_sentence_level.py` | `et_csv_data` | same |
| `convert_full_SST.py` | `ZuCo_SST_data/all/` | not checked in |
| `ZuCo_SST_data/save_SST_data.py` | `all/`, writes `output.csv` | neither is checked in |

If a script dies with `FileNotFoundError` on one of those names, you
are re-running a producer. The consumer CSVs are already present.

## Low: checkpoint comment vs metric

```python
best_val_acc = 0.0  # 初始化最佳F1分数
if val_acc > best_val_acc:
    ...
    print(f"... with F1: {best_val_acc:.4f}")
```

The file is saved on validation **accuracy**.

## Low: `num_labels` is half-honored

Fusion loss uses `logits.view(-1, 3)`. Changing `num_labels` does not
reconfigure that view.

## Low: `spilt.py`

Both split helpers are named `spilt.py`. Cosmetic. The ZuCo 40-row
valid split they produce is also unstratified (7 / 14 / 19).

## Low: RoBERTa pooler assumption

`EyeTrackingModel.forward` always uses `pooler_output`. Fine for the
stock `roberta-base` constructor used here. Not fine for a custom
RoBERTa without a pooler.

## Not bugs, just easy to misread

- Full-SST gaze is projected. It is z-scored, so it *looks* like the
  ZuCo standard table.
- Only three normalized strings are shared between the two combined
  tables.
- `model_ZuCo_SST.py` does not use `train.csv`.
- Word-mean nFixations ≠ sentence-mean nFixations (fixated-word
  denominator).
