# Known pitfalls

Bugs, naming traps, and environment mismatches discovered by reading the
scripts against the checked-in CSVs. Fixing the training scripts is out of
scope for this docs pass; the example code demonstrates the **intended**
behavior where that helps.

## Paths that do not match the tree

| Script constant | Checked-in location |
| --- | --- |
| `et_csv_data/` in `read_ZuCo_mat.py`, `get_average_sentence_level.py` | `ZuCo_et_csv_data/` |
| `subdir='\\ZuCo_mat_data\\'` | MATLAB dumps not in git; Windows slashes |
| `training_data/word_averages_v2.csv` in `convert_zuco_data.py` | `ZuCo_et_csv_data/word/word_averages_v2.csv` |
| `ZuCo_SST_data/all/{LABEL}/*.txt` | not checked in (zip may contain a snapshot) |
| `models/` | not created automatically |

Re-running converters without renaming/copying will `FileNotFoundError`
even though the CSVs already exist under the `ZuCo_*` names.

## Filename typos

- `spilt.py` in both `ZuCo_SST_data/` and `SST_data/` — “split.”
- `stts_all_sentence_level.csv` — “ssts.”
- `get_average_sentence_level.py` still documents `folder_path = 'et_csv_data'`.

## `nFix` vs `nFixations`

Track A columns use `nFixations`. Track B uses `nFix`. The fusion order is
aligned in Python (`nFixations, FFD, GPT, TRT, GD` vs `nFix, FFD, GPT, TRT,
GD`). A naive `pd.concat` of the two CSVs will **not** line those columns
up; `nFixations` will be NaN on full-SST rows.

PROVO uses `fixProp` instead of `GD`.

## Test metrics in `model_full_SST.py` overwrite the lists

```python
preds = torch.argmax(logits, dim=1)
all_preds = preds.cpu().numpy()   # should be extend
all_labels = labels.cpu().numpy()
```

The validation loop uses `extend` correctly. The test loop replaces the
accumulator every batch, so printed **Test Acc/P/R/F1** are last-batch
metrics. Until that is patched, trust the validation lines, or compute test
metrics with `examples/metrics_example.py` on saved predictions.

## Checkpoint log says F1, selects accuracy

```python
best_val_acc = 0.0
if val_acc > best_val_acc:
    ...
    print(f"... with F1: {best_val_acc:.4f}")
```

The saved model is the best **accuracy** on valid, not best F1.

## `batch_size` ignored in Track A

`DataLoader(..., batch_size=16, ...)` is a literal. Changing the config
`batch_size = 16` at the top of `model_ZuCo_SST.py` does nothing.

## No seeds for PyTorch / CUDA / numpy

Only `StratifiedKFold(..., random_state=42)` and the sklearn splits use a
seed. Encoder init, dropout, and data shuffle still vary. Fold **indices**
are stable; fold **metrics** are not bit-stable.

## Subject 3 index alignment

`3_SR.csv` has 300 sentence rows because task 1 subject 2 drops 101
trials. Averaging with `groupby(level=0)` after `concat` aligns on the
**0..n-1 index of each file**, so late IDs mix different MATLAB sentences
across subjects. `examples/subject_coverage.py` prints the row counts.
Do not publish “N=12 per sentence” without a join on a stable sentence key
(the word tables’ `Sent_ID` is closer to that, and even then subject 3 has
fewer words).

## Aggressive 0 → NaN at sentence average

`get_average_sentence_level.py` replaces **all** zeros except `id` with
NaN. True-zero `omissionRate` or `SFD` becomes missing. Word-level v2
averaging does the opposite (keeps zeros). Pick one story when you describe
“mean gaze.”

## `EyeTrackingModel` always loads from the class, not `model_type` string

```python
self.base_model = base_model.from_pretrained(
    'bert-base-uncased' if base_model == BertModel else 'roberta-base'
)
```

That is fine for the four `get_model` branches. It would be wrong if you
passed a fine-tuned `BertModel` subclass; it compares the class object.

## RoBERTa + `BertTokenizer` footgun

`model_type.startswith('bert')` selects the tokenizer. A typo like
`bert_roberta_eye_tracking` would load BERT WordPiece onto a RoBERTa
encoder. Stick to the four documented strings.

## Full-SST test loop vs sklearn `average='weighted'`

If you patch `extend` and a class is missing from a small last batch,
sklearn will warn. On the full test set all three labels should appear.
`zero_division` is left at the sklearn default.

## `convert_sst_to_et.py` drops non-letters

Tokens must match `^[A-Za-z]+$`. Contractions, numbers, and `'s` disappear.
Predicted gaze is therefore **not** on the same tokenization as RoBERTa
BPE. That is acceptable for sentence-level mean pooling of gaze, but it
rules out naive token-index alignment for early fusion.

## `pandas` / `datasets` required by training, not by examples

The original scripts import `datasets.Dataset` only to call `tokenizer`
in a map. Examples avoid that stack so they run with numpy + stdlib.

## Zip in `ZuCo_SST_data/`

`ZuCo_SST_data.zip` may duplicate CSVs. Prefer the loose CSV files as the
source of truth; do not commit extracted junk from unzipping on top of them.
