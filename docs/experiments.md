# Experiments

Personal experiment log for the two training tracks. Use this as the checklist when you launch a new run, not as a paper results table. The original scripts do not write TensorBoard events or a `results.json`; they print metrics to stdout.

## Question

Does sentence-level eye-tracking improve three-class SST sentiment accuracy for BERT / RoBERTa, on (a) the small ZuCo-aligned set and (b) the full SST set with projected gaze?

Always keep the text-only twin of each fusion run. A gain that appears only on `roberta_eye_tracking` and not on `roberta` is the number that matters.

## Tracks

### A. ZuCo SST — real gaze, small n

| Item | Value |
| --- | --- |
| Script | `model_ZuCo_SST.py` |
| Table | `ZuCo_SST_data/combined_sst_et_standard.csv` |
| Rows | ~400 |
| Split | `StratifiedKFold(5, shuffle=True, random_state=42)` |
| Epochs | 20 |
| Batch | 16 |
| LR | `5e-5` Adam |
| Metrics | accuracy, weighted P / R / F1, then mean over folds |

Suggested extra runs (edit `model_type` at the top of the file):

1. `bert`
2. `bert_eye_tracking`
3. `roberta`
4. `roberta_eye_tracking`

Optional ablations that still fit this script:

- Drop to 10 epochs if a fold is already saturated on train loss.
- Swap the table to `combined_sst_et_min_max.csv` to see whether z-score vs. min-max changes the fusion gain.
- Train a gaze-only logistic regression (`examples/sentiment_baselines.py`) as a floor. If gaze-only is near chance, fusion gains should be small unless the transformer uses gaze as a tiny residual.

### B. Full SST — projected gaze, larger n

| Item | Value |
| --- | --- |
| Script | `model_full_SST.py` |
| Tables | `SST_data/train_full_sst.csv`, `valid_full_sst.csv`, `test_full_sst.csv` |
| Split | 80 / 10 / 10 from `SST_data/spilt.py` (`random_state=42`) |
| Epochs | 5 |
| Batch | 256 |
| LR | `5e-5` Adam |
| Selection | best **validation accuracy** |
| Metrics | valid each epoch; test once on the saved checkpoint |

Fix or document the test-loop overwrite before you treat a printed test score as the full test set (see [architecture.md](architecture.md)).

Suggested extra runs: the same four `model_type` values.

## Metrics

`calculate_metrics` uses sklearn with `average='weighted'` for precision, recall, and F1. Weighted averages can hide a weak neutral class. When you paste numbers into a note, also log:

- per-class F1 (`average=None`)
- confusion matrix
- support counts

The example script `examples/sentiment_baselines.py` already prints a classification report so you have a template.

## What to record for each run

Create a dated file under `result/` (the folder is in the repo and currently empty) with:

```text
date:
script:
model_type:
data:
epochs:
batch_size:
lr:
device:
seed:          # only the ZuCo k-fold seed is set today
valid_acc:
valid_f1:
test_or_fold_mean_acc:
test_or_fold_mean_f1:
notes:         # e.g. "test loop last-batch only"
```

Stdout from `tqdm` is enough if you redirect it:

```bash
python3 model_ZuCo_SST.py 2>&1 | tee result/zuco_roberta_et_$(date +%Y%m%d).log
```

## CPU baselines that should precede a GPU run

These do not replace the transformers. They tell you whether the CSVs contain any linear signal.

| Script | Question |
| --- | --- |
| `examples/inspect_datasets.py` | Did the join keep all ids and all three labels? |
| `examples/gaze_feature_stats.py` | Do any gaze columns differ by class? |
| `examples/sentiment_baselines.py` | Gaze-only vs. length-only vs. TF-IDF vs. TF-IDF+gaze |
| `examples/fusion_toy.py` | Same as the last comparison, with an explicit concat implementation |
| `examples/word_level_gaze.py` | Are predicted word-level features well-formed? |

If TF-IDF already saturates the ZuCo set, a 20-epoch RoBERTa run will overfit; treat CV means cautiously.

Linear floors from `examples/fusion_toy.py` on this checkout (balanced logistic regression):

| Track | text TF-IDF weighted F1 | concat TF-IDF+gaze16 | Δ |
| --- | --- | --- | --- |
| ZuCo (n_test=40) | 0.520 | 0.500 | −0.020 |
| Full SST (n_test=1186) | 0.624 | 0.628 | +0.004 |

Gaze-only sits near chance (~0.33–0.35). Refresh `examples/sample_outputs/fusion_toy.json` if you change a split.

## Seeds and comparability

- ZuCo folds are comparable across `model_type` values as long as you do not change `random_state=42` or the CSV.
- Full SST shuffle in `spilt.py` is also `random_state=42`. Retrain variants on the committed split files rather than re-splitting.
- PyTorch itself is unseeded. Two GPU runs of the same `model_type` can differ. For a personal paper draft, set `torch.manual_seed`, `numpy.random.seed`, and `random.seed` at the top of the script.

## Negative results are useful

On this dataset a likely outcome is: **real ZuCo gaze helps a little or not at all once the transformer is strong, and projected full-SST gaze helps even less.** Write that down. The value of the repo is the aligned tables and the fusion code, not a guaranteed accuracy bump.
