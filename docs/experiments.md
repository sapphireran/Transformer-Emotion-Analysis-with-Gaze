# Experiments

This page records what the two trainers actually do, not what a cleaned
re-implementation might do. Hyperparameters are literals at the top of
`model_ZuCo_SST.py` and `model_full_SST.py`.

---

## Shared protocol

- **Task**: 3-way sentiment (`0` neg, `1` neu, `2` pos).
- **Encoders**: `bert-base-uncased` or `roberta-base`.
- **Gaze**: five floats, linear map to 16-d, concat with pooler.
- **Loss**: cross-entropy.
- **Optimiser**: Adam, `lr = 5e-5`, no weight decay argument (PyTorch
  default `0.0`).
- **Metrics**: accuracy and *weighted* precision / recall / F1.
- **Device**: `cuda` if `torch.cuda.is_available()` else `cpu`.

The interesting comparison is always **paired**: same encoder, with vs
without gaze, same splits / folds. Mixing BERT-text-only with
RoBERTa+gaze is not an ablation of gaze.

---

## Experiment A — measured gaze, 400 sentences

**Script**: `model_ZuCo_SST.py`  
**Table**: `ZuCo_SST_data/combined_sst_et_standard.csv`

| Knob | Value |
| --- | --- |
| Rows | 400 |
| Folds | 5, stratified on `sentiment_label`, `random_state=42` |
| Epochs per fold | 20 |
| Batch size | 16 (hard-coded in the `DataLoader` as well as the constant) |
| Checkpoint | none |
| Gaze columns | `nFixations, FFD, GPT, TRT, GD` |

Each fold trains a **fresh** encoder from Hub weights. There is no
validation split *inside* a fold and no early stopping. The fold metric
is computed once after epoch 20. The script then prints the mean of the
five fold scores.

Why CV and not `train.csv` / `valid.csv` / `test.csv`: a 40-row valid
set with 7 negatives is too noisy to pick hyperparameters, and a single
40-row test set is a coin flip at the accuracy deltas you should expect
from a 16-d gaze head.

### What to log (personal checklist)

For each `model_type`:

1. Mean ± std of fold accuracy and of **macro** F1 (macro is not in the
   script; compute it from stored preds if you re-run).
2. Per-class recall on the pooled out-of-fold predictions.
3. A gaze-only logistic baseline on the same five columns
   (`examples/toy_text_gaze_fusion.py` does this with hashed text).
4. Wall time and whether any fold hit the 20-epoch mark with a training
   loss still falling.

Chance on this set is not 33.3% if you always predict the majority
class: 140 / 400 = **35.0%** positive. A text encoder should be far
above that; a gaze-only probe should be near it unless duration is
leaking label information (for example if negative reviews are longer).

---

## Experiment B — projected gaze, full SST

**Script**: `model_full_SST.py`  
**Tables**: `SST_data/{train,valid,test}_full_sst.csv`

| Knob | Value |
| --- | --- |
| Train / valid / test | 9,482 / 1,185 / 1,186 |
| Epochs | 5 |
| Batch size | 256 |
| Checkpoint | `models/best_{model_type}_model.pth` |
| Selection rule | highest **validation accuracy** |
| Gaze columns | `nFix, FFD, GPT, TRT, GD` |

The comment above `best_val_acc` says “best F1”; the `if` compares
accuracy. Neutral is the smallest class (~19% train). Accuracy-based
selection can prefer a model that dumps neutrals into pos/neg.

After training, the script reloads the checkpoint and runs the test
loader. Read [known-issues.md](known-issues.md) before quoting that
test line: the loop currently keeps only the last test batch.

### Label prior

| Split | Neg | Neu | Pos | Majority prior |
| --- | ---: | ---: | ---: | ---: |
| Train | 3710 | 1833 | 3939 | 41.5% pos |
| Valid | 476 | 209 | 500 | 42.2% pos |
| Test | 463 | 199 | 524 | 44.2% pos |

Splits were drawn with `random_state=42` and no `stratify=` argument in
`SST_data/spilt.py`. The priors drifted. `examples/split_balance.py`
prints this table.

---

## Suggested ablation grid

Keep the script constants. Only change `model_type` (and, if you are
comparing fusion recipes, the gaze layer). A minimal honest grid:

1. `roberta` on ZuCo CV.
2. `roberta_eye_tracking` on ZuCo CV.
3. `bert` / `bert_eye_tracking` if you need an encoder ablation.
4. Repeat 1–2 on full SST **after** the test-loop fix.

Optional cheaper probes (no Hub weights):

- Gaze-only logistic regression on the five z-scored columns.
- Text-only hashed bag-of-words.
- Concat of the two (the toy example).
- Length-only (`SentLen` or word count) to see how much “gaze” is
  length.

---

## Seeds

| Location | Seed |
| --- | --- |
| `StratifiedKFold` | 42 |
| Both `spilt.py` files | 42 |
| `torch`, `numpy`, `random` | **unset** |

Fold membership is reproducible. Weight init and dropout are not. If
you publish a ±, set `torch.manual_seed` and `numpy.random.seed` at the
top of the trainer, and run at least three seeds on Experiment A.

---

## Compute notes

Experiment A is “small data, large model”: 400 sentences, 20 epochs, 5
folds, full fine-tune. That is the run that needs a GPU.

Experiment B is the opposite shape: ~10k sentences, 5 epochs, batch
256. Memory-bound more than epoch-bound. `max_length=128` padding is
fixed, so a 20-word review still costs 128 tokens.

The example suite is sized for a laptop CPU and should finish in
seconds to a couple of minutes (`toy_text_gaze_fusion.py` is the
slowest; it fits several logistic models on 400 rows).

---

## Personal result log

Fill this in when you re-run; the clone does not ship metrics.

| Setting | Acc | Weighted F1 | Macro F1 | Notes |
| --- | --- | --- | --- | --- |
| ZuCo RoBERTa | | | | Hub trainer, not re-run here |
| ZuCo RoBERTa + gaze | | | | Hub trainer, not re-run here |
| ZuCo majority (toy) | 0.350 ± 0.000 | 0.181 ± 0.000 | 0.173 ± 0.000 | always guess positive |
| ZuCo length-only (toy) | 0.335 ± 0.024 | 0.257 ± 0.015 | 0.248 ± 0.011 | word count |
| ZuCo gaze-only (toy) | 0.368 ± 0.030 | 0.354 ± 0.028 | 0.349 ± 0.026 | five z-scores |
| ZuCo text hashed (toy) | 0.455 ± 0.043 | 0.438 ± 0.056 | 0.433 ± 0.058 | 4096-d 1–2 grams |
| ZuCo fusion hashed (toy) | 0.450 ± 0.051 | 0.441 ± 0.052 | 0.436 ± 0.051 | text + gaze; ≈ text |
| Full SST RoBERTa | | | | after test-loop fix |
| Full SST RoBERTa + gaze | | | | after test-loop fix |

Toy numbers are 5-fold mean ± std from
`examples/toy_text_gaze_fusion.py` (seed 42), also saved in
`docs/sample-reports/toy_text_gaze_fusion.txt`. Gaze-only is barely
above majority; hashed fusion does not beat hashed text. That is a
linear-probe result, not a RoBERTa result.

Until those cells are filled, treat any claim about “gaze helps” as a
hypothesis, not a result.
