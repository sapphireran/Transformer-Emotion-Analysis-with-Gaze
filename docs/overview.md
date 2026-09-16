# Project overview

Personal working notes for **Transformer Emotion Analysis with Gaze**.

The experiment is a late-fusion classifier: a frozen-or-finetuned sentence encoder (BERT-base or RoBERTa-base) plus a tiny multilayer perceptron over five sentence-level eye-tracking channels. The label space is 3-way sentiment on movie-review sentences.

This note is the table of contents for the rest of `docs/`. It is written against the scripts that already live in the repo, not against an idealized pipeline.

## Questions the repo is trying to answer

1. **Does observed gaze help** on the 400 ZuCo Task 1 sentences that already have both gold sentiment and 12-reader ET?
2. **Does predicted or transferred gaze help** when the same five channels are attached to the much larger SST split (~11.8k sentences)?
3. **How much of any gain is just length / difficulty?** Gaze duration and fixation count are correlated with sentence length and with word-level surprisal. The example baselines in `examples/` exist to keep that confound visible.

The original training scripts answer (1) and (2) with transformers. The new example library answers a cheaper version of (3): gaze-only logistic regression, majority-class baselines, and split-leakage checks.

## Two experimental tracks

```
                    ZuCo .mat (Task 1, 12 subjects)
                              |
                              v
                    sentence / word CSVs
                              |
              +---------------+---------------+
              |                               |
              v                               v
     400 SST sentences                  word averages
     + observed ET                      + predicted gaze
              |                               |
              v                               v
     model_ZuCo_SST.py                 SST_data/*_full_sst.csv
     5-fold CV, 20 epochs              model_full_SST.py
                                       train/valid/test, 5 epochs
```

### Track A — observed gaze (ZuCo SST)

- **Size:** 400 sentences (`ZuCo_SST_data/combined_sst_et_standard.csv`).
- **Labels:** 3-way, from the ZuCo SST dump (`ssts_ZuCo.csv`).
- **Gaze:** subject-averaged, then standardized (or min-max in the sibling file).
- **Protocol:** `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)` inside `model_ZuCo_SST.py`.
- **Why it exists:** this is the only track where the ET channels are *measured*, not imputed.

There is also a convenience 80/10/10 split of the same 400 rows (`train.csv` / `valid.csv` / `test.csv`, 320 / 40 / 40). The transformer script does **not** use that split; it re-folds the combined table. The split is still useful for the example baselines.

### Track B — full SST with attached gaze

- **Size:** 11,853 sentences in `SST_data/stts_all_sentence_level.csv`.
- **Fused table:** `SST_data/combined_full_sst_et.csv`, then `train_full_sst.csv` (9,482), `valid_full_sst.csv` (1,185), `test_full_sst.csv` (1,186).
- **Gaze columns:** `nFix`, `GD`, `TRT`, `FFD`, `GPT` (note the shorter `nFix` name versus ZuCo's `nFixations`).
- **Protocol:** single train/valid/test run in `model_full_SST.py`, checkpoint on validation accuracy.

Word-level predicted gaze for this track lives under `gaze_prediction/data/` (`prediction_test.csv`, `prediction_test_v2.csv`, plus a PROVO reference table). Those files are the source of the sentence-level numbers after aggregation.

## What "fusion" means here

Both `EyeTrackingModel` classes (they are duplicated in the two training scripts) do the same thing:

1. Run `BertModel` or `RobertaModel` on `input_ids` / `attention_mask` (max length 128).
2. Take `pooler_output` (768-d).
3. Project the 5 ET features with a linear layer to 16-d.
4. Concatenate to 784-d, apply dropout 0.1, classify into 3 logits.

There is no cross-attention, no word-aligned gaze, and no uncertainty estimate. Sentence-level means are treated as a side channel. That is a deliberate first cut: it is easy to ablate and it matches the columns that are already in the CSVs.

A dependency-free copy of the tensor math is in `examples/gazekit/fusion.py`.

## What this repo is not

- It is not a packaged training library. Hyperparameters are module-level constants in the two `model_*.py` files.
- It does not ship ZuCo `.mat` files. `read_ZuCo_mat.py` expects a local `ZuCo_mat_data/` tree that is not in git.
- It does not evaluate calibrated probabilities or significance tests. Reported numbers from the original scripts are accuracy / weighted P / R / F1.
- The full-SST test loop currently **overwrites** `all_preds` each batch instead of extending it. Treat printed test metrics from `model_full_SST.py` as last-batch-only until that script is patched. See [limitations.md](limitations.md).

## How to read the rest of the notes

| Doc | Read it when you need… |
| --- | --- |
| [datasets.md](datasets.md) | Exact row counts, paths, and which script wrote which file |
| [data-dictionary.md](data-dictionary.md) | Column-by-column definitions |
| [gaze-features.md](gaze-features.md) | What FFD / GPT / TRT / GD actually measure |
| [preprocessing.md](preprocessing.md) | Scaling, NaN policy, subject-level exclusions |
| [model-architecture.md](model-architecture.md) | Layer sizes, forward pass, tokenizer choices |
| [training-and-evaluation.md](training-and-evaluation.md) | Epochs, batch sizes, metrics, CV vs holdout |
| [reproduction.md](reproduction.md) | Commands to regenerate tables and run training |
| [limitations.md](limitations.md) | Known bugs, confounds, and next personal experiments |
| [example-results.md](example-results.md) | Actual numbers from the example scripts on this checkout |

The example walkthrough is [../examples/README.md](../examples/README.md).
