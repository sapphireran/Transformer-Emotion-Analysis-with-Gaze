# Training guide

This is a personal reproduction note for the two training scripts. It does not claim published scores. Run times below assume a single GPU for the transformer jobs and a CPU for the example demo.

## Environment

Python 3.10+ is enough for the example scripts (stdlib only). The training scripts also need:

```
pip install -r requirements.txt
```

Weights download from Hugging Face on first run (`bert-base-uncased` or `roberta-base`). Cache them if you are offline later.

Device selection is already in both scripts:

```python
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
```

CPU training of `roberta-base` on 11k SST rows will be slow. The 400-row ZuCo job is the better CPU smoke test.

Create the checkpoint directory before the full-SST script:

```
mkdir -p models
```

`torch.save` will fail if `models/` is missing.

## Quick commands

```bash
# 400-row ZuCo, 5-fold, ~20 epochs per fold
python model_ZuCo_SST.py

# 11k-row SST, train/valid/test, 5 epochs
python model_full_SST.py
```

Change `model_type` at the top of the script:

```python
model_type = 'roberta_eye_tracking'  # or bert, roberta, bert_eye_tracking
```

There is no CLI. Treat the file header as the config.

## Recommended comparison grid

For a fair "does gaze help?" check, keep tokenizer, seed, epochs, and batch size fixed and only flip the fusion flag:

| Run | `model_type` | Gaze used? |
| --- | --- | --- |
| A | `roberta` | no |
| B | `roberta_eye_tracking` | yes |
| C | `bert` | no |
| D | `bert_eye_tracking` | yes |

On ZuCo, report the **mean ± std** of the five folds, not a single split. On full SST, report valid and test, and be aware of the test-loop bug documented in `docs/notes-and-gotchas.md`.

A cheap non-transformer baseline (gaze-only softmax, text-hash softmax, fused softmax) is `examples/fusion_forward_demo.py`. Use it to sanity-check that the five numeric columns carry *some* class signal before you spend a GPU hour.

## Hyperparameters (as checked in)

| Knob | ZuCo script | Full SST script |
| --- | ---: | ---: |
| `num_eye_tracking_features` | 5 | 5 |
| `hidden_layer_size` | 16 | 16 |
| `num_labels` | 3 | 3 |
| `num_epochs` | 20 | 5 |
| `learning_rate` | 5e-5 | 5e-5 |
| `batch_size` | 16 | 256 |
| folds / split | 5-fold stratified | fixed 80/10/10 |
| checkpoint | none | best valid accuracy |

Adam is used on **all** parameters, including the full encoder. This is fine-tuning, not a frozen-encoder probe.

## Metrics

`calculate_metrics` uses scikit-learn with `average='weighted'` for precision, recall, and F1. Weighted F1 tracks accuracy when classes are balanced and hides minority-class failures when they are not. On full SST, also compute **macro** F1 (the example metrics helper does both).

## What "good" looks like before you trust a run

1. Text-only RoBERTa on SST should beat a majority baseline (~42% positive). If it does not, tokenization or labels are wrong.
2. Fusion should not *crash* when gaze columns are permuted — if accuracy is unchanged after shuffling gaze in the valid set, the gaze branch is unused or dead.
3. On ZuCo, fold scores should move by a few points, not by 30. A 30-point swing usually means a fold collapsed to one class.
4. Training loss should trend down in the tqdm postfix. If it does not after two epochs, check the learning rate and that labels are `0/1/2`, not `-1/0/1`.

## Shuffle test for the gaze branch

A minimal ablation you can add without changing architecture:

```python
# after loading the valid/test tensor
perm = torch.randperm(eye_tracking_features.size(0))
shuffled = eye_tracking_features[perm]
```

If shuffled-gaze accuracy ≈ real-gaze accuracy, the linear gaze layer is not contributing. If shuffled-gaze is *worse*, the model was using the real values.

## Resource notes

- Batch 256 × 128 tokens × RoBERTa is GPU-friendly and CPU-hostile.
- 5-fold × 20 epochs × 400 rows × batch 16 is a small job.
- Saving `state_dict` only (as the SST script does) is enough to reload with `get_model(...)`.
- Do not commit `.pth` files. `.gitignore` already excludes `models/` and `*.pth`.
