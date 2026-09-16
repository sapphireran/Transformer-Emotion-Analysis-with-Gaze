# Experiments (as coded)

Hyperparameters are literals at the top of each training file. There is no
CLI. Changing an experiment means editing the file.

## Track A — human gaze, 400 SST–ZuCo sentences

**Script:** `model_ZuCo_SST.py`  
**Table:** `ZuCo_SST_data/combined_sst_et_standard.csv`

| Knob | Value in file |
| --- | --- |
| `model_type` | `'roberta_eye_tracking'` |
| `num_labels` | 3 |
| `num_eye_tracking_features` | 5 |
| `hidden_layer_size` | 16 |
| `num_epochs` | 20 |
| `learning_rate` | `5e-5` |
| `batch_size` (config) | 16 |
| Actual `DataLoader` batch | **16, hard-coded** in `DataLoader(...)` (the `batch_size` variable is unused) |
| CV | `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)` |
| Checkpoint | none |

Each fold trains 20 epochs, then evaluates **once** on the fold's test
indices (no per-epoch validation inside the fold). Printed lines:

```
Validation Acc / P / R / F1   # per fold
Average Validation Acc / P / R / F1
```

“Validation” here is the CV hold-out, not `ZuCo_SST_data/valid.csv`.

### Intended comparisons on Track A

Edit `model_type` and re-run (four cells):

| `model_type` | Gaze used? |
| --- | --- |
| `bert` | no |
| `bert_eye_tracking` | yes |
| `roberta` | no |
| `roberta_eye_tracking` | yes |

Keep seed 42 and the same CSV so folds line up. Because each run
re-tokenizes and re-seeds only the KFold object (PyTorch / CUDA seeds are
**not** set), expect small numeric drift.

### Label counts

Run `examples/label_and_gaze_summary.py` for the exact histogram on the
combined table. The 80/10/10 files (`train.csv` 321, `valid.csv` 40,
`test.csv` 40) are a convenience split if you want a single hold-out
instead of CV; they are **not** stratified in `spilt.py`.

## Track B — predicted gaze, full SST

**Script:** `model_full_SST.py`  
**Tables:** `SST_data/train_full_sst.csv`, `valid_full_sst.csv`, `test_full_sst.csv`

| Knob | Value in file |
| --- | --- |
| `model_type` | `'roberta_eye_tracking'` |
| `num_epochs` | 5 |
| `learning_rate` | `5e-5` |
| `batch_size` | **256** |
| Device | CUDA if `torch.cuda.is_available()` else CPU |
| Selection | highest **validation accuracy** |
| Checkpoint path | `models/best_{model_type}_model.pth` |

Create `models/` before training or `torch.save` will fail.

After the epoch loop the script instantiates a **new** model, loads the
best state dict, and runs the test loader. See [known-pitfalls.md](known-pitfalls.md)
for the test-loop bug (`all_preds = ...` instead of `extend`): **test
metrics currently reflect the last batch only.** Valid metrics per epoch
are accumulated correctly.

### Intended comparisons on Track B

Same four `model_type` values. Predicted gaze can be **pure noise** relative
to human ZuCo; a text-only RoBERTa that beats fusion on Track B is an
informative negative result, not a failed run.

## Shared implementation details

- Tokenizer max length 128 for both tracks.
- Dropout 0.1 on the concatenated vector only (fusion).
- Adam on the full encoder.
- Weighted P/R/F1 via sklearn.
- Fusion CE uses `logits.view(-1, 3)` — `num_labels` is assumed 3.

## Compute notes (personal)

Track A, 5 folds × 20 epochs × ~400 sentences × batch 16 is a small
encoder fine-tune (minutes on a single GPU, slower on CPU).

Track B, 5 epochs × ~9.5k sentences × batch 256 is the heavier job.
`batch_size=256` at sequence length 128 may OOM on 8–12 GB cards with
RoBERTa-base in fp32; drop to 32/64 if needed. There is no
`gradient_accumulation_steps`.

## What the example scripts add

They are **not** a replacement for the two training files. They answer
questions you want before spending a GPU hour:

| Script | Question |
| --- | --- |
| `inspect_sentence_gaze.py` | Are raw / scaled tables finite and same length? |
| `inspect_word_gaze.py` | Skip rate, word-length vs nFixations |
| `subject_coverage.py` | Who is missing sentences? |
| `label_and_gaze_summary.py` | Class balance; gaze means by label |
| `sst_split_sanity.py` | Disjoint splits? ID space overlap with ZuCo? |
| `toy_fusion_forward.py` | Do concat shapes match 768+16? |
| `gaze_only_baseline.py` | Is gaze linearly associated with label at all? |
| `metrics_example.py` | Weighted vs macro vs per-class F1 |

Record GPU training numbers in a personal log (not checked in) next to
these baselines so “fusion helped” is compared to a chance/linear floor.
