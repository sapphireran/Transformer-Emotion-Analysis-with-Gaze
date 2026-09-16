# Running experiments

Personal lab-notebook style: edit the header, run from the repo root,
write down `model_type` and the printed metrics. There is no CLI and
no experiment tracker.

## Environment

```bash
python3 -m pip install -r requirements.txt
```

You need a Hugging Face cache that can hold `bert-base-uncased` or
`roberta-base`. First run downloads several hundred MB.

Check the device line the scripts print (`Using device: cuda|cpu`).

## Minimal “are the files intact?” run

No extra pip packages beyond `numpy`:

```bash
python3 examples/02_schema_check.py
python3 examples/01_inspect_datasets.py
```

Exit code 0 from the schema checker means the committed headers,
row counts, and label sets still match what the docs describe.

## Ablation checklist (ZuCo 400)

For each of the four `model_type` values, run `model_ZuCo_SST.py`
and record the five-fold **mean** Acc / P / R / F1.

Suggested additional edits (one at a time):

| Knob | File header | Why |
| --- | --- | --- |
| `dataset_path` → `combined_sst_et_min_max.csv` | `model_ZuCo_SST.py` | scaling ablation |
| `num_epochs` 5 vs 20 | same | overfitting check |
| `hidden_layer_size` 8 / 16 / 32 | same | gaze capacity (should barely matter) |
| drop GPT or nFixations from the column list | `load` + `num_eye_tracking_features` | feature ablation |

Keep `random_state=42` on the KFold so folds stay aligned across
ablations. Training noise will still move the means; repeat if a
gap is < 1–2 points.

## Ablation checklist (full SST)

Same four `model_type`s in `model_full_SST.py`.

```bash
mkdir -p models
python3 model_full_SST.py
```

Record **validation** Acc / F1 each epoch and the best checkpoint
path. Ignore the printed **test** line until the overwrite bug in
[06-training-loops.md](06-training-loops.md) is fixed, or evaluate
the saved checkpoint with a short script that `extend`s predictions.

Because gaze is transferred, a win for `*_eye_tracking` here does
not automatically replicate on the 400.

## What to log (minimum)

```
date
git commit
script (ZuCo vs full SST)
model_type
epochs, batch, lr
device
val / fold metrics (copy-paste the printout)
notes (OOM? changed columns? first download?)
```

A `notes/` directory is intentionally not added; keep a personal
log wherever you already write. Do not commit huge `.pth` files
(see `.gitignore`).

## Expected artifacts

| Run | Writes |
| --- | --- |
| examples | `examples/sample_outputs/*.txt` (and printed tables) |
| `model_ZuCo_SST.py` | stdout only |
| `model_full_SST.py` | `models/best_{model_type}_model.pth` plus stdout |
| `read_ZuCo_mat.py` | `et_csv_data/{1-12}_SR.csv` (path as coded) |
| `get_average_sentence_level.py` | three average/scaled CSVs |
| `convert_full_SST.py` | `ZuCo_SST_data/ssts_ZuCo.csv` |

## GPU memory knobs

If you OOM on full SST:

1. Lower `batch_size` (256 → 32 or 16).
2. Keep `max_length=128` or drop it to 64 (mean length is ~19 tokens;
   64 still covers the tail except pathological rows).
3. Do not switch to `bert-large` without gradient checkpointing.

ZuCo batch 16 at 128 tokens is light.

## Re-running preprocessing

Only do this if you have the MATLAB sources and have patched the
Windows paths. Procedure is in
[04-preprocessing-pipeline.md](04-preprocessing-pipeline.md). After
regenerating, run `examples/02_schema_check.py` and diff the combined
standard CSV against `main`.

## Plots already in `result/`

```
result/provo_data_scatter_hist_plots.png
result/test_data_scatter_hist_plots.png
result/train_data_scatter_hist_plots.png
```

These are exploratory gaze-distribution figures, not model-quality
plots. They are not produced by any script still in the tree.

## When not to train

Skip the transformer scripts if you only changed documentation.
The example suite is the regression check for CSV shape and for
the “does gaze have any linear signal?” question.
