# Transformer Emotion Analysis with Gaze

Personal research code for **three-class sentiment classification** that concatenates a transformer sentence representation with a small eye-tracking (gaze) feature vector.

The working hypothesis is straightforward: first-pass and rereading measures from human reading (ZuCo) carry extra signal about how difficult, surprising, or affectively loaded a sentence is. If that signal is real, a BERT or RoBERTa classifier that sees both text and gaze should beat a text-only baseline on the same labels.

This repository is a personal notebook of that experiment. It is **not** a company project and does not contain proprietary data pipelines.

## What is here

| Path | Role |
| --- | --- |
| `model_ZuCo_SST.py` | 5-fold stratified CV on the 400 ZuCo sentiment sentences, with or without gaze |
| `model_full_SST.py` | Train / valid / test on the full Stanford Sentiment Treebank (~11.8k sentences) plus predicted gaze |
| `utils_ZuCo.py` | MATLAB → pandas transformer for ZuCo sentence- and word-level eye tracking |
| `read_ZuCo_mat.py` | Export one CSV per subject from ZuCo Task 1 (sentiment reading) |
| `get_average_sentence_level.py` | Subject-average + min-max / standard scaling |
| `convert_full_SST.py` | Folder of labeled `.txt` files → `ssts_ZuCo.csv` |
| `SST_data/` | Full SST text, predicted gaze columns, 80/10/10 split |
| `ZuCo_SST_data/` | 400 labeled ZuCo sentences joined to averaged gaze |
| `ZuCo_et_csv_data/` | Per-subject sentence- and word-level gaze |
| `gaze_prediction/data/` | PROVO-style word gaze and predicted SST word gaze |
| `docs/` | Personal notes: datasets, features, architecture, pipeline, known issues |
| `examples/` | Runnable inspections and a tiny gaze-fusion demo (no GPU required) |

## Labels

All classifiers use the same three-way mapping:

| Folder / string | Integer |
| --- | ---: |
| `NEGATIVE` | 0 |
| `NEUTRAL` | 1 |
| `POSITIVE` | 2 |

On the **ZuCo-aligned** set the counts are 123 / 137 / 140 (negative / neutral / positive). On the **full SST** set they are 4649 / 2241 / 4963.

## Gaze features used at train time

The fusion models read five sentence-level numbers:

`nFix` / `nFixations`, `FFD`, `GPT`, `TRT`, `GD`

A 16-unit linear layer projects those five values. The projected vector is concatenated with the transformer `pooler_output` (768-d for `bert-base` / `roberta-base`), dropped out at 0.1, and classified into three logits.

See [docs/eye-tracking-features.md](docs/eye-tracking-features.md) for definitions of FFD, GPT, TRT, GD, SFD, pupil size, and omission rate.

## Two experimental settings

### 1. Native ZuCo sentiment (small, real gaze)

- File: `ZuCo_SST_data/combined_sst_et_standard.csv` (or the min-max sibling)
- 400 sentences that subjects actually read in ZuCo Task 1
- Gaze is the mean across up to 12 readers, then standard- or min-max-scaled
- Script: `model_ZuCo_SST.py` — `StratifiedKFold(n_splits=5)`, 20 epochs, batch size 16

### 2. Full SST (large, predicted gaze)

- Files: `SST_data/train_full_sst.csv`, `valid_full_sst.csv`, `test_full_sst.csv`
- 9482 / 1185 / 1186 sentences
- Gaze columns are **not** human recordings of those SST sentences; they are predicted or transferred values
- Script: `model_full_SST.py` — 5 epochs, batch size 256, checkpoint on validation accuracy

Details and caveats: [docs/experiments.md](docs/experiments.md).

## Model switch

Both training scripts share the same `model_type` string:

```text
bert | roberta | bert_eye_tracking | roberta_eye_tracking
```

Text-only variants use Hugging Face `*ForSequenceClassification`. Gaze variants use the local `EyeTrackingModel` wrapper around `BertModel` / `RobertaModel`.

Default in both scripts is `roberta_eye_tracking`.

## Quick start (docs and examples)

The example scripts only need the Python standard library. From the repo root:

```bash
python3 examples/inspect_datasets.py
python3 examples/label_distribution.py
python3 examples/feature_stats.py
python3 examples/schema_validate.py
python3 examples/gaze_fusion_demo.py
python3 examples/sentence_gaze_join.py
```

Each script prints a short report. Together they are a walkthrough of the files this repo actually ships.

## Training (optional, needs GPU + Hugging Face)

```bash
python3 -m pip install -r requirements.txt

# Native ZuCo, 5-fold CV
python3 model_ZuCo_SST.py

# Full SST split
mkdir -p models
python3 model_full_SST.py
```

`model_full_SST.py` writes `models/best_{model_type}_model.pth`. Raw ZuCo `.mat` files are **not** in this clone; `read_ZuCo_mat.py` expects a local `ZuCo_mat_data/` directory if you want to regenerate the CSVs.

## Documentation map

1. [Research overview](docs/overview.md)
2. [Datasets and file inventory](docs/datasets.md)
3. [Eye-tracking feature glossary](docs/eye-tracking-features.md)
4. [Model architecture](docs/model-architecture.md)
5. [Data pipeline](docs/data-pipeline.md)
6. [Experiments and known issues](docs/experiments.md)
7. [Reproduction notes](docs/reproduction.md)
8. [References](docs/references.md)
9. [Examples](examples/README.md)

## Status

Personal research snapshot. The training loops work as research scripts, not as a packaged library. Known evaluation bugs and path quirks are written down in `docs/experiments.md` instead of being silently “fixed” in the narrative.

## License / data

ZuCo and SST are third-party academic corpora. Redistribute them only under their original terms. Predicted gaze files in this repo are derived artifacts from the personal experiment, not official ZuCo releases.
