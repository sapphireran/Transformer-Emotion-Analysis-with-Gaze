# Transformer Emotion Analysis with Gaze

Personal research workspace: **three-class sentiment** on movie-review sentences, with an optional **eye-tracking** branch fused into BERT or RoBERTa.

This clone keeps the original training scripts and CSVs, and adds a docs/examples layer that you can run with the Python standard library — no GPU, no Hugging Face download.

```
text ──► BERT / RoBERTa pooler (768) ──┐
                                       ├─► concat ──► dropout ──► Linear(784 → 3)
gaze ──► Linear(5 → 16) ───────────────┘
```

Labels: `0` negative, `1` neutral, `2` positive.

## Why two data tracks?

| Track | Rows | Gaze source | Training script |
| --- | ---: | --- | --- |
| ZuCo-SST | 400 | averaged human recordings (12 readers) | `model_ZuCo_SST.py` (5-fold) |
| Full SST | 11,853 | projected / predicted features | `model_full_SST.py` (80/10/10) |

Use ZuCo when you care about real reading behavior. Use full SST when you care about scale. Do not interpret full-SST `TRT` as a human measurement.

## Repository map

```
docs/                      architecture, datasets, gaze glossary, pipeline, training
examples/                  stdlib inspectors, schema checks, fusion demo
examples/lib/              reusable helpers (CSV, stats, tiny linear algebra)
tests/                     unittest coverage for examples/lib
model_ZuCo_SST.py          RoBERTa/BERT ± gaze, stratified 5-fold
model_full_SST.py          same models, fixed SST split + checkpoint
utils_ZuCo.py              MATLAB → sentence/word tables
ZuCo_SST_data/             400 labeled sentences + scaled gaze
ZuCo_et_csv_data/          per-subject and averaged ET
SST_data/                  11k labeled sentences + projected gaze
gaze_prediction/data/      word-level predicted ET
```

## Run the examples (no extra packages)

```bash
python3 examples/inspect_datasets.py
python3 examples/schema_check.py
python3 examples/label_balance.py
python3 examples/split_audit.py
python3 examples/gaze_feature_report.py
python3 examples/fusion_forward_demo.py
python3 examples/word_level_preview.py
python3 -m unittest discover -s tests -v
```

Write markdown snapshots:

```bash
python3 examples/generate_markdown_tables.py --write
```

## Train the transformer models

```bash
pip install -r requirements.txt
mkdir -p models
python model_ZuCo_SST.py      # small, 5-fold
python model_full_SST.py      # large, needs a GPU to be pleasant
```

Set `model_type` at the top of either script to `bert`, `roberta`, `bert_eye_tracking`, or `roberta_eye_tracking`.

## Docs

- [Architecture](docs/architecture.md)
- [Datasets](docs/datasets.md)
- [Gaze features](docs/gaze-features.md)
- [Data pipeline](docs/data-pipeline.md)
- [Training](docs/training.md)
- [Examples](docs/examples.md)
- [Notes and gotchas](docs/notes-and-gotchas.md)

Read the gotchas page before quoting a test score from `model_full_SST.py`. The test loop currently keeps only the last batch.

## Personal scope

This is a personal repo (`sapphireran/Transformer-Emotion-Analysis-with-Gaze`). It is not a company codebase. Do not add internal data, private reviews, or work-only models here.

ZuCo and SST are third-party corpora — cite the original authors if you publish derived results.
