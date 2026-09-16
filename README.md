# Transformer Emotion Analysis with Gaze

Personal research code for combining **transformer language models** with **human eye-tracking features** for three-class sentiment analysis.

The project joins two data sources:

- **Stanford Sentiment Treebank (SST)** movie-review sentences with negative / neutral / positive labels.
- **Zurich Cognitive Language Processing Corpus (ZuCo)** eye-tracking recordings from twelve readers on a subset of those sentences.

Two experimental tracks share the same fusion idea:

| Track | Script | Gaze source | Evaluation |
| --- | --- | --- | --- |
| ZuCo SST | `model_ZuCo_SST.py` | Real subject-averaged eye movements | 5-fold stratified CV |
| Full SST | `model_full_SST.py` | Projected / predicted gaze on the full SST split | held-out train / valid / test |

Text-only BERT and RoBERTa baselines sit next to **BERT + gaze** and **RoBERTa + gaze** variants. The fusion models project five reading-time features through a small linear layer, concatenate that vector with the transformer pooler output, and classify.

This repository is personal research. It is not affiliated with a company product.

## Why gaze?

Reading-time features are a cheap, continuous signal of how hard a sentence is to process. First-pass measures such as first fixation duration (FFD) and gaze duration (GD) tend to reflect lexical access. Later measures such as go-past time (GPT) and total reading time (TRT) pick up re-reading and integration. The hypothesis under test is that those signals still help a pretrained transformer decide sentiment when they are fused at the sentence level.

## Repository layout

```text
.
├── model_ZuCo_SST.py          # 5-fold CV on ZuCo-aligned SST
├── model_full_SST.py          # train / valid / test on full SST + gaze
├── utils_ZuCo.py              # MATLAB -> DataFrame transformer
├── read_ZuCo_mat.py           # export per-subject sentence CSVs
├── convert_full_SST.py        # folder of .txt reviews -> labeled CSV
├── get_average_sentence_level.py
├── ZuCo_et_csv_data/          # raw and averaged eye-tracking tables
├── ZuCo_SST_data/             # ZuCo sentences joined with gaze
├── SST_data/                  # full SST with projected gaze
├── gaze_prediction/           # word-level predicted gaze tables
├── docs/                      # design notes and reproduction notes
└── examples/                  # CPU-only inspection and baseline scripts
```

## Gaze features

The fusion heads always consume five sentence-level features:

| Feature | Meaning |
| --- | --- |
| `nFix` / `nFixations` | Mean number of fixations on the words that were looked at |
| `FFD` | First fixation duration |
| `GPT` | Go-past / regression-path time |
| `TRT` | Total reading time |
| `GD` | Gaze duration (first-pass dwell) |

ZuCo tables also keep `omissionRate`, `meanPupilSize`, `SFD` (single fixation duration), and sentence length. Those extra columns are useful for analysis but are not fed into the current classifier heads. See [docs/gaze-features.md](docs/gaze-features.md).

Sentiment labels are stored as integers:

- `0` negative
- `1` neutral
- `2` positive

## Models

`EyeTrackingModel` wraps `bert-base-uncased` or `roberta-base`:

1. Encode the sentence (`max_length=128`).
2. Take `pooler_output` (768-d).
3. Project the five gaze features to a 16-d hidden vector.
4. Concatenate, apply dropout `0.1`, and classify into three logits.

Text-only runs use `BertForSequenceClassification` or `RobertaForSequenceClassification` and ignore the gaze columns.

Default hyperparameters (from the training scripts):

| Setting | ZuCo SST | Full SST |
| --- | --- | --- |
| Epochs | 20 | 5 |
| Batch size | 16 | 256 |
| Learning rate | `5e-5` | `5e-5` |
| Optimizer | Adam | Adam |
| Fusion hidden size | 16 | 16 |

Architecture notes and a known test-loop bug live in [docs/architecture.md](docs/architecture.md).

## Data pipeline

High-level flow:

```text
ZuCo .mat (12 subjects)
        │
        ▼
  utils_ZuCo.DataTransformer
        │
        ▼
  per-subject CSVs  ──►  subject-average + scale
        │
        ▼
  join SST sentence text + sentiment
        │
        ├──► ZuCo SST  (~400 labeled rows)
        │
        └──► word-level averages ──► gaze projection onto full SST
                                              │
                                              ▼
                                    train / valid / test CSVs
```

Details, file inventories, and scaling conventions are in [docs/data-pipeline.md](docs/data-pipeline.md) and [docs/dataset-inventory.md](docs/dataset-inventory.md).

## Quick start (docs / examples)

The example scripts are CPU-only and operate on the CSVs already in the repo. They do not download transformers or train GPU models.

```bash
python3 -m pip install -r requirements.txt
python3 examples/inspect_datasets.py
python3 examples/join_zuco_sst.py
python3 examples/subject_agreement.py
python3 examples/gaze_feature_stats.py
python3 examples/sentiment_baselines.py
python3 examples/fusion_toy.py
python3 examples/word_level_gaze.py
```

Training the transformer models needs PyTorch, Hugging Face `transformers` / `datasets`, and a GPU for any realistic batch size. Reproduction notes are in [docs/reproduction.md](docs/reproduction.md).

## Documentation

- [Architecture](docs/architecture.md) — fusion head, training loops, pitfalls
- [Data pipeline](docs/data-pipeline.md) — ZuCo export, averaging, SST joins
- [Gaze features](docs/gaze-features.md) — measure glossary and scaling
- [Experiments](docs/experiments.md) — tracks, metrics, what to log
- [Reproduction](docs/reproduction.md) — environment and run commands
- [Dataset inventory](docs/dataset-inventory.md) — every CSV and its schema
- [Examples](examples/README.md) — CPU walkthroughs
- [Examples walkthrough](docs/examples-walkthrough.md) — how to read the printed tables
- [Baseline notes](docs/baseline-notes.md) — numbers from this checkout

## Status of this personal repo

The original training scripts are research snapshots: hardcoded paths, mixed English / Chinese comments, and a few known issues (Windows-style MATLAB paths, a test-set overwrite bug in `model_full_SST.py`, and `spilt.py` filenames). Documentation and examples in `docs/` and `examples/` describe the code as it is, not as a cleaned product.
