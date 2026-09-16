# Transformer Emotion Analysis with Gaze

Personal research notes and code for fusing **eye-tracking (ET) features** with **transformer sentence encoders** on Stanford Sentiment Treebank (SST) movie reviews. The small, fully observed gaze set comes from **ZuCo** Task 1 (normal reading of SST-style reviews). A larger SST split uses the same five gaze channels after they have been projected onto the full treebank.

This repository started as a collection of one-off training and conversion scripts. This tree adds a documented data map, feature glossary, architecture notes, and a small `examples/gazekit` library so the CSVs can be inspected without launching a GPU training run.

## Why gaze + sentiment

Reading-time measures are a cheap behavioral signal of processing difficulty, reanalysis, and (noisily) arousal. The working hypothesis in this project is:

1. Sentiment-bearing movie reviews already exist in ZuCo Task 1 with word- and sentence-level fixation statistics for 12 readers.
2. A transformer (BERT or RoBERTa) already encodes lexical polarity well.
3. Concatenating a compact projection of five classic ET channels onto the pooled transformer state may help on the small ZuCo-labeled set, and is at least a clean ablation against text-only baselines.

The five channels used by the fusion models are **nFixations**, **FFD**, **GPT**, **TRT**, and **GD**. See [docs/gaze-features.md](docs/gaze-features.md).

## Repository map

| Path | Role |
| --- | --- |
| `utils_ZuCo.py` | `DataTransformer` that reads ZuCo `.mat` files and writes sentence- or word-level tables |
| `read_ZuCo_mat.py` | Batch export of 12 Task 1 subjects to CSV |
| `get_average_sentence_level.py` | Subject-average + min-max / standard scaling |
| `ZuCo_et_csv_data/` | Per-subject sentence ET, averages, and word-level tables |
| `ZuCo_SST_data/` | 400 review sentences with gold 3-way labels + fused ET |
| `SST_data/` | Full SST sentence list and an 80/10/10 ET-augmented split |
| `model_ZuCo_SST.py` | 5-fold stratified CV on the 400-sentence ZuCo set |
| `model_full_SST.py` | Train / valid / test run on the large SST split |
| `gaze_prediction/data/` | Word-level predicted gaze (PROVO-style and SST-scale) |
| `result/` | Legacy scatter / histogram plots of gaze channels |
| `docs/` | Personal research notes (start at [docs/overview.md](docs/overview.md)) |
| `examples/` | Runnable inspection / baseline / fusion-shape examples |

## Quick start (examples only)

The example library talks to the checked-in CSVs. It does **not** download BERT/RoBERTa weights.

```bash
python3 -m pip install -r requirements.txt
python3 -m pytest
python3 examples/scripts/01_explore_zuco_sst.py
python3 examples/scripts/03_gaze_only_baseline.py
```

Full transformer training still uses the original scripts and `requirements-train.txt`. Those runs expect a CUDA GPU for anything beyond a smoke test.

## Label convention

All fused tables in this repo use integer sentiment labels:

| Integer | Class | Source folder / SST tag |
| --- | --- | --- |
| `0` | negative | `NEGATIVE` |
| `1` | neutral | `NEUTRAL` |
| `2` | positive | `POSITIVE` |

## Model variants

`model_type` in both training scripts is one of:

- `bert` / `roberta` — Hugging Face sequence classification, text only
- `bert_eye_tracking` / `roberta_eye_tracking` — pooled transformer state concatenated with a 5→16 linear gaze projection, then a 3-way classifier

Details and tensor shapes live in [docs/model-architecture.md](docs/model-architecture.md).

## Personal scope

This is a personal research archive (datasets, conversion scripts, and experiment notes). It is not a product codebase. The `examples/` package is intentionally small and dependency-light so later notes can be regenerated from the CSVs without retraining.

## License / data

ZuCo and SST are third-party academic datasets. Use them under their original licenses. The scripts and notes in this fork are personal study material.
