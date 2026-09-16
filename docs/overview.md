# Overview

Sentiment classifiers see words. Readers also leave a **when and how long
they looked** trace. This project asks whether those traces help a transformer
decide if a movie review is negative, neutral, or positive.

## Why late fusion

Eye-tracking features are a different modality from subword tokens:

- They are **low dimensional** (five numbers after sentence pooling).
- They are **aligned to the same sentence** the model already encodes.
- They are **expensive to collect**, so the large SST track has to *project*
  them instead of measuring them.

The architecture therefore keeps a pretrained BERT or RoBERTa encoder and
adds a tiny linear map over the gaze vector. The two representations meet
only at the classifier:

```
sentence ──► BERT / RoBERTa ──► pooler_output (768)
                                      │
                                      ▼
gaze [5] ──► Linear(5, 16) ──► concat ──► Dropout ──► Linear(*, 3)
```

That is the `EyeTrackingModel` class duplicated in `model_ZuCo_SST.py` and
`model_full_SST.py`. Text-only runs skip the gaze branch and use the stock
`*ForSequenceClassification` heads.

A NumPy walkthrough of the same shapes is
`examples/fusion_architecture_demo.py`.

## Two tracks, one label space

### 1. ZuCo-SST (real gaze, small *n*)

ZuCo Task 1 had participants read movie-review sentences while an eye tracker
ran. After converting the MATLAB structs and averaging the 12 subjects, each
of **400** SST-domain sentences has a real ET signature. The corresponding
sentiment labels live in `ZuCo_SST_data/ssts_ZuCo.csv`.

`model_ZuCo_SST.py` does **not** use the 320/40/40 CSV split for reporting.
It reloads `combined_sst_et_standard.csv` and runs **StratifiedKFold(5)** so
every sentence is scored once.

### 2. Full SST (projected gaze, large *n*)

Full SST is two orders of magnitude larger and has no headset. Word-level
gaze is predicted (see `gaze_prediction/data/`), then pooled to a sentence
vector (`nFix`, `GD`, `TRT`, `FFD`, `GPT`). `model_full_SST.py` trains on
`SST_data/train_full_sst.csv`, checkpoints on validation accuracy, and
evaluates the held-out test file.

The 80/10/10 files were produced by `SST_data/spilt.py` with
`random_state=42`. Neutral reviews are under-represented (~19%), so weighted
metrics matter.

## What "emotion analysis" means here

The title says emotion; the labels are **SST polarity**. There is no Ekman
category head, no VAD regression, and no EEG branch in the training scripts
even though ZuCo recorded EEG. Extending the fusion layer to EEG bands would
be a separate experiment.

## What the new tree is for

| Path | Role |
| --- | --- |
| `docs/` | Human-readable map of features, files, and training choices |
| `examples/` | Scripts that print dataset facts and run a tiny fusion loop |
| `gaze_emotion/` | Shared constants, scaling, metrics, CSV loaders, toy classifier |
| `tests/` | Checks that those helpers match the documented behavior |

They intentionally avoid `transformers` so a laptop can audit the CSVs
without a GPU or a 500 MB weight download.
