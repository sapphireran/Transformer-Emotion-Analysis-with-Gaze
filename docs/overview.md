# Overview

## Question

Does **eye-tracking information** help a transformer decide the sentiment of a
movie-review sentence?

Humans do not read every word the same way. Negation, sarcasm, and rare
adjectives often attract longer first-pass gaze and more regressions. ZuCo
recorded those signals while people read SST-style reviews. This repo asks
whether a cheap projection of five sentence-level gaze numbers, concatenated
onto BERT/RoBERTa's pooled embedding, moves accuracy / F1 relative to text
alone.

## Why two tracks

ZuCo only recorded a **few hundred** of the SST items (400 joined rows in
`ZuCo_SST_data/combined_sst_et_standard.csv`). Full SST has on the order of
**12k** sentences (`SST_data/stts_all_sentence_level.csv`).

| Track | Gaze source | Table | Training script |
| --- | --- | --- | --- |
| A. Human | Mean of 12 ZuCo readers, sentence level | `ZuCo_SST_data/combined_sst_et_*.csv` | `model_ZuCo_SST.py` |
| B. Predicted | Word-level predictor, then sentence aggregate | `SST_data/*_full_sst.csv` | `model_full_SST.py` |

Track A is the scientifically cleaner comparison: the gaze is real. The sample
is small, so the script uses **stratified 5-fold CV** instead of a single
split (a hold-out `train.csv` / `valid.csv` / `test.csv` still exists for
sanity checks).

Track B is the scaling experiment: predict gaze for sentences nobody in ZuCo
read, then train a transformer on the usual SST split. Predicted columns are
**not** interchangeable with ZuCo milliseconds; they are model outputs, often
already scaled.

## Fusion idea (one picture)

```
sentence text ──► BERT/RoBERTa ──► pooler_output (B, 768)
                                           │
                                           │ concat
                                           ▼
gaze (B, 5) ──► Linear(5, 16) ──► (B, 16) ──► Dropout ──► Linear(784, 3) ──► logits
```

Text-only baselines skip the gaze branch and use Hugging Face
`BertForSequenceClassification` / `RobertaForSequenceClassification`.

Details: [model-architecture.md](model-architecture.md).

## What is *not* in this repo

- The original ZuCo MATLAB `.mat` files (`ZuCo_mat_data/` is referenced by
  `utils_ZuCo.py` but not checked in).
- A trained gaze-prediction network (only **outputs** under
  `gaze_prediction/data/`).
- EEG channels. `DataTransformer` can see EEG-bearing structs, but the CSVs
  and models here are **eye-tracking only**.
- A config system. Paths, `model_type`, epochs, and batch size are literals
  at the top of each training script.

## Mental model of IDs

- **ZuCo subject files** `1_SR.csv` … `12_SR.csv` correspond to transformer
  subject indices `0 … 11` in `DataTransformer.__call__`.
- **Sentence `id` / `sentence_id`** in the human tables is the ZuCo sentence
  index after skipping known-bad trials (see [data-pipeline.md](data-pipeline.md)).
- **Word rows** use `Sent_ID` like `0_NR` (normal reading) plus `Word_ID`.
- **Full SST** `sentence_id` in `train_full_sst.csv` is **not** the ZuCo
  index; it is an identifier from the larger SST dump.

Never join Track A and Track B on `sentence_id` without checking that you are
in the same ID space.
