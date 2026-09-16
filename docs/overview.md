# Project overview

This is a personal experiment in **whether reading-behavior features help a transformer decide the sentiment of a movie-review sentence**.

The hypothesis is simple enough to write down:

1. Sentiment in short reviews is mostly lexical — a transformer already sees that.
2. Some reviews are sarcastic, mixed, or bland. Those sentences often produce different fixation patterns (rereading, long go-past times, many refixations).
3. If you project five classic eye-tracking measures into a small vector and concatenate it with the pooled transformer state, the classifier might pick up that extra signal.

The code in this repo tests (3) in two regimes: a **measured** 400-sentence ZuCo overlap, and a **predicted-gaze** copy of the full Stanford Sentiment Treebank.

## Why two tables exist

ZuCo Task 1 (normal reading) recorded 12 people reading a few hundred SST movie-review sentences while EEG and eye-tracking ran. That overlap is the only place this project has *real* gaze aligned to *gold* 3-class labels.

Four hundred rows is enough for a 5-fold check and not enough to train `roberta-base` from scratch in the usual SST setup. The `SST_data/` tables expand the text side to ~11.8k sentences and fill gaze with a predictor trained from ZuCo / PROVO-style word measures (`gaze_prediction/`).

Those two tables must not be averaged together. One column set is z-scored human data; the other is model output stored under the short name `nFix`.

```text
ZuCo .mat  ->  utils_ZuCo.DataTransformer
           ->  ZuCo_et_csv_data/{1-12}_SR.csv
           ->  average + scale
           ->  join ssts_ZuCo.csv
           ->  ZuCo_SST_data/combined_sst_et_standard.csv
           ->  model_ZuCo_SST.py

SST sentences -> word placeholders (convert_sst_to_et.py)
              -> gaze_prediction/
              -> sentence aggregate
              -> SST_data/combined_full_sst_et.csv
              -> model_full_SST.py
```

[data-pipeline.md](data-pipeline.md) walks each arrow. [gaze-features.md](gaze-features.md) defines the columns. [model-architecture.md](model-architecture.md) draws the fusion head.

## Model switch in the training scripts

Both `model_ZuCo_SST.py` and `model_full_SST.py` read a string:

| `model_type` | Encoder | Gaze branch |
|---|---|---|
| `bert` | `BertForSequenceClassification` | no |
| `roberta` | `RobertaForSequenceClassification` | no |
| `bert_eye_tracking` | `BertModel` + `EyeTrackingModel` | yes |
| `roberta_eye_tracking` | `RobertaModel` + `EyeTrackingModel` | yes |

Text-only runs exist so a personal log can say “gaze helped” or “gaze did nothing” on the same split.

## What the examples add

The transformer jobs want GPU memory and Hub downloads. The personal examples do not. They use the checked-in CSVs plus logistic regression to answer smaller questions:

- Are the files complete and labeled the way the scripts think they are?
- How correlated are the five fusion features? Do the 12 readers even agree?
- Does a TF-IDF model already saturate the 400-sentence set? Does adding gaze move F1?
- What does one sentence look like at word level when you line up `word_averages_v2.csv`?

Those scripts live in `examples/` and import `tea_gaze`. They are companions, not a rewrite of the original trainers.
