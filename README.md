# Transformer Emotion Analysis with Gaze

Personal research notes and experiment code for **ternary movie-review sentiment**
when a transformer encoder can also see **eye-tracking features**.

This repository is a personal project. It is not a product, not a company
codebase, and not an official release of the ZuCo or SST datasets. The scripts
here document one way to:

1. Read ZuCo sentiment-reading gaze (12 subjects, 400 SST sentences).
2. Average and scale those features at sentence and word level.
3. Train BERT / RoBERTa with a late-fusion gaze head.
4. Transfer predicted gaze onto the full Stanford Sentiment Treebank.

The original training scripts (`model_ZuCo_SST.py`, `model_full_SST.py`) are
kept as-is. The `docs/` and `examples/` trees expand the personal write-up:
what each CSV is, what the gaze columns mean, how the fusion layer is wired,
and small programs that run on the checked-in tables without a GPU.

## Why gaze belongs next to text

Reading time is not a synonym for sentiment, but it is a noisy record of
**where a reader slowed down, skipped, or returned**. On ZuCo Task 1, people
read SST movie reviews while eye-trackers recorded fixations. Those traces
are a second view of the same sentence:

- High **omission rate** often means function words were skipped.
- Long **go-past time** often means a regression after a hard span.
- Large **total reading time** relative to first-pass gaze duration often
  means the reader came back.

The hypothesis explored here is modest: a small MLP over five (or eight)
gaze scalars, concatenated with the transformer pooler vector, can help a
3-way sentiment head on a 400-sentence set, and the same fusion recipe can
be reused on full SST once gaze is *predicted* rather than measured.

## Repository map

```text
.
├── model_ZuCo_SST.py          # 5-fold CV on 400 ZuCo+SST rows
├── model_full_SST.py          # train/valid/test on ~11.8k SST rows
├── utils_ZuCo.py              # MATLAB → DataFrame transformer
├── read_ZuCo_mat.py           # dump 12 subject CSVs from .mat
├── convert_full_SST.py        # folder of .txt reviews → ssts_ZuCo.csv
├── get_average_sentence_level.py
├── ZuCo_et_csv_data/          # real gaze, sentence + word
├── ZuCo_SST_data/             # 400 labeled sentences + scaled gaze
├── SST_data/                  # full SST with projected gaze
├── gaze_prediction/data/      # word-level predicted / PROVO tables
├── docs/                      # personal write-up
└── examples/                  # runnable inspections and a toy fusion model
```

Start with [docs/README.md](docs/README.md). The short path through the
write-up is:

| Doc | What it answers |
| --- | --- |
| [docs/datasets.md](docs/datasets.md) | Which CSV is which, row counts, label splits |
| [docs/gaze-features.md](docs/gaze-features.md) | nFix, FFD, SFD, GD, TRT, GPT, pupil, omission |
| [docs/pipeline.md](docs/pipeline.md) | `.mat` → subject CSV → average → SST join |
| [docs/model-architecture.md](docs/model-architecture.md) | BERT/RoBERTa + 16-d gaze MLP |
| [docs/experiments.md](docs/experiments.md) | Hyperparameters and evaluation protocol |
| [docs/known-issues.md](docs/known-issues.md) | Bugs left in the historical scripts |
| [docs/glossary.md](docs/glossary.md) | Short definitions |

## Two experiment settings

### A. Measured gaze (ZuCo ∩ SST)

- **400** sentences that appear in both ZuCo Task 1 and SST.
- Labels: `0` negative (123), `1` neutral (137), `2` positive (140).
- Gaze: subject-averaged, then min-max or z-scored.
- Protocol in `model_ZuCo_SST.py`: stratified 5-fold, 20 epochs, batch 16.

### B. Predicted gaze (full SST)

- **11,853** sentences (`SST_data/combined_full_sst_et.csv`).
- Split 80 / 10 / 10 → 9482 / 1185 / 1186.
- Gaze columns are the five-feature subset used by the fusion head:
  `nFix`, `GD`, `TRT`, `FFD`, `GPT`.
- Protocol in `model_full_SST.py`: 5 epochs, batch 256, keep best valid acc.

Word-level predicted gaze for the full treebank lives in
`gaze_prediction/data/prediction_test_v2.csv` (191,971 tokens).

## Fusion head (intended math)

Let \(h \in \mathbb{R}^{768}\) be the encoder pooler output and
\(g \in \mathbb{R}^{5}\) the gaze vector.

\[
z = \mathrm{Dropout}\big([h;\; W_g g + b_g]\big),\quad
W_g \in \mathbb{R}^{16 \times 5}
\]

\[
\hat{y} = \mathrm{softmax}(W_c z + b_c),\quad
W_c \in \mathbb{R}^{3 \times 784}
\]

Text-only ablations skip \(g\) and use `BertForSequenceClassification` or
`RobertaForSequenceClassification` with `num_labels=3`.

A GPU-free stand-in of the same *idea* (hashed bag-of-words + logistic
regression, with and without gaze) is
[examples/toy_text_gaze_fusion.py](examples/toy_text_gaze_fusion.py).

## Run the personal examples

The example programs only need the lightweight stack:

```bash
python3 -m pip install -r requirements-examples.txt
python3 examples/inspect_datasets.py
python3 examples/gaze_feature_stats.py
python3 examples/sentence_gaze_walkthrough.py --sentence-id 3
python3 examples/toy_text_gaze_fusion.py
python3 -m pytest tests/ -q
```

The original transformer trainers still need `requirements.txt` plus
downloaded `bert-base-uncased` / `roberta-base` weights and, realistically,
a GPU.

```bash
python3 model_ZuCo_SST.py
python3 model_full_SST.py
```

Edit `model_type` at the top of each script:
`bert`, `roberta`, `bert_eye_tracking`, or `roberta_eye_tracking`.

## Sentiment label convention

Used everywhere in the joined CSVs:

| Integer | Class |
| ---: | --- |
| 0 | Negative |
| 1 | Neutral |
| 2 | Positive |

`convert_full_SST.py` is the mapping from the `NEGATIVE` / `NEUTRAL` /
`POSITIVE` folder names.

## Data sources (cite the originals)

- **ZuCo**: Hollenstein, N., et al. *ZuCo, a simultaneous EEG and
  eye-tracking resource for natural sentence reading.* Scientific Data, 2018.
- **SST**: Socher, R., et al. *Recursive Deep Models for Semantic
  Compositionality Over a Sentiment Treebank.* EMNLP, 2013.
- **PROVO** (word-level table under `gaze_prediction/data/provo.csv`):
  Luke, S. G., & Christianson, K. *The Provo Corpus: A large eye-tracking
  corpus with predictability ratings.* Behavior Research Methods, 2018.

Raw MATLAB files are **not** stored here. `utils_ZuCo.py` expects them under
`ZuCo_mat_data/` if you re-extract subject CSVs.

## License and scope

Scripts and notes in this clone are personal research material. Dataset
terms remain with the original authors. Do not treat the predicted-gaze
columns as measured human eye movements.
