# Transformer Emotion Analysis with Gaze

Personal research notes for a pair of sentiment-classification experiments
that fuse **transformer text encodings** with **eye-tracking (gaze) features**.

This repository is Sapphire Ran's personal lab notebook plus the training and
data-prep scripts that sit next to it. It is **not** a company codebase and is
not packaged as a library. The goal of this README is to make the original
scripts readable months later: what was measured, which files matter, how the
two experiment tracks differ, and where the known traps are.

If you only want a walkthrough of one real sentence, start at
[`docs/06-worked-examples.md`](docs/06-worked-examples.md). If you want the
feature definitions first, start at
[`docs/02-gaze-feature-glossary.md`](docs/02-gaze-feature-glossary.md).

---

## Why gaze + sentiment?

Human readers do not spend time uniformly across a sentence. Content words,
unexpected adjectives, and emotionally loaded tokens attract more fixations
and longer total reading time. Function words (`a`, `the`, `of`) are often
skipped. The ZuCo corpus records those signals while people read movie-review
sentences that already have Stanford Sentiment Treebank (SST) labels.

The working hypothesis in this repo is:

> A transformer already sees the words. Adding a small vector of gaze
> statistics (how long people looked, how often they went back) should help
> the classifier when the lexical signal is ambiguous — especially on short,
> sarcastic, or mixed-polarity reviews.

Two tracks test that idea at different scales:

| Track | Script | Text source | Gaze source | Size | Protocol |
| --- | --- | --- | --- | --- | --- |
| **ZuCo-SST** | `model_ZuCo_SST.py` | 400 ZuCo movie-review sentences | Real eye tracking, 12 readers, averaged | 400 rows | 5-fold stratified CV |
| **Full SST** | `model_full_SST.py` | Full SST sentence set | Predicted / transferred gaze features | 11,853 rows | 80 / 10 / 10 split |

The ZuCo track is the cleaner scientific claim (real gaze, small *n*). The
full-SST track asks whether a *predicted* gaze signal still helps when you
have more text but no eye tracker.

---

## Repository map

```
.
├── model_ZuCo_SST.py            # RoBERTa/BERT ± gaze, 5-fold CV on 400 sentences
├── model_full_SST.py            # same fusion idea, hold-out eval on full SST
├── utils_ZuCo.py                # MATLAB → DataFrame transformer (sentence / word)
├── read_ZuCo_mat.py             # dump per-subject sentence CSVs from .mat files
├── convert_full_SST.py          # folder of .txt reviews → ssts_ZuCo.csv
├── get_average_sentence_level.py
├── ZuCo_SST_data/               # 400 labeled sentences + scaled gaze
├── ZuCo_et_csv_data/            # raw and scaled gaze, sentence + word
├── SST_data/                    # full SST + transferred gaze features
├── gaze_prediction/data/        # predicted word-level gaze, PROVO reference
├── result/                      # scatter/histogram plots of gaze distributions
├── docs/                        # personal commentary (this expansion)
└── examples/                    # stdlib-only walkthroughs of the CSVs
```

There is no `requirements.txt` in the original commit. The training scripts
import `torch`, `transformers`, `datasets`, `sklearn`, `pandas`, and `tqdm`.
The MATLAB reader also needs `scipy`. The example scripts under `examples/`
use only the Python 3 standard library so they run in a bare environment.

---

## Sentiment labels

Both tracks use the same three-way mapping:

| Folder / string | Integer | Meaning |
| --- | --- | --- |
| `NEGATIVE` | `0` | Negative review |
| `NEUTRAL`  | `1` | Neutral / mixed |
| `POSITIVE` | `2` | Positive review |

`convert_full_SST.py` and `ZuCo_SST_data/save_SST_data.py` both apply this
map. Neutral is the minority class on full SST (2,241 / 11,853) and roughly
balanced on ZuCo (137 / 400).

---

## Gaze features used by the models

The fusion head always takes **five** features:

```
nFixations (or nFix), FFD, GPT, TRT, GD
```

ZuCo also records `omissionRate`, `meanPupilSize`, and `SFD`. Those columns
are in the CSVs and are useful for analysis, but they are **not** concatenated
into the classifier in `EyeTrackingModel`.

One-line meanings (see the glossary for units, typical ranges, and a real
sentence walkthrough):

- **nFix / nFixations** — how many times the eyes landed on the region.
- **FFD** — first fixation duration (first look).
- **GD** — gaze duration (first pass, before leaving the word).
- **TRT** — total reading time (every look, including regressions).
- **GPT** — go-past time (from first entry until the eyes move *past* the word).
- **SFD** — single fixation duration (only if the word was fixated exactly once).
- **omissionRate** — fraction of words never fixated.
- **meanPupilSize** — pupil diameter, a coarse arousal / effort proxy.

On the ZuCo sentence-level file `ZuCo_et_csv_data/average_data.csv` (400
sentences, 12-reader mean, raw milliseconds / counts):

| Feature | Mean | Std | Min | Max |
| --- | --- | --- | --- | --- |
| omissionRate | 0.32 | 0.07 | 0.16 | 0.60 |
| nFixations | 1.69 | 0.31 | 1.20 | 3.58 |
| meanPupilSize | 797 | 61 | 680 | 970 |
| GD (ms) | 141 | 21 | 108 | 273 |
| TRT (ms) | 203 | 48 | 131 | 427 |
| FFD (ms) | 117 | 8 | 102 | 166 |
| SFD (ms) | 72 | 11 | 39 | 122 |
| GPT (ms) | 242 | 57 | 153 | 587 |

After `StandardScaler`, those same columns have mean ≈ 0 and std ≈ 1 in
`ZuCo_SST_data/combined_sst_et_standard.csv`. That is the file
`model_ZuCo_SST.py` actually trains on.

---

## The fusion model, in one paragraph

`EyeTrackingModel` loads `bert-base-uncased` or `roberta-base`, takes the
pooled `[CLS]` / `<s>` vector (hidden size 768), runs the 5-D gaze vector
through a linear layer to 16 dimensions, concatenates `(768 + 16)`, applies
dropout 0.1, and classifies into 3 logits. Text-only ablations use
`BertForSequenceClassification` / `RobertaForSequenceClassification` with the
same label count. Switch tracks with the `model_type` string:

```
'bert' | 'roberta' | 'bert_eye_tracking' | 'roberta_eye_tracking'
```

Both training scripts default to `'roberta_eye_tracking'`.

---

## How to run the original experiments

These commands assume a GPU machine with the Hugging Face stack installed.
They will download `roberta-base` on first run.

```bash
# Track A — 400 ZuCo sentences, 5-fold CV, 20 epochs, batch 16
python model_ZuCo_SST.py

# Track B — full SST hold-out, 5 epochs, batch 256, writes models/best_*.pth
python model_full_SST.py
```

Data-prep scripts (need the original ZuCo `.mat` files, which are **not** in
this clone):

```bash
python read_ZuCo_mat.py                 # per-subject sentence CSVs
python get_average_sentence_level.py    # 12-reader mean + min-max / z-score
python convert_full_SST.py              # .txt folders → ssts_ZuCo.csv
```

The checked-in CSVs already contain the outputs of those steps, so you can
inspect data without MATLAB.

---

## How to run the documentation examples

No third-party packages. From the repo root:

```bash
python3 examples/inspect_datasets.py
python3 examples/walk_sentence_gaze.py
python3 examples/walk_sentence_gaze.py --sentence-id 4
python3 examples/walk_sentence_gaze.py --sentence-id 80
python3 examples/fusion_sketch.py
```

(`python3` is the interpreter on this machine; `python` is not
installed.)

`inspect_datasets.py` reprints the split sizes and label counts used
throughout the docs so the numbers stay honest if a CSV is regenerated.
`walk_sentence_gaze.py` prints word-level nFix / FFD / TRT / GPT for one
ZuCo sentence and comments on which tokens the readers lingered on.
`fusion_sketch.py` is a tiny, dependency-free sketch of the 768+16 concat
so the architecture note can be checked by hand.

---

## Dataset snapshot (this clone)

| File | Rows | What it is |
| --- | --- | --- |
| `ZuCo_SST_data/combined_sst_et_standard.csv` | 400 | ZuCo sentences + z-scored gaze |
| `ZuCo_SST_data/train.csv` / `valid.csv` / `test.csv` | 320 / 40 / 40 | 80/10/10 split (not used by the CV script) |
| `ZuCo_et_csv_data/{1-12}_SR.csv` | 400 each | Per-reader sentence gaze |
| `ZuCo_et_csv_data/word/word_averages_v2.csv` | 7,129 | 12-reader mean, word level |
| `SST_data/combined_full_sst_et.csv` | 11,853 | Full SST + 5 predicted gaze cols |
| `SST_data/train_full_sst.csv` | 9,482 | 80% |
| `SST_data/valid_full_sst.csv` | 1,185 | 10% |
| `SST_data/test_full_sst.csv` | 1,186 | 10% |
| `gaze_prediction/data/prediction_test.csv` | 1,751 | Predicted word-level gaze |
| `gaze_prediction/data/provo.csv` | 2,659 | PROVO-style reference gaze |

ZuCo label counts: **123 / 137 / 140** (neg / neu / pos).
Full SST label counts: **4,649 / 2,241 / 4,963**.

---

## Personal documentation

Read in this order if you are coming back to the project cold:

1. [Reading order and conventions](docs/00-reading-order.md)
2. [Research notes — why this setup](docs/01-research-notes.md)
3. [Gaze feature glossary, with numbers](docs/02-gaze-feature-glossary.md)
4. [Data pipeline, file by file](docs/03-data-pipeline.md)
5. [Model architecture commentary](docs/04-model-architecture.md)
6. [Training, splits, and metrics](docs/05-training-and-evaluation.md)
7. [Worked examples from the CSVs](docs/06-worked-examples.md)
8. [Lab notes, quirks, and next questions](docs/07-personal-lab-notes.md)

---

## License / data credit

The scripts in this clone do not ship a license file. ZuCo is the Zurich
Cognitive Language Processing Corpus (Hollenstein et al.). SST is the
Stanford Sentiment Treebank (Socher et al.). PROVO is the Provo Corpus
(Luke & Christianson). Use those datasets under their original terms.
This repo only stores derived CSVs and personal notes.
