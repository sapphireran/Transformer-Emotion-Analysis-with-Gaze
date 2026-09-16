# Transformer Emotion Analysis with Gaze

Personal research repo for **sentence-level sentiment classification** that
fuses a transformer text encoder (BERT or RoBERTa) with **eye-tracking (ET)
features**. The experiments sit on two related corpora:

| Setting | Text | Gaze | Script |
|---|---|---|---|
| Full SST | Stanford Sentiment Treebank movie reviews (~11.8k sentences) | Predicted / projected sentence-level ET features | `model_full_SST.py` |
| ZuCo ∩ SST | 400 movie-review sentences that also appear in ZuCo Task 1 | Recorded ET, averaged over 12 readers | `model_ZuCo_SST.py` |

Sentiment is a 3-way label: `0 = negative`, `1 = neutral`, `2 = positive`.

This repository previously shipped as a single commit of scripts and CSVs
with no README. The `docs/` and `examples/` trees document the data, the
late-fusion architecture, and the preprocessing pipeline, and they add
runnable checks that do **not** require a GPU or Hugging Face weights.

---

## Why fuse gaze with text?

Eye-tracking measures how long and how often a reader fixates each word.
In psycholinguistics those measures (first-fixation duration, go-past
time, total reading time, …) track processing difficulty. The hypothesis
tested here is that **adding a compact ET vector next to the transformer
`pooler_output`** can help a sentiment classifier, either because:

1. recorded gaze on ZuCo reviews carries residual affective signal, or
2. a gaze-prediction model can project similar features onto SST sentences
   that were never read in an eye tracker.

The fusion itself is deliberately small: a linear map `5 → 16` on the ET
vector, concatenate with the 768-d pooled text embedding, dropout, then a
3-way linear classifier. See [docs/model-architecture.md](docs/model-architecture.md).

---

## Repository map

```
.
├── model_full_SST.py          # BERT/RoBERTa ± ET on the full SST split
├── model_ZuCo_SST.py          # same architecture, 5-fold CV on 400 ZuCo sentences
├── utils_ZuCo.py              # DataTransformer: .mat → sentence/word ET tables
├── read_ZuCo_mat.py           # dump per-subject sentence-level CSVs
├── convert_full_SST.py        # folder of .txt reviews → ssts_ZuCo.csv
├── get_average_sentence_level.py
├── SST_data/                  # full SST text + projected ET + 80/10/10 split
├── ZuCo_SST_data/             # 400-sentence ZuCo∩SST join + 80/10/10 split
├── ZuCo_et_csv_data/          # per-subject sentence ET, averages, scalings
│   └── word/                  # per-subject word ET + cross-subject averages
├── gaze_prediction/data/      # word-level files in the predictor's schema
├── result/                    # pairwise scatter / histogram plots of ET features
├── docs/                      # dataset, feature, model, and pipeline notes
└── examples/                  # runnable inspection / baseline / fusion demos
```

---

## Datasets in this checkout

Exact row counts are from the files currently in the tree (header excluded):

| File | Rows | Role |
|---|---:|---|
| `SST_data/stts_all_sentence_level.csv` | 11,853 | raw SST sentences + text polarity |
| `SST_data/combined_full_sst_et.csv` | 11,853 | SST + 5 projected ET columns |
| `SST_data/train_full_sst.csv` | 9,482 | 80% train (`random_state=42`) |
| `SST_data/valid_full_sst.csv` | 1,185 | 10% valid |
| `SST_data/test_full_sst.csv` | 1,186 | 10% test |
| `ZuCo_SST_data/ssts_ZuCo.csv` | 400 | ZuCo Task 1 sentences + labels |
| `ZuCo_SST_data/combined_sst_et_standard.csv` | 400 | same + z-scored sentence ET |
| `ZuCo_SST_data/combined_sst_et_min_max.csv` | 400 | same + min-max sentence ET |
| `ZuCo_et_csv_data/{1–12}_SR.csv` | 400 / **299** | per-subject sentence ET (subject 3 is short) |
| `ZuCo_et_csv_data/word/{1–12}_SR.csv` | 7,129 / **5,293** | per-subject word ET |
| `gaze_prediction/data/prediction_test_v2.csv` | 191,971 | word-level predicted gaze for SST |
| `gaze_prediction/data/provo.csv` | 2,659 | PROVO word-level ET (has `fixProp`, not `GD`) |

Subject 3 (`3_SR.csv`) is shorter because ZuCo Task 1 is missing a block of
sentences for that reader. `utils_ZuCo.py` already skips those indices.
Details: [docs/datasets.md](docs/datasets.md).

---

## Model variants

`model_type` in both training scripts is one of:

| `model_type` | Text encoder | Gaze branch |
|---|---|---|
| `bert` | `BertForSequenceClassification` | none |
| `roberta` | `RobertaForSequenceClassification` | none |
| `bert_eye_tracking` | `BertModel` pooler + ET linear | 5 features → 16-d |
| `roberta_eye_tracking` | `RobertaModel` pooler + ET linear | 5 features → 16-d |

Default in both scripts is `roberta_eye_tracking`.

Hyperparameters differ by corpus:

| | Full SST | ZuCo ∩ SST |
|---|---|---|
| Epochs | 5 | 20 |
| Batch size | 256 | 16 |
| LR | `5e-5` (Adam) | `5e-5` (Adam) |
| Split | fixed train/valid/test | `StratifiedKFold(5)` |
| Checkpoint | best validation **accuracy** | last fold epoch (no early stop) |

---

## Quick start (examples, no GPU)

The example scripts only need pandas / numpy / scikit-learn. They read the
checked-in CSVs and never download transformer weights.

```bash
python -m pip install -r requirements.txt

python examples/inspect_datasets.py
python examples/data_integrity.py
python examples/gaze_feature_report.py
python examples/gaze_only_baseline.py
python examples/late_fusion_demo.py
python examples/word_level_preview.py
python examples/sentence_walkthrough.py

# or everything:
python examples/run_all.py
```

What each script does is listed in [examples/README.md](examples/README.md).

---

## Training the original models

You need a machine with enough RAM (and ideally a GPU) plus the training
extras:

```bash
python -m pip install -r requirements-train.txt
mkdir -p models
```

```bash
# 400-sentence ZuCo study (5-fold CV, ~minutes to tens of minutes on GPU)
python model_ZuCo_SST.py

# full SST study (downloads roberta-base on first run)
python model_full_SST.py
```

Edit the `model_type` assignment near the top of each script to compare
text-only vs text+gaze. Reproduction notes, path pitfalls, and a known
test-loop bug are in [docs/reproduction.md](docs/reproduction.md) and
[docs/known-quirks.md](docs/known-quirks.md).

Raw ZuCo `.mat` files are **not** in this checkout. `read_ZuCo_mat.py`
and `utils_ZuCo.py` expect them under `ZuCo_mat_data/<task>/` if you want
to regenerate the CSVs from scratch.

---

## Eye-tracking features

The five features the classifiers actually consume:

| Name in SST tables | Name in ZuCo tables | Meaning |
|---|---|---|
| `nFix` | `nFixations` | number of fixations |
| `FFD` | `FFD` | first-fixation duration |
| `GPT` | `GPT` | go-past / regression-path time |
| `TRT` | `TRT` | total reading time |
| `GD` | `GD` | gaze duration (first-pass) |

ZuCo sentence tables also keep `omissionRate`, `meanPupilSize`, and `SFD`
(single-fixation duration). Those extra columns are **not** passed into
`EyeTrackingModel`. Glossary: [docs/eye-tracking-features.md](docs/eye-tracking-features.md).

---

## Documentation

| Doc | Contents |
|---|---|
| [docs/README.md](docs/README.md) | index of the notes |
| [docs/datasets.md](docs/datasets.md) | schemas, splits, label counts |
| [docs/eye-tracking-features.md](docs/eye-tracking-features.md) | ET measure definitions |
| [docs/data-pipeline.md](docs/data-pipeline.md) | `.mat` → CSV → model inputs |
| [docs/model-architecture.md](docs/model-architecture.md) | late-fusion diagram and shapes |
| [docs/reproduction.md](docs/reproduction.md) | how to re-run each script |
| [docs/known-quirks.md](docs/known-quirks.md) | path mismatches, eval bug, naming |

---

## License and data attribution

Code in this personal repo is research scaffolding. The underlying corpora
are third-party:

- **ZuCo** (Zurich Cognitive Language Processing Corpus) — EEG + ET while
  reading. Task 1 is natural reading of movie reviews.
- **SST** (Stanford Sentiment Treebank) — movie-review sentences with
  polarity labels.
- **PROVO** (Provo Corpus) — word-level eye-tracking norms; used here only
  as a comparison distribution in `gaze_prediction/data/provo.csv`.

Check each corpus's original license before redistributing the CSVs
further. The `result/` PNGs are exploratory scatter/histogram plots of
those ET feature distributions.
