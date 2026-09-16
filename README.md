# Transformer Emotion Analysis with Gaze

Personal research code for **three-class movie-review sentiment** (negative / neutral / positive) with **transformer text encoders** and **eye-tracking (gaze) features**.

Two tracks share the same fusion idea and differ in how gaze is obtained:

| Track | Script | Text + labels | Gaze | Protocol |
| --- | --- | --- | --- | --- |
| ZuCo-SST (measured) | `model_ZuCo_SST.py` | 400 ZuCo Task 1 sentences | 12-subject sentence averages | 5-fold stratified CV |
| Full SST (predicted) | `model_full_SST.py` | 11,853 SST sentences | predicted / standardized gaze | 80 / 10 / 10 split |

There is **no pretrained checkpoint in this repo**. Training scripts download `bert-base-uncased` or `roberta-base` at runtime.

This tree now also has a **docs/** guide and **CPU-only examples/** that inspect the committed CSVs, reconstruct the ZuCo join, and replay the fusion layer with NumPy (no GPU).

## Why gaze?

Reading-time features (fixation counts, first-fixation duration, go-past time, and so on) are a noisy proxy for processing difficulty. The hypothesis in this project is that a small MLP on those features, concatenated with a BERT/RoBERTa pooled vector, can help a sentiment classifier. On the files in this repo, raw Pearson correlation between gaze and the 3-class label is **weak** (ZuCo FFD ≈ 0.07; full SST predicted features ≈ −0.05). Predicted SST gaze channels are also **highly collinear**. The examples package measures both facts so they are not anecdotal.

## Model variants

Set `model_type` at the top of either training script:

| `model_type` | Encoder | Gaze branch |
| --- | --- | --- |
| `bert` | `BertForSequenceClassification` | no |
| `roberta` | `RobertaForSequenceClassification` | no |
| `bert_eye_tracking` | `BertModel` + concat | yes |
| `roberta_eye_tracking` | `RobertaModel` + concat | yes (default) |

Fusion (matches `EyeTrackingModel` in both training files):

```
sentence  → tokenizer (max_length=128) → BERT/RoBERTa → pooler_output [B, 768]
5 gaze dims → Linear(5, 16)  (no activation)
concat [B, 784] → Dropout(0.1) → Linear(784, 3) → logits
```

The five gaze dimensions actually fed to the classifier:

- Full SST: `nFix`, `FFD`, `GPT`, `TRT`, `GD`
- ZuCo-SST: `nFixations`, `FFD`, `GPT`, `TRT`, `GD`  
  (ZuCo tables also store `omissionRate`, `meanPupilSize`, `SFD`, `SentLen`; those extra columns are **not** passed into `EyeTrackingModel`.)

Labels: `NEGATIVE=0`, `NEUTRAL=1`, `POSITIVE=2`.

## Repository layout

```
.
├── model_ZuCo_SST.py          # 5-fold CV on 400 ZuCo sentences
├── model_full_SST.py          # train / valid / test on full SST
├── utils_ZuCo.py              # MATLAB → DataFrame (needs original .mat files)
├── read_ZuCo_mat.py           # per-subject sentence CSVs
├── get_average_sentence_level.py
├── convert_full_SST.py
├── ZuCo_et_csv_data/          # measured gaze, sentence + word
├── ZuCo_SST_data/             # 400 sentences + joined gaze
├── SST_data/                  # 11,853 sentences + predicted gaze
├── gaze_prediction/data/      # word-level predicted gaze + Provo
├── result/                    # historical scatter/histogram plots
├── docs/                      # datasets, features, pipeline, issues
└── examples/                  # runnable analysis + tests
```

A file-by-file map is in [docs/file-map.md](docs/file-map.md).

## Data at a glance

**ZuCo Task 1 (normal reading of SST movie reviews)** — 12 subjects, sentence-level files `ZuCo_et_csv_data/{1–12}_SR.csv`. Subject 3 (`3_SR.csv`) has **299** rows; everyone else has **400**. Combined text+gaze: 400 rows, labels `{0: 123, 1: 137, 2: 140}`.

**Stanford Sentiment Treebank (sentence-level, 3-class)** — `SST_data/stts_all_sentence_level.csv`, 11,853 rows, labels `{NEGATIVE: 4649, NEUTRAL: 2241, POSITIVE: 4963}`. The same counts appear in `combined_full_sst_et.csv` as `{0,1,2}`.

**Splits (disjoint `sentence_id`, random seed 42 in `spilt.py`)**

| Split | ZuCo-SST | Full SST |
| --- | ---: | ---: |
| train | 320 | 9,482 |
| valid | 40 | 1,185 |
| test | 40 | 1,186 |

`model_ZuCo_SST.py` does **not** use those 320/40/40 CSVs; it re-splits `combined_sst_et_standard.csv` with `StratifiedKFold(n_splits=5)`.

## Quick start — examples (CPU)

```bash
python3 -m pip install -r requirements-examples.txt
make test
make examples
```

Scripts (also listed in [examples/README.md](examples/README.md)):

- inventory every committed table
- Pearson / collinearity of gaze channels
- reconstruct `combined_sst_et_*.csv` from text ⨝ scaled gaze
- show that subject 3 is row-aligned only through original sentence 149
- replay the 5→16→concat→3 fusion with NumPy

## Quick start — original training

Needs GPU-class hardware, Hugging Face weights, and the packages in `requirements.txt`.

```bash
python3 -m pip install -r requirements.txt
python3 model_ZuCo_SST.py      # default: roberta_eye_tracking, 20 epochs, batch 16
python3 model_full_SST.py      # default: roberta_eye_tracking, 5 epochs, batch 256
```

`model_full_SST.py` writes `models/best_{model_type}_model.pth`. Create `models/` first. Read [docs/known-issues.md](docs/known-issues.md) before trusting the printed test scores: the test loop currently keeps only the **last batch**.

MATLAB `.mat` dumps for ZuCo are **not** in this repo. `read_ZuCo_mat.py` expects `ZuCo_mat_data/task1/` (and the Windows-style default subdirectory in `utils_ZuCo.get_matfiles`).

## Documentation

- [docs/README.md](docs/README.md) — guide index
- [docs/datasets.md](docs/datasets.md) — ZuCo, SST, Provo, predicted gaze
- [docs/gaze-features.md](docs/gaze-features.md) — nFix, FFD, GD, TRT, GPT, SFD, pupil, omission
- [docs/data-pipeline.md](docs/data-pipeline.md) — MAT → CSV → join → split
- [docs/architecture.md](docs/architecture.md) — fusion diagrams and shapes
- [docs/training.md](docs/training.md) — hyperparameters and metrics
- [docs/known-issues.md](docs/known-issues.md) — alignment, test-loop, path, naming
- [docs/references.md](docs/references.md) — ZuCo, SST, Provo, transformers
- [docs/examples-walkthrough.md](docs/examples-walkthrough.md) — what the scripts print

## License and provenance

Personal research repository. Upstream corpora have their own licenses and citation requirements (see [docs/references.md](docs/references.md)). Do not redistribute ZuCo MATLAB files from this tree; they are not included.
