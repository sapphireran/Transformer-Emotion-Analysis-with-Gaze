# Transformer Emotion Analysis with Gaze

Personal research repository (Sapphire Ran). The question is simple to state and
annoying to measure:

> If a transformer already reads a movie review, does a vector of eye-tracking
> features — how long people looked, how often they went back — change a
> 3-way sentiment decision?

The repo keeps two experimental tracks, the CSVs those tracks consume, and now
a docs/examples layer that can be run without downloading BERT or RoBERTa
weights.

| Track | Table | Gaze source | Script | Protocol |
| --- | --- | --- | --- | --- |
| ZuCo ∩ SST | 400 sentences | Human readers (ZuCo Task 1, 12 subjects, averaged) | `model_ZuCo_SST.py` | 5-fold stratified CV |
| Full SST | 11,853 sentences | Predicted word gaze, then aggregated | `model_full_SST.py` | 80/10/10 hold-out |

Labels are SST-3: `0 = negative`, `1 = neutral`, `2 = positive`.

This is a personal project. It is not affiliated with a company codebase.

## Repository layout

```
.
├── model_ZuCo_SST.py          # BERT/RoBERTa ± gaze on the 400-sentence table
├── model_full_SST.py          # same architecture on full SST
├── utils_ZuCo.py              # MATLAB → DataFrame transformer (needs .mat files)
├── read_ZuCo_mat.py           # per-subject sentence CSVs
├── convert_full_SST.py        # folder of review .txt files → ssts_ZuCo.csv
├── get_average_sentence_level.py
├── ZuCo_SST_data/             # joined ZuCo + SST tables and 80/10/10 split
├── ZuCo_et_csv_data/          # raw / scaled sentence gaze, plus word/
├── SST_data/                  # full SST + predicted sentence gaze
├── gaze_prediction/data/      # Provo + word-level predicted gaze
├── docs/                      # research notes, schemas, limitations
├── examples/                  # runnable analysis scripts (no GPU)
└── tests/                     # checks for the examples library
```

Original ZuCo `.mat` files and the SST `all/{NEGATIVE,POSITIVE,NEUTRAL}` text
folders are **not** checked in. Everything needed to inspect the derived tables
and to re-run the example scripts is.

## Quick start (docs and examples)

```bash
python -m pip install -r requirements-examples.txt
python examples/01_inspect_catalog.py
python examples/07_split_audit.py
python -m pytest
```

The example stack is pandas / numpy / scikit-learn / matplotlib. It does not
install `torch` or `transformers`.

Useful scripts:

| Script | What it does |
| --- | --- |
| `examples/01_inspect_catalog.py` | Open every catalogued CSV |
| `examples/02_sentence_gaze_stats.py` | Means, correlations, heatmaps |
| `examples/03_label_balance.py` | SST-3 mix vs majority baseline |
| `examples/04_compare_scalers.py` | Raw vs z-score vs min-max |
| `examples/05_word_level_profiles.py` | One ZuCo sentence, word by word |
| `examples/06_text_vs_gaze_baselines.py` | TF-IDF ± gaze logistic CV |
| `examples/07_split_audit.py` | ID leakage and label drift |
| `examples/08_write_dataset_report.py` | Refresh `docs/generated/dataset-report.md` |

## Training the transformers

Both training scripts share one fusion pattern:

```
sentence  →  BERT or RoBERTa pooler (768)
gaze (5)  →  Linear(5, 16)
concat(784) → Dropout(0.1) → Linear(784, 3)
```

`model_type` can be `bert`, `roberta`, `bert_eye_tracking`, or
`roberta_eye_tracking`. Hyperparameters are hardcoded at the top of each file.

```bash
python -m pip install torch transformers datasets scikit-learn pandas tqdm
python model_ZuCo_SST.py
python model_full_SST.py
```

`model_ZuCo_SST.py` expects `ZuCo_SST_data/combined_sst_et_standard.csv` and
uses columns `nFixations, FFD, GPT, TRT, GD`. `model_full_SST.py` expects the
`SST_data/{train,valid,test}_full_sst.csv` split and columns
`nFix, FFD, GPT, TRT, GD`.

A GPU is strongly recommended. CPU will run, just slowly.

## What the numbers look like (from the checked-in CSVs)

ZuCo ∩ SST is almost balanced: 123 / 137 / 140 negative / neutral / positive.
Full SST is not: 4,649 / 2,241 / 4,963. The majority class on full SST is
already about 42% accuracy; on ZuCo it is 35%.

On the 400 ZuCo sentences, several gaze channels are nearly redundant after
subject averaging. `nFixations` correlates with `TRT` at **0.96** and with
`GPT` at **0.91**. That is why the examples treat the five-channel vector as a
small, collinear extra — not as five independent cognitive measurements.

Mean sentence length on ZuCo is about 18 words (range 3–42). Mean omission
rate is 0.32: readers skip roughly one word in three. Mean first-fixation
duration is 117 ms; mean total reading time is 203 ms per fixated word.

## Documentation

Start at [docs/README.md](docs/README.md). The short path through the notes:

1. [Overview](docs/overview.md) — why two tracks exist
2. [Datasets](docs/datasets.md) — files, row counts, schemas
3. [Gaze features](docs/gaze-features.md) — nFix, FFD, GD, TRT, GPT, SFD
4. [Preprocessing](docs/preprocessing.md) — MATLAB → CSV → join → scale
5. [Architecture](docs/architecture.md) — fusion diagram
6. [Training](docs/training-and-evaluation.md) — CV vs hold-out, metrics
7. [Reproduction](docs/reproduction.md) — what you can and cannot rebuild
8. [Known limitations](docs/notes/known-limitations.md) — alignment, eval bugs
9. [Worked example](docs/notes/worked-example-sentence-0.md) — sentence 0
10. [Generated report](docs/generated/dataset-report.md) — refreshed by example 08

## License and data credit

Code in this repository is personal research code. The underlying corpora are
not redistributed as official releases:

- **ZuCo** (Hollenstein et al.) — EEG and eye-tracking during natural reading
- **Stanford Sentiment Treebank** — movie-review sentences and sentiment labels
- **Provo** — word-level reading measures used here only as a gaze-prediction source

If you reuse the derived CSVs, cite the original corpus papers, not just this
repo.
