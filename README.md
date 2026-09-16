# Transformer Emotion Analysis with Gaze

Personal research repository for **three-class sentiment classification** that fuses a transformer text encoder (BERT or RoBERTa) with sentence-level eye-tracking features.

The working hypothesis is that reading-time measures — first-pass duration, go-past time, total reading time, gaze duration, and fixation count — carry a weak but usable signal about how a sentence is processed, and that concatenating a learned projection of those features with the transformer's pooled embedding can change classification behavior relative to text-only baselines.

This is a personal study repo. It is not a packaged library and it is not affiliated with any employer.

## What is in the repo

Two training entry points share the same fusion idea:

| Script | Data | Protocol | Typical use |
| --- | --- | --- | --- |
| `model_ZuCo_SST.py` | 400 ZuCo sentiment-reading sentences with **measured** gaze | 5-fold stratified CV, 20 epochs, batch 16 | Small, real eye-tracking experiment |
| `model_full_SST.py` | 11,853 SST sentences with **predicted / transferred** gaze | 80/10/10 split, 5 epochs, batch 256 | Larger movie-review setting |

Both scripts switch among four model types via the `model_type` string:

- `bert` / `roberta` — Hugging Face sequence classification (text only)
- `bert_eye_tracking` / `roberta_eye_tracking` — custom `EyeTrackingModel` that concatenates pooled text features with a linear projection of 5 gaze features

Supporting scripts convert ZuCo `.mat` recordings, average subjects, scale features, and join sentiment labels. Those steps are documented in [docs/data-pipeline.md](docs/data-pipeline.md). Runnable walkthroughs that do **not** download BERT/RoBERTa live under [examples/](examples/).

## Quick start

```bash
# Training stack (GPU recommended)
python3 -m pip install -r requirements.txt

# Lightweight docs / examples stack
python3 -m pip install -r requirements-examples.txt

# Inventory the checked-in CSVs and write a report
python3 examples/inspect_datasets.py

# Run every example and write reports under examples/output/
python3 examples/run_all.py
```

Training (edit `model_type` at the top of the script first):

```bash
python3 model_ZuCo_SST.py      # 400-sentence ZuCo CV
python3 model_full_SST.py      # full SST split
```

`model_full_SST.py` writes `models/best_{model_type}_model.pth`. Create the `models/` directory before the first run, or the `torch.save` call will fail.

## Sentiment labels

All joined tables use integer labels:

| Integer | Meaning | Source folder / SST tag |
| --- | --- | --- |
| 0 | negative | `NEGATIVE` |
| 1 | neutral | `NEUTRAL` |
| 2 | positive | `POSITIVE` |

ZuCo (400 sentences) is nearly balanced: 123 / 137 / 140. Full SST (11,853 sentences) is not: 4,649 / 2,241 / 4,963. Neutral is the minority class on SST, which is why the training scripts report **weighted** precision, recall, and F1.

## Eye-tracking features

The fusion models always consume five features. Column names differ by table:

| Fusion input | ZuCo column | SST / predicted-gaze column | Typical unit |
| --- | --- | --- | --- |
| 1 | `nFixations` | `nFix` | count (then scaled) |
| 2 | `FFD` | `FFD` | milliseconds (then scaled) |
| 3 | `GPT` | `GPT` | milliseconds (then scaled) |
| 4 | `TRT` | `TRT` | milliseconds (then scaled) |
| 5 | `GD` | `GD` | milliseconds (then scaled) |

ZuCo tables also keep `omissionRate`, `meanPupilSize`, and `SFD`. Those extra columns are **not** passed into `EyeTrackingModel`. Feature definitions and scaling notes are in [docs/eye-tracking-features.md](docs/eye-tracking-features.md).

## Repository layout

```
.
├── model_ZuCo_SST.py          # BERT/RoBERTa ± gaze, 5-fold CV on ZuCo
├── model_full_SST.py          # same architecture, SST train/valid/test
├── utils_ZuCo.py              # DataTransformer for ZuCo MATLAB files
├── read_ZuCo_mat.py           # export per-subject sentence CSVs
├── get_average_sentence_level.py
├── convert_full_SST.py
├── ZuCo_SST_data/             # 400 labeled sentences + joined gaze
├── ZuCo_et_csv_data/          # 12 subjects, averages, word-level tables
├── SST_data/                  # 11,853 SST sentences + transferred gaze
├── gaze_prediction/data/      # word-level predicted gaze + Provo sample
├── result/                    # scatter/histogram plots of predicted gaze
├── docs/                      # architecture, pipeline, dictionary, notes
└── examples/                  # runnable inspection / baseline scripts
```

## Dataset snapshot (checked-in CSVs)

Counts below were generated from the files in this checkout. Re-run `python3 examples/inspect_datasets.py` if the CSVs change.

**ZuCo sentiment-reading (real gaze)**

- 400 sentences after joining `ssts_ZuCo.csv` with subject-averaged eye tracking
- 12 subjects (`1_SR.csv` … `12_SR.csv`); subject 3 is missing 101 sentence rows (299 instead of 400), which matches the task-1 skip logic in `utils_ZuCo.py`
- Word-level tables have 7,129 tokens per complete subject; subject 3 has 5,293
- Raw sentence-level means (subject average, before z-scoring): ~1.69 fixations/word, FFD ≈ 117 ms, GD ≈ 141 ms, TRT ≈ 203 ms, GPT ≈ 242 ms, omission rate ≈ 0.32

**Stanford Sentiment Treebank (transferred gaze)**

- 11,853 sentences in `combined_full_sst_et.csv`
- Split 9,482 / 1,185 / 1,186 (train / valid / test), random 80/10/10 with `random_state=42`
- Gaze columns are already standardized (combined mean ≈ 0, std ≈ 1)
- `sst_et_test.csv` is a word-level **placeholder** (all gaze values are 0). Predicted values live in `gaze_prediction/data/prediction_test_v2.csv` (191,971 word rows)

## Model sketch

```
sentence ──► BERT / RoBERTa ──► pooler_output (768)
                                      │
                                      ▼
gaze[5] ──► Linear(5 → 16) ──► concat ──► Dropout(0.1) ──► Linear(784 → 3)
```

Text-only variants skip the gaze branch and use `*ForSequenceClassification`. A CPU-only walkthrough of the concat geometry, without downloading pretrained weights, is `examples/fusion_architecture_demo.py`.

## Documentation

- [docs/README.md](docs/README.md) — index
- [docs/architecture.md](docs/architecture.md) — fusion model and training loops
- [docs/data-dictionary.md](docs/data-dictionary.md) — every checked-in CSV
- [docs/data-pipeline.md](docs/data-pipeline.md) — from ZuCo `.mat` to training tables
- [docs/eye-tracking-features.md](docs/eye-tracking-features.md) — feature glossary
- [docs/models-and-training.md](docs/models-and-training.md) — hyperparameters and metrics
- [docs/reproduction.md](docs/reproduction.md) — how to re-run or approximate a run
- [docs/known-issues.md](docs/known-issues.md) — path mismatches and evaluation bugs
- [docs/experiment-notes.md](docs/experiment-notes.md) — personal notes on setup choices
- [docs/citations.md](docs/citations.md) — ZuCo, SST, Provo

## Examples

See [examples/README.md](examples/README.md). The scripts only read the CSVs already in the repo. They do not call social APIs and they do not download transformer checkpoints.

```bash
python3 examples/schema_check.py
python3 examples/label_distribution.py
python3 examples/gaze_feature_profiles.py
python3 examples/dummy_baseline.py
```

## License

MIT. Dataset files retain the licenses of their original distributors (ZuCo, SST, Provo). Do not treat the checked-in CSVs as a substitute for obtaining those corpora from their authors when you need the official release.
