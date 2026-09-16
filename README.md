# Transformer Emotion Analysis with Gaze

Personal research code for **three-class sentiment classification** that fuses
transformer text encodings with **eye-tracking (gaze) features**.

The hypothesis is simple: how a reader looks at a sentence — how often they
fixate, how long first and later fixations last, whether they regress — carries
signal about difficulty, surprise, and affect that is complementary to the
words themselves. This repo wires that idea into BERT and RoBERTa classifiers
and keeps the preprocessing that turns ZuCo MATLAB recordings into the CSVs
the models actually train on.

This is a **personal experiment repo**, not a packaged library. Scripts are
run from the repository root. There is no installable package and no public
model checkpoint.

## What lives where

```
.
├── model_ZuCo_SST.py          # 5-fold CV on the 400-sentence ZuCo ∩ SST set
├── model_full_SST.py          # train / valid / test on ~11.8k SST sentences
├── utils_ZuCo.py              # MATLAB → sentence/word feature tables
├── read_ZuCo_mat.py           # dump per-subject sentence-level CSVs
├── get_average_sentence_level.py
├── convert_full_SST.py        # folder of .txt reviews → labeled CSV
├── ZuCo_SST_data/             # 400 sentences with real (averaged) gaze
├── ZuCo_et_csv_data/          # raw and scaled gaze, sentence + word
├── SST_data/                  # full SST with transferred gaze columns
├── gaze_prediction/           # word-level format used to estimate SST gaze
├── result/                    # exploratory scatter / histogram plots
├── docs/                      # design notes, feature glossary, quirks
└── examples/                  # lightweight scripts (stdlib + numpy only)
```

## Two experimental tracks

| Track | Script | Text + gaze source | Protocol | Why it exists |
| --- | --- | --- | --- | --- |
| **Measured gaze** | `model_ZuCo_SST.py` | ZuCo Task 1 sentences that also appear in SST | Stratified 5-fold CV, 20 epochs, batch 16 | Gold eye-tracking, small *n* |
| **Transferred gaze** | `model_full_SST.py` | Full SST-2/3 style sentence set | 80 / 10 / 10 split, 5 epochs, batch 256 | Scale, but gaze is predicted / mapped, not recorded |

Both scripts share the same fusion idea:

1. Encode the sentence with BERT-base or RoBERTa-base and take `pooler_output`.
2. Project five gaze numbers through a small linear layer (`5 → 16`).
3. Concatenate the 768-d text vector with the 16-d gaze vector.
4. Dropout 0.1, then a 3-way linear classifier.

Text-only controls (`bert`, `roberta`) skip step 2–3 and use Hugging Face
`*ForSequenceClassification` instead.

## Sentiment labels

Used everywhere after the folder-of-text conversion step:

| Folder / string | Integer | Meaning |
| --- | --- | --- |
| `NEGATIVE` | `0` | Negative review / sentence |
| `NEUTRAL` | `1` | Neutral |
| `POSITIVE` | `2` | Positive |

`SST_data/stts_all_sentence_level.csv` still uses the string labels and has
**no header row**. Everything under `*_full_sst.csv` and `ZuCo_SST_data/`
uses the integer scheme above.

## Gaze features the classifiers see

The fusion models read **five** columns (names differ slightly by table):

| Training table | Count | Duration features |
| --- | --- | --- |
| `ZuCo_SST_data/combined_sst_et_standard.csv` | `nFixations` | `FFD`, `GPT`, `TRT`, `GD` |
| `SST_data/*_full_sst.csv` | `nFix` | `FFD`, `GPT`, `TRT`, `GD` |

Sentence-level ZuCo tables also keep `omissionRate`, `meanPupilSize`, and
`SFD`. Those extra columns are **not** passed into `EyeTrackingModel` today.
See [docs/03-eye-tracking-features.md](docs/03-eye-tracking-features.md).

## Dataset sizes (counted from the CSVs in this checkout)

**ZuCo ∩ SST (measured gaze)** — 400 sentences, fairly balanced:

- negative `0`: 123
- neutral `1`: 137
- positive `2`: 140
- mean length ≈ 17.8 whitespace tokens (range 3–43)

Held-out CSVs exist (`train.csv` 320 / `valid.csv` 40 / `test.csv` 40) but
`model_ZuCo_SST.py` ignores them and re-splits with `StratifiedKFold`.

**Full SST + transferred gaze** — 11,853 sentences:

| Split | Rows | neg / neu / pos |
| --- | ---: | --- |
| train | 9,482 | 3710 / 1833 / 3939 |
| valid | 1,185 | 476 / 209 / 500 |
| test | 1,186 | 463 / 199 / 524 |
| all | 11,853 | 4649 / 2241 / 4963 |

Neutral is the minority class on the full split. Mean length ≈ 19.2 tokens.

**Per-subject ZuCo sentence tables** (`ZuCo_et_csv_data/{1–12}_SR.csv`):
most subjects have 400 rows; subject `3_SR.csv` has 299 because
`DataTransformer` drops known-bad sentence ranges (see `utils_ZuCo.py`).

**Word-level averages**: 7,129 tokens in
`ZuCo_et_csv_data/word/word_averages_v2.csv`.

## Lightweight examples (no GPU, no transformers)

The scripts under `examples/` only need the Python standard library and
`numpy` (already present in this environment):

```bash
python3 examples/01_inspect_datasets.py
python3 examples/02_schema_check.py
python3 examples/03_gaze_feature_summary.py
python3 examples/04_label_and_length.py
python3 examples/05_word_level_gaze.py
python3 examples/06_toy_late_fusion.py
```

They print tables, write `examples/sample_outputs/`, and include a tiny
gaze-only softmax baseline so the late-fusion idea is visible without
downloading BERT weights. Details: [examples/README.md](examples/README.md).

## Full model training (heavy)

```bash
python3 -m pip install -r requirements.txt

# measured gaze, 5-fold CV
# edit model_type in the script: bert | roberta | bert_eye_tracking | roberta_eye_tracking
python3 model_ZuCo_SST.py

# transferred gaze, saved checkpoint under models/
mkdir -p models
python3 model_full_SST.py
```

A CUDA GPU is strongly recommended. The ZuCo script trains **five** fresh
transformers; the full-SST script trains one model for five epochs at batch
256.

MATLAB `.mat` files for ZuCo are **not** in this checkout. `read_ZuCo_mat.py`
expects `ZuCo_mat_data/task1/` with 12 subject files. The derived CSVs are
what the rest of the repo uses.

## Documentation

- [Project overview](docs/01-project-overview.md) — motivation and claims
- [Datasets](docs/02-datasets.md) — every CSV, column, and split
- [Eye-tracking features](docs/03-eye-tracking-features.md) — nFix, FFD, GPT, TRT, GD, SFD
- [Preprocessing pipeline](docs/04-preprocessing-pipeline.md) — MATLAB → averages → SST join
- [Model architecture](docs/05-model-architecture.md) — fusion diagram and code map
- [Training loops](docs/06-training-loops.md) — optimizers, metrics, a test-set bug
- [Running experiments](docs/07-running-experiments.md) — knobs and expected artifacts
- [Known quirks](docs/08-known-quirks.md) — path separators, overwrites, missing files
- [Glossary](docs/glossary.md)

## Sources (external)

- **ZuCo**: Hollenstein et al., *ZuCo, a simultaneous EEG and eye-tracking
  corpus for natural sentence reading*. Task 1 in this repo is the sentiment
  / movie-review reading task.
- **SST**: Socher et al., Stanford Sentiment Treebank. The 400-sentence
  overlap is the subset that ZuCo participants actually read.
- Encoders: `bert-base-uncased` and `roberta-base` from Hugging Face.

## License / use

Personal research notes and scripts. Do not treat numbers printed by the
example baselines as paper-ready results.
