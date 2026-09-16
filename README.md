# Transformer Emotion Analysis with Gaze

Personal research archive for **three-class sentiment classification** on Stanford
Sentiment Treebank (SST) sentences, with optional **eye-tracking / gaze
features** fused into BERT or RoBERTa.

The working hypothesis is that reading-time signals — how long a reader
fixates, how often they regress, how many words they skip — carry information
about affective processing that a text-only transformer does not see. This
repo stores the datasets, preprocessing scripts, two training entry points,
and (in `docs/` + `examples/`) a walkthrough of the data and the fusion
architecture **without** requiring a GPU or a Hugging Face download.

This is a **personal** project. Nothing here is company code.

## What is in the box

| Path | Role |
| --- | --- |
| `model_full_SST.py` | Train BERT/RoBERTa ± gaze on the full SST holdout split (~11.8k sentences). |
| `model_ZuCo_SST.py` | Same architecture, 5-fold CV on the 400-sentence ZuCo SST subset with **measured** gaze. |
| `utils_ZuCo.py` / `read_ZuCo_mat.py` | Read ZuCo `.mat` files and emit per-subject CSVs. |
| `SST_data/` | Full SST sentences + standardized sentence-level gaze features, already split. |
| `ZuCo_SST_data/` | 400 ZuCo-aligned SST sentences with real subject-averaged eye tracking. |
| `ZuCo_et_csv_data/` | Per-subject sentence-level and word-level gaze (12 readers). |
| `gaze_prediction/` | Predicted word-level gaze on SST, plus a PROVO reference extract. |
| `result/` | Exploratory scatter/histogram plots of predicted vs PROVO gaze. |
| `docs/` | Datasets, features, models, pipeline, training notes, known issues. |
| `examples/` | Runnable analysis scripts that only need pandas/numpy/scikit-learn. |

## Labels

All numeric labels in the processed CSVs use the same mapping:

| Label | Meaning |
| ---: | --- |
| `0` | Negative |
| `1` | Neutral |
| `2` | Positive |

The raw SST dump `SST_data/stts_all_sentence_level.csv` still uses the
strings `NEGATIVE` / `NEUTRAL` / `POSITIVE` and has **no header row**.

## Two experimental tracks

1. **Measured gaze (ZuCo).** Twelve participants read 400 SST sentences while
   eye movements were recorded. Sentence-level features are averaged across
   subjects, then standardized or min-max scaled. `model_ZuCo_SST.py` runs
   stratified 5-fold cross-validation (20 epochs, batch size 16).
2. **Predicted / aligned gaze (full SST).** The full ~11,853-sentence SST
   table carries five standardized gaze columns (`nFix`, `GD`, `TRT`, `FFD`,
   `GPT`). `model_full_SST.py` uses an 80/10/10 split (batch size 256, 5
   epochs) and checkpoints the best validation **accuracy**.

Both tracks share the same fusion idea: take the transformer `pooler_output`,
project the five gaze features through a small linear layer (`5 → 16`),
concatenate, drop out, and classify.

```
sentence ──► BERT / RoBERTa ──► pooler (768)
                                    │
gaze [5] ──► Linear 5→16 ───────────┴──► concat (784) ──► Dropout ──► Linear ──► 3 logits
```

Text-only ablations (`bert`, `roberta`) skip the gaze branch and use
`*ForSequenceClassification` from Hugging Face.

## Quick start (docs and examples)

These commands do **not** train a transformer. They inspect the CSVs that
are already in the repo.

```bash
python3 -m pip install -r requirements.txt
python3 examples/01_inspect_datasets.py
python3 examples/run_all.py --output-dir examples/output
python3 -m unittest discover -s tests -v
```

See [examples/README.md](examples/README.md) for the full script list and
[docs/README.md](docs/README.md) for the written notes.

## Train (optional, heavy)

```bash
python3 -m pip install -r requirements-train.txt
# edit model_type in the script: bert | roberta | bert_eye_tracking | roberta_eye_tracking
python3 model_ZuCo_SST.py      # 5-fold CV on 400 ZuCo sentences
python3 model_full_SST.py      # holdout on full SST; writes models/best_<type>_model.pth
```

A CUDA GPU is assumed for anything but a smoke test. The scripts print
`Using device: cpu` if no GPU is visible.

More detail: [docs/training_and_evaluation.md](docs/training_and_evaluation.md).

## Dataset snapshot

Counts were taken from the CSVs in this checkout:

| File | Rows | What it is |
| --- | ---: | --- |
| `SST_data/combined_full_sst_et.csv` | 11,853 | Full SST + 5 standardized gaze columns. Labels: 4,649 / 2,241 / 4,963 (neg / neu / pos). |
| `SST_data/train_full_sst.csv` | 9,482 | 80% split (`random_state=42`). |
| `SST_data/valid_full_sst.csv` | 1,185 | 10% split. |
| `SST_data/test_full_sst.csv` | 1,186 | 10% split. |
| `ZuCo_SST_data/combined_sst_et_standard.csv` | 400 | ZuCo SST + standardized gaze. Labels: 123 / 137 / 140. |
| `ZuCo_et_csv_data/{1–12}_SR.csv` | 400 × 12 | Per-subject sentence-level raw gaze. |
| `ZuCo_et_csv_data/word/{1–12}_SR.csv` | 7,129 × 12 | Per-subject word-level gaze. |
| `gaze_prediction/data/prediction_test_v2.csv` | 191,971 | Word-level predicted gaze for the full SST. |
| `gaze_prediction/data/provo.csv` | 2,659 | PROVO reference words (`fixProp` instead of `GD`). |

## Eye-tracking features used by the classifiers

| Column (full SST) | Column (ZuCo) | Psycholinguistic meaning |
| --- | --- | --- |
| `nFix` | `nFixations` | Fixation count (how often the eyes land). |
| `FFD` | `FFD` | First fixation duration. |
| `GPT` | `GPT` | Go-past / regression-path time. |
| `TRT` | `TRT` | Total reading time. |
| `GD` | `GD` | Gaze duration (first-pass time). |

ZuCo tables also keep `omissionRate`, `meanPupilSize`, `SFD`, and `SentLen`.
Those are useful for analysis (see `examples/03_gaze_feature_stats.py`) but
are **not** concatenated into the classifier.

Feature definitions: [docs/eye_tracking_features.md](docs/eye_tracking_features.md).

## Sources (personal research citations)

- **SST** — Socher et al., *Recursive Deep Models for Semantic Compositionality
  Over a Sentiment Treebank*, EMNLP 2013.
- **ZuCo** — Hollenstein et al., *ZuCo, a simultaneous EEG and eye-tracking
  resource for natural sentence reading*, Scientific Data 2018.
- **PROVO** — Luke & Christianson, *The Provo Corpus: A large eye-tracking
  corpus with predictability norms*, Behavior Research Methods 2018.
- **BERT** — Devlin et al., NAACL 2019. **RoBERTa** — Liu et al., 2019.

Raw ZuCo `.mat` files and the original SST release are **not** vendored
here. The processed CSVs are.

## License / use

Personal research notebook in repo form. SST, ZuCo, and PROVO remain under
their upstream licenses. Do not treat the predicted gaze columns as
measured human eye tracking.
