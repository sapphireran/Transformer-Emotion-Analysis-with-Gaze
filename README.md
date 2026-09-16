# Transformer Emotion Analysis with Gaze

Personal research code for **sentiment classification** that fuses a transformer
text encoder (BERT or RoBERTa) with **eye-tracking (gaze) features**.

The project has two tracks:

1. **Human gaze on ZuCo–SST overlap.** About 400 Stanford Sentiment Treebank
   (SST) movie-review sentences that were also read in the Zurich Cognitive
   Language Processing Corpus (ZuCo). Real fixation statistics from 12 readers
   are averaged, scaled, and concatenated with the transformer pooled
   representation.
2. **Predicted gaze on full SST.** The rest of SST has no human recordings.
   Word-level gaze is predicted (see `gaze_prediction/`), aggregated to the
   sentence, and fed to the same fusion head so the larger SST splits can be
   trained end to end.

This repository is a personal experiment notebook in code form: MATLAB ZuCo
dumps are flattened to CSV, SST labels are joined on `sentence_id`, and two
training scripts compare text-only vs. text+gaze models.

## Quick map

| Path | Role |
| --- | --- |
| `utils_ZuCo.py` | `DataTransformer` — MATLAB `.mat` → sentence/word tables, NaN fill, scaling |
| `read_ZuCo_mat.py` | Export 12 subject CSVs for ZuCo task 1 (normal reading) |
| `get_average_sentence_level.py` | Mean across subjects, then min-max / z-score |
| `ZuCo_et_csv_data/` | Per-subject and averaged **sentence-level** gaze |
| `ZuCo_et_csv_data/word/` | Per-subject and averaged **word-level** gaze |
| `ZuCo_SST_data/` | 400 SST sentences + joined gaze + 80/10/10 split |
| `SST_data/` | Full SST sentences, predicted sentence-level gaze, train/valid/test |
| `gaze_prediction/data/` | Predicted word-level gaze (SST + a PROVO sample) |
| `model_ZuCo_SST.py` | 5-fold CV on the 400-sentence human-gaze set |
| `model_full_SST.py` | Train / early-stop / test on full SST + predicted gaze |
| `docs/` | Design notes, feature glossary, pipeline, pitfalls |
| `examples/` | Runnable inspections and a toy fusion forward pass |

Read [`docs/README.md`](docs/README.md) for the long-form notes, then run the
scripts in [`examples/`](examples/README.md) against the CSVs already in the
tree. You do **not** need GPU, PyTorch, or the original `.mat` files for the
examples.

## Labels

Sentiment is stored as integers in the joined tables:

| Integer | Meaning | SST folder / string |
| --- | --- | --- |
| `0` | negative | `NEGATIVE` |
| `1` | neutral | `NEUTRAL` |
| `2` | positive | `POSITIVE` |

Training heads use `num_labels = 3` and `CrossEntropyLoss`.

## Gaze vector used at train time

Both training scripts feed **five** sentence-level features into a linear
projection (`5 → 16`) and concatenate that with the transformer's pooled
hidden state (`768` for `bert-base` / `roberta-base`):

| ZuCo human table | Full-SST predicted table | Psycholinguistic name |
| --- | --- | --- |
| `nFixations` | `nFix` | fixation count |
| `FFD` | `FFD` | first fixation duration |
| `GPT` | `GPT` | go-past / regression-path time |
| `TRT` | `TRT` | total reading time |
| `GD` | `GD` | gaze duration (first pass) |

The human tables also keep `omissionRate`, `meanPupilSize`, and `SFD` for
analysis; those three are **not** in the fusion head today. See
[`docs/gaze-features.md`](docs/gaze-features.md).

## Training (optional)

The two model scripts are self-contained and still use the original hard-coded
paths and hyperparameters. They expect PyTorch, Hugging Face `transformers`,
and a CUDA device if you do not want to train on CPU.

```bash
pip install -r requirements.txt

# Human gaze, 5-fold stratified CV (~400 sentences)
python model_ZuCo_SST.py

# Predicted gaze, full SST train/valid/test
python model_full_SST.py
```

Switch `model_type` inside each file among:

- `bert` / `roberta` — text only (`*ForSequenceClassification`)
- `bert_eye_tracking` / `roberta_eye_tracking` — fusion `EyeTrackingModel`

Checkpoints from the full-SST script are written to
`models/best_{model_type}_model.pth`. Create that directory first.

Known issues in the original training loops (test-set metric overwrite, ignored
`batch_size` in the ZuCo CV loader) are listed in
[`docs/known-pitfalls.md`](docs/known-pitfalls.md). The example scripts show
the **intended** metric accumulation, not the buggy loop.

## Examples (no GPU)

```bash
python examples/inspect_sentence_gaze.py
python examples/inspect_word_gaze.py
python examples/subject_coverage.py
python examples/label_and_gaze_summary.py
python examples/sst_split_sanity.py
python examples/toy_fusion_forward.py
python examples/gaze_only_baseline.py
python examples/metrics_example.py
```

## Data provenance (personal notes)

- **ZuCo** — Hollenstein et al., *Scientific Data*, 2018. Simultaneous EEG and
  eye-tracking during naturalistic reading. This repo uses **task 1** (normal
  reading of movie reviews) exported from MATLAB `sentenceData` structs.
- **SST** — Socher et al., EMNLP 2013. Fine-grained movie-review sentiment.
  The 400-sentence overlap is the subset of SST items that ZuCo participants
  actually read.
- **PROVO** — a small predicted-gaze sample lives under
  `gaze_prediction/data/provo.csv` (includes `fixProp` instead of `GD`).

MATLAB source dumps (`ZuCo_mat_data/`) are **not** checked in. CSV exports
under `ZuCo_et_csv_data/` are.

## License / use

Personal research code. Datasets remain under their original terms (ZuCo,
SST, PROVO). Do not treat the predicted gaze columns as human recordings.
