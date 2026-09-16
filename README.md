# Transformer emotion analysis with gaze

Personal research repo: fuse **sentence-level eye-tracking features** onto
BERT / RoBERTa and train 3-class sentiment (negative / neutral / positive).

This is a public personal project (`sapphireran/Transformer-Emotion-Analysis-with-Gaze`).
It does not contain workplace code, private messages, or proprietary data.

Two tracks share a label map and a concat head, not a gaze source:

| Track | Gaze | Size | Trainer |
| --- | --- | ---: | --- |
| ZuCo movie-review subset | recorded, 12 readers, then averaged | 400 sentences | `model_ZuCo_SST.py` (5-fold CV) |
| Full SST | predicted / projected | 11,853 sentences | `model_full_SST.py` (80/10/10) |

```
sentence ──► BERT or RoBERTa pooler (768)
                              ├─ concat ─► dropout ─► Linear ─► 3 logits
5 gaze numbers ──► Linear(5, 16) ─┘
```

## Layout

```
model_ZuCo_SST.py          5-fold trainer on real ZuCo gaze
model_full_SST.py          train/valid/test trainer on predicted gaze
utils_ZuCo.py              MATLAB → DataFrame (needs ZuCo .mat files)
read_ZuCo_mat.py           writes per-subject sentence CSVs
get_average_sentence_level.py
convert_full_SST.py        txt folders → ssts_ZuCo.csv

ZuCo_SST_data/             400 sentences + joined ET + optional 80/10/10
ZuCo_et_csv_data/          per-reader sentence and word ET
SST_data/                  full SST + predicted sentence ET
gaze_prediction/data/      word-level predicted gaze, PROVO extract

tea_gaze/                  helpers used only by docs/examples
docs/                      personal research notes
examples/                  runnable walkthroughs (no GPU)
tests/                     checks against the checked-in CSVs
```

## Docs and examples

Start at [docs/README.md](docs/README.md). The notes cover datasets, the
column dictionary, gaze-feature meanings, architecture, the script pipeline,
reproduction commands, and known bugs in the original trainers.

```bash
python -m pip install -r requirements.txt
export PYTHONPATH=.
python examples/01_inspect_datasets.py
python examples/05_gaze_only_baseline.py
python -m pytest tests/test_tea_gaze.py -q
```

All eight examples are listed in [examples/README.md](examples/README.md).
They do **not** download Hugging Face weights.

## Training (original scripts)

```bash
mkdir -p models
# edit model_type at the top of the file: bert | roberta | bert_eye_tracking | roberta_eye_tracking
python model_ZuCo_SST.py
python model_full_SST.py
```

`model_full_SST.py` writes `models/best_<model_type>_model.pth` when
validation **accuracy** improves. The test-loop accumulation has a known
overwrite bug; see [docs/known-issues.md](docs/known-issues.md) before quoting
a test number.

## Labels and features

```
NEGATIVE = 0
NEUTRAL  = 1
POSITIVE = 2
```

Trainers concatenate five measures: fixation count, first fixation duration,
go-past time, total reading time, gaze duration. ZuCo spells the count
`nFixations`; full SST spells it `nFix`. Linear correlation of those
features with sentiment is weak (~0.05). Predicted full-SST features are
nearly collinear with each other. Details and tables:
[docs/gaze-features.md](docs/gaze-features.md),
[docs/experiment-notes.md](docs/experiment-notes.md).

## Data sources

Public only: [ZuCo](docs/citations.md), Stanford Sentiment Treebank, a small
PROVO extract. Raw `.mat` recordings are not in git; the derived CSVs are.

## Status of this personal snapshot

The concat-fusion trainers and the joined CSVs are here. The docs/examples
tree is the lab notebook I use to inspect those files without rerunning a
GPU job. A seeded `roberta` vs `roberta_eye_tracking` comparison still
belongs in [docs/experiment-notes.md](docs/experiment-notes.md) once I have
a clean GPU run.
