# Transformer Emotion Analysis with Gaze

Personal research workspace: **three-class sentiment** on movie-review
sentences, with an optional **eye-tracking** branch fused into BERT or
RoBERTa.

This clone keeps the original training scripts and CSVs, and adds a
docs/examples layer that runs on the Python standard library — no GPU,
no Hugging Face download.

```
text ──► BERT / RoBERTa pooler (768) ──┐
                                       ├─► concat ──► dropout ──► Linear(784 → 3)
gaze ──► Linear(5 → 16) ───────────────┘
```

Labels: `0` negative, `1` neutral, `2` positive.

## Why two data tracks?

| Track | Rows | Gaze source | Training script |
| --- | ---: | --- | --- |
| ZuCo–SST | 400 | averaged human recordings (12 readers) | `model_ZuCo_SST.py` (5-fold) |
| Full SST | 11,853 | projected / predicted features | `model_full_SST.py` (80/10/10) |

Use ZuCo when you care about real reading behavior. Use full SST when
you care about scale. Do not interpret full-SST `TRT` as a human
measurement.

## Read this before quoting a number

1. **Reader 3 is remapped.** `ZuCo_et_csv_data/3_SR.csv` has 299 rows.
   Original sentences 150–249 and 399 were dropped, then ids were
   compacted to `0..298`. The published averages group by row index, so
   246 sentence means mix in the wrong reader-3 sentence. Details:
   [docs/subject-alignment.md](docs/subject-alignment.md).
2. **The full-SST test loop keeps the last batch.** Printed test
   metrics cover 162 of 1,186 rows at `batch_size=256`. Validation
   numbers from the same script are fine.
   [docs/known-bugs.md](docs/known-bugs.md).

## Repository map

```
docs/                 personal reading notes + generated notebook
examples/             stdlib inspectors (inventory, alignment, baselines)
zuco_lab/             helpers those examples import
tests/                unittest for zuco_lab
model_ZuCo_SST.py     RoBERTa/BERT ± gaze, stratified 5-fold
model_full_SST.py     same models, fixed SST split + checkpoint
utils_ZuCo.py         MATLAB → sentence/word tables
ZuCo_SST_data/        400 labeled sentences + scaled gaze
ZuCo_et_csv_data/     per-reader and averaged ET
SST_data/             11k labeled sentences + projected gaze
gaze_prediction/data/ word-level predicted ET + PROVO
```

## Run the examples (no extra packages)

```bash
python3 examples/run_all.py
python3 -m unittest discover -s tests -v
```

Or one at a time — see [examples/README.md](examples/README.md).
Snapshots land in `examples/outputs/` and `docs/generated/`.

## Train the transformer models

```bash
pip install -r requirements.txt
mkdir -p models
python model_ZuCo_SST.py      # small, 5-fold
python model_full_SST.py      # large; wants a GPU
```

Set `model_type` at the top of either script to `bert`, `roberta`,
`bert_eye_tracking`, or `roberta_eye_tracking`.

## Docs

- [Project map](docs/project-map.md)
- [Datasets](docs/datasets.md)
- [Feature dictionary](docs/feature-dictionary.md)
- [Subject alignment](docs/subject-alignment.md)
- [Pipeline](docs/pipeline.md)
- [Models](docs/models.md)
- [Known bugs](docs/known-bugs.md)
- [Hypotheses](docs/hypotheses.md)
- [Glossary](docs/glossary.md)
- [Generated notebook](docs/generated/lab_notebook.md)

## Personal layer

`zuco_lab` only reads the checked-in tables. It does not rewrite
`average_data.csv` or patch the training loops, so a later GPU run
still matches this clone. The subject-3 remapping and the test-loop
overwrite are documented on purpose.
