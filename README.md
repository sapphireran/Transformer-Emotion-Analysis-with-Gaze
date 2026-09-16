# Transformer Emotion Analysis with Gaze

Personal research code for **3-way movie-review sentiment** when a transformer
sees both the sentence and a compact **eye-tracking** vector.

The repo joins two public resources:

- **ZuCo** (Zurich Cognitive Language Processing Corpus): EEG + eye tracking
  recorded while 12 people read English sentences. Task 1 here is sentiment
  reading (`*_SR.csv`).
- **Stanford Sentiment Treebank (SST)**: the same review domain at a much
  larger scale. Full SST does not come with real gaze, so those tables carry
  **projected** nFix / FFD / GPT / TRT / GD features.

There are two experimental tracks, each with its own training script:

| Track | Script | Gaze source | Protocol | Scale |
| --- | --- | --- | --- | --- |
| ZuCo-SST | `model_ZuCo_SST.py` | real subject-averaged ET | 5-fold stratified CV | 400 sentences |
| Full SST | `model_full_SST.py` | projected gaze on SST | 80 / 10 / 10 split | 11,853 sentences |

Both tracks can run **text-only** BERT/RoBERTa or the **late-fusion** variants
`bert_eye_tracking` / `roberta_eye_tracking`.

This is a personal project. The original scripts are research notebooks in
`.py` form. The `docs/`, `examples/`, and `gaze_emotion/` tree added later
document those scripts and let you inspect the checked-in CSVs without
downloading Hugging Face weights.

## Repository map

```
model_ZuCo_SST.py          # 5-fold CV on ZuCo-aligned SST sentences
model_full_SST.py          # train / val / test on full SST + projected gaze
utils_ZuCo.py              # MATLAB -> sentence/word tables, scaling, NaNs
read_ZuCo_mat.py           # dump 12 Task-1 subjects to CSV
get_average_sentence_level.py
convert_full_SST.py

ZuCo_et_csv_data/          # per-subject sentence ET, plus averages
ZuCo_et_csv_data/word/     # per-subject word ET, plus averages
ZuCo_SST_data/             # 400 SST sentences joined to averaged ET
SST_data/                  # full SST sentences + projected ET
gaze_prediction/data/      # word-level predicted gaze (SST, PROVO)
result/                    # scatter / histogram plots from earlier runs

docs/                      # feature dictionary, pipeline, caveats
examples/                  # runnable inspections (no GPU required)
gaze_emotion/              # small library used by the examples and tests
```

## Gaze features in one paragraph

Readers do not look at every word equally. The fusion models keep five
sentence-level channels that psycholinguistics already treats as difficulty
and rereading signals: **nFixations**, **FFD**, **GPT**, **TRT**, **GD**.
ZuCo tables also store omission rate, pupil size, and single-fixation
duration. The [gaze feature guide](docs/gaze-features.md) defines each one.

Labels are `0 = NEGATIVE`, `1 = NEUTRAL`, `2 = POSITIVE`.

## Quick start (docs and examples)

The examples only need NumPy, pandas, and (for tests) pytest. They do **not**
download BERT or RoBERTa.

```bash
python3 -m pip install -r requirements-dev.txt
python3 examples/inspect_datasets.py
python3 examples/gaze_feature_tour.py
python3 examples/label_and_split_audit.py
python3 examples/fusion_architecture_demo.py
python3 examples/synthetic_training_loop.py
python3 -m pytest tests -q
```

Or run the whole suite:

```bash
bash examples/run_all.sh
```

## Training the original models

Those scripts need PyTorch, Hugging Face `transformers` / `datasets`, and a
GPU if you want more than a smoke test.

```bash
python3 -m pip install torch transformers datasets tqdm scikit-learn pandas
python3 model_ZuCo_SST.py      # 5 folds, 20 epochs, batch 16
python3 model_full_SST.py      # 5 epochs, batch 256, writes models/best_*.pth
```

Switch the architecture by editing `model_type` at the top of the script:

- `bert` / `roberta` — text only
- `bert_eye_tracking` / `roberta_eye_tracking` — pooler output ⊕ projected gaze

Shared hyperparameters (also listed in `gaze_emotion.constants`):

- 5 gaze channels, hidden projection size 16, 3 labels
- max token length 128, dropout 0.1, Adam `5e-5`

See [models and training](docs/models-and-training.md) and
[reproducing experiments](docs/reproducing-experiments.md).

## Data sizes (checked-in CSVs)

**ZuCo-SST (standard-scaled, subject-averaged)**

| Split | Rows | NEG / NEU / POS |
| --- | ---: | --- |
| combined | 400 | 123 / 137 / 140 |
| train.csv | 320 | 103 / 107 / 110 |
| valid.csv | 40 | 7 / 14 / 19 |
| test.csv | 40 | 13 / 16 / 11 |

**Full SST + projected gaze**

| Split | Rows | NEG / NEU / POS |
| --- | ---: | --- |
| combined | 11,853 | 4,649 / 2,241 / 4,963 |
| train | 9,482 | 3,710 / 1,833 / 3,939 |
| valid | 1,185 | 476 / 209 / 500 |
| test | 1,186 | 463 / 199 / 524 |

Subject 3's sentence file has 299 trials, not 400 — the converter skips known
bad recording ranges. Details are in [datasets](docs/datasets.md).

## Documentation

- [Project overview](docs/overview.md)
- [Gaze feature dictionary](docs/gaze-features.md)
- [Dataset catalog](docs/datasets.md)
- [Data pipeline](docs/data-pipeline.md)
- [Models and training](docs/models-and-training.md)
- [Reproducing experiments](docs/reproducing-experiments.md)
- [Known issues](docs/known-issues.md)
- [Examples](examples/README.md)

## License and data

Code in this repository is personal research. ZuCo and SST remain under their
upstream licenses; do not redistribute those corpora beyond what their owners
allow. MATLAB source files are **not** checked in — only derived CSVs.
