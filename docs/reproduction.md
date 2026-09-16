# Reproduction notes (personal laptop)

These are the commands I use on this repo. They assume the working directory
is the repository root.

## Environment

Python 3.10+ is enough for the example library. The original trainers also
need PyTorch, Hugging Face `transformers` / `datasets`, and a download of
`bert-base-uncased` or `roberta-base`.

```bash
python -m pip install -r requirements.txt
```

`requirements.txt` splits the heavy training extras into a comment so a
docs-only session can stay on pandas / numpy / scikit-learn.

Add the repo root to `PYTHONPATH` when running examples as scripts:

```bash
export PYTHONPATH=.
python examples/01_inspect_datasets.py
```

Or:

```bash
python -m examples.inspect   # not wired; use the numbered scripts
```

## Example suite (no GPU, no model download)

```bash
export PYTHONPATH=.
python examples/01_inspect_datasets.py
python examples/02_label_and_feature_stats.py
python examples/03_compare_scaling.py
python examples/04_toy_fusion_forward.py
python examples/05_gaze_only_baseline.py
python examples/06_word_to_sentence.py
python examples/07_schema_and_glossary.py
python examples/08_split_sanity.py
```

`tests/test_tea_gaze.py` covers the same helpers with the checked-in CSVs:

```bash
export PYTHONPATH=.
python -m pytest tests/test_tea_gaze.py -q
```

## Original trainers (GPU optional, downloads weights)

```bash
mkdir -p models
python model_ZuCo_SST.py      # 5-fold CV on 400 rows; slow on CPU
python model_full_SST.py      # writes models/best_<model_type>_model.pth
```

Edit the `model_type` string at the top of each file. Valid values:

```
bert
roberta
bert_eye_tracking
roberta_eye_tracking
```

`model_full_SST.py` will crash on `torch.save` if `models/` does not exist.
Create it first.

## Rebuilding derived CSVs (only if you have the raw sources)

These steps need files that are **not** in this git snapshot.

```bash
# ZuCo MATLAB dumps in ZuCo_mat_data/task1/
python read_ZuCo_mat.py

# after pointing folder_path at ZuCo_et_csv_data
python get_average_sentence_level.py

# raw per-label txt files in ZuCo_SST_data/all/
python convert_full_SST.py

# 80/10/10 from the already-combined tables
python ZuCo_SST_data/spilt.py
python SST_data/spilt.py
```

I do not rerun those unless I am replacing a table on purpose. The committed
CSVs are the experiment record.

## Seeds that are already baked in

| Script | Seed | Effect |
| --- | --- | --- |
| `model_ZuCo_SST.py` | `random_state=42` | StratifiedKFold shuffle |
| `ZuCo_SST_data/spilt.py` | `random_state=42` | 80/10/10 |
| `SST_data/spilt.py` | `random_state=42` | 80/10/10 |
| `examples/04_toy_fusion_forward.py` | `seed=7` | toy ET projection + logistic |
| `examples/05_gaze_only_baseline.py` | `seed=42` | baseline KFold |

PyTorch itself is not seeded in the original trainers, so GPU fusion runs
will move around even with the same CSV split.

## Hardware notes

- ZuCo CV, batch 16, 20 epochs × 5 folds, RoBERTa: plan on a GPU.
- Full SST, batch 256, 5 epochs: also a GPU unless you like overnight jobs.
- The example suite finishes in well under a minute on CPU.

## What “reproduced” means for this personal repo

1. Example scripts print the same schema / label counts as
   [datasets.md](datasets.md).
2. `pytest` is green.
3. A trainer run, if I have GPU time, prints fold or valid metrics I can
   paste into [experiment-notes.md](experiment-notes.md).

I do not treat a single unseeded GPU run as a final number.
