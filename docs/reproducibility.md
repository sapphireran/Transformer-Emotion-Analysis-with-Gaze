# Reproducibility

Personal notes so a later checkout can recreate the *documented* path without guessing.

## What is frozen in git

- The 400-row ZuCo+SST joins and their 80/10/10 cut
- Per-reader sentence CSVs (`1_SR.csv` … `12_SR.csv`) and the averages
- Word-level averages (`word_averages_v2.csv`)
- Full SST + predicted gaze splits
- A PROVO extract and two predictor dumps
- The two transformer scripts and the MATLAB helper

MATLAB `.mat` files, the `all/NEGATIVE|POSITIVE|NEUTRAL` text folders, and Hugging Face weights are **not** in git.

## Recreate the CPU docs/examples

Python 3.10+ is enough. From the repo root:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
PYTHONPATH=. pytest
PYTHONPATH=. python examples/run_all.py
```

Seeds that matter for the examples:

| Place | Seed | Why |
|---|---|---|
| `tea_gaze.baselines.cross_validate` | `random_state=42` | same `StratifiedKFold` as `model_ZuCo_SST.py` |
| `examples/04_sentence_walkthrough.py` | `train_test_split(..., 42)` | stable 80% fit for the write-up |
| `examples/05_full_sst_sample.py` | `sample(..., 7)` and `800, random_state=1` | stable preview rows |

`tea_gaze.eval.weighted_metrics` is a from-scratch weighted P/R/F1 so the markdown reports do not depend on a particular sklearn metrics version. The logistic models themselves do depend on sklearn.

## Recreate Track A (measured gaze, transformer)

1. Install `torch`, `transformers`, `datasets`, `tqdm`, `scikit-learn`, `pandas`.
2. Leave `dataset_path = 'ZuCo_SST_data/combined_sst_et_standard.csv'`.
3. Set `model_type` to one of `bert`, `roberta`, `bert_eye_tracking`, `roberta_eye_tracking`.
4. Run `python model_ZuCo_SST.py`.
5. Record the five fold lines and the four averages. There is no checkpoint.

GPU vs CPU changes runtime, not the split (the KFold seed is fixed). Tokenizer downloads need Hub access on first run.

## Recreate Track B (predicted gaze, transformer)

1. `mkdir -p models`
2. Confirm `SST_data/train_full_sst.csv` (9482), `valid_full_sst.csv` (1185), `test_full_sst.csv` (1186).
3. Fix the test-loop `extend` bug in [known-issues.md](known-issues.md) if you want a real test score.
4. Run `python model_full_SST.py`.
5. Best snapshot path: `models/best_{model_type}_model.pth` selected by validation **accuracy**.

## Regenerating CSVs (optional)

Only if you have the official ZuCo `.mat` release:

1. Put 12 files per task under `ZuCo_mat_data/task1` (and fix the backslash path in `get_matfiles`).
2. `python read_ZuCo_mat.py` → per-reader CSVs.
3. Point `get_average_sentence_level.py` at that folder and re-scale.
4. Re-join labels from `ssts_ZuCo.csv` (or rebuild labels from the uncommitted `all/` folder).

Full SST regeneration needs the original SST file, NLTK `punkt`, and whatever trained predictor produced `nFix`. That predictor code is not fully checked in — only `gaze_prediction/data/convert_zuco_data.py` and the output tables.

## How to tell measured from predicted

| Clue | Measured ZuCo | Predicted full SST |
|---|---|---|
| Header | `nFixations` | `nFix` |
| Extra columns | `omissionRate`, `meanPupilSize`, `SFD` | none |
| Rows | 400 | ~11.8k |
| Neutral share | ~34% | ~19% |

`tea_gaze.paths.DATASETS` records `gaze_source` as `measured` or `predicted` so examples do not have to guess.
