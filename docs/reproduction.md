# Reproduction

How to re-run the personal pipeline on this checkout. The MATLAB originals are not in git; everything from sentence-level CSVs onward is.

## Environment

Python 3.12 was used for the CPU examples in this branch.

```bash
python3 -m pip install -r requirements.txt
```

Transformer training needs extra packages (not pinned here because CUDA wheels vary):

```bash
python3 -m pip install torch transformers datasets tqdm scikit-learn pandas numpy
```

Create `models/` before a full-SST run; `torch.save` does not create the directory:

```bash
mkdir -p models result
```

GPU is optional for the examples, effectively required for `batch_size=256` RoBERTa.

## Run the documentation examples

From the repository root (scripts resolve paths from `cwd`):

```bash
python3 examples/inspect_datasets.py
python3 examples/gaze_feature_stats.py
python3 examples/sentiment_baselines.py
python3 examples/fusion_toy.py
python3 examples/word_level_gaze.py
```

`examples/inspect_datasets.py` exits non-zero if a committed table is missing a required column or a sentiment class. Use it as a cheap regression check after you touch CSVs.

## Train (ZuCo track)

```bash
# edit model_type near the top of the file, then:
python3 model_ZuCo_SST.py
```

Expect five folds × 20 epochs of `tqdm` bars and a final block:

```text
Average Validation Acc: ...
Average Validation P: ...
Average Validation R: ...
Average Validation F1: ...
```

There is no checkpoint. If you want weights, add `torch.save` yourself after the fold that you care about.

## Train (full SST track)

```bash
mkdir -p models
python3 model_full_SST.py
```

Writes `models/best_{model_type}_model.pth`. Read [architecture.md](architecture.md) before quoting the printed test line.

## Regenerate derived tables

Only needed if you have the ZuCo `.mat` files or you changed averaging rules.

1. Put twelve Task-1 MATLAB files where `utils_ZuCo.get_matfiles('task1')` can see them. On Linux, change the `\\ZuCo_mat_data\\` prefix to a real path first.
2. `python3 read_ZuCo_mat.py` — per-subject sentence CSVs.
3. Point `get_average_sentence_level.py` at `ZuCo_et_csv_data` and run it.
4. Rebuild `ZuCo_SST_data/ssts_ZuCo.csv` with `convert_full_SST.py` if the text folders exist (`ZuCo_SST_data/all/{NEGATIVE,POSITIVE,NEUTRAL}`).
5. Join on `id` / `sentence_id` (pandas `merge`). The join that produced `combined_sst_et_*.csv` is not a committed script; `examples/inspect_datasets.py` shows the expected columns after a merge.
6. Word-level: run `ZuCo_et_csv_data/word/get_average.py` from that directory.
7. Full SST split: `SST_data/spilt.py` from inside `SST_data/`.

## Path pitfalls

| Script | Cwd it expects |
| --- | --- |
| `model_*.py` | repository root |
| `read_ZuCo_mat.py` | repository root |
| `convert_full_SST.py` | repository root |
| `get_average_sentence_level.py` | folder that contains `et_csv_data/` |
| `ZuCo_SST_data/spilt.py` | `ZuCo_SST_data/` |
| `SST_data/spilt.py` | `SST_data/` |
| `SST_data/convert_sst_to_et.py` | `SST_data/` |
| `ZuCo_et_csv_data/word/get_average.py` | `ZuCo_et_csv_data/word/` |
| `gaze_prediction/data/convert_zuco_data.py` | `gaze_prediction/data/` |
| `examples/*.py` | repository root |

## What is not reproducible from git alone

- Original ZuCo `.mat` recordings
- The folder of per-sentence `.txt` reviews (`ZuCo_SST_data/all/`)
- The gaze-prediction **model** that wrote `prediction_test_v2.csv` (only the predictions are stored)
- GPU-identical transformer runs (unseeded)

The committed CSVs are the reproduction surface for analysis and for the example scripts.
