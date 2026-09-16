# Reproduction notes

Personal checklist for getting back to a runnable state on a new machine. This clone is enough for documentation examples. Full training needs extra downloads.

## 1. Repository layout

Clone and stay at the root. Every training script and every `examples/*.py` uses paths relative to the repo root (`SST_data/...`, `ZuCo_SST_data/...`).

```bash
cd Transformer-Emotion-Analysis-with-Gaze
python3 examples/inspect_datasets.py
```

If `inspect_datasets.py` prints `ok` for every expected file, the snapshot is intact.

## 2. Python

Examples: **stdlib only** (Python 3.9+). Optional pretty tables use the same `csv` / `statistics` modules.

Training:

```text
torch
transformers
datasets
pandas
numpy
scikit-learn
tqdm
```

`nltk` is only required to *re-run* `SST_data/convert_sst_to_et.py`.  
`scipy` is only required to *re-run* `read_ZuCo_mat.py`.

See `requirements.txt` at the repo root. Versions are lower bounds that match a current personal laptop, not a locked research conda env.

## 3. Offline example run (no network, no GPU)

```bash
python3 examples/inspect_datasets.py
python3 examples/label_distribution.py
python3 examples/feature_stats.py
python3 examples/schema_validate.py
python3 examples/sentence_gaze_join.py
python3 examples/sample_rows.py
python3 examples/gaze_fusion_demo.py
```

Expected: all eight exit 0 (`run_all.py` wraps them). `schema_validate.py` fails non-zero if a required column is missing or a gaze cell is non-finite. This clone’s `python3 examples/run_all.py` completed with exit 0 after the example scripts were added.

## 4. Training Setting A (ZuCo CV)

Needs: `transformers` model downloads (`bert-base-uncased` and/or `roberta-base`) and enough RAM to fine-tune. GPU strongly recommended.

```bash
python3 -m pip install -r requirements.txt
python3 model_ZuCo_SST.py
```

Edit the `model_type` string at the top of the file to run the four variants. There is no CLI.

## 5. Training Setting B (full SST)

```bash
mkdir -p models
python3 model_full_SST.py
```

Creates `models/best_{model_type}_model.pth`. See `docs/experiments.md` before trusting the printed **test** line.

## 6. Regenerating CSVs (optional, incomplete)

### From ZuCo MATLAB

1. Obtain ZuCo Task 1 `.mat` files (official distribution).
2. Place 12 files in a directory the script can see.
3. Fix `get_matfiles`’s Windows `subdir` default or pass a POSIX path.
4. Point `read_ZuCo_mat.py` at `ZuCo_et_csv_data/` instead of `et_csv_data/`.
5. Re-average **by `id`**, not by row number, if `3_SR.csv` still has 299 rows.

### From labeled text folders

`ZuCo_SST_data/all/{NEGATIVE,POSITIVE,NEUTRAL}/*.txt` is not in git. If you still have that tree locally, `convert_full_SST.py` rebuilds `ssts_ZuCo.csv`.

### From SST

`SST_data/stts_all_sentence_level.csv` *is* in git (headerless). `convert_sst_to_et.py` rebuilds the zeroed word table. Filling those zeros requires the missing gaze predictor.

## 7. Hardware notes (personal)

| Workload | Rough need |
| --- | --- |
| `examples/*` | seconds, CPU, < 200 MB RAM |
| ZuCo 5-fold RoBERTa, batch 16, max_len 128 | one 8–12 GB GPU; several hours for all four `model_type`s |
| Full SST RoBERTa, batch 256, max_len 128 | 12+ GB GPU preferred; 5 epochs is short |

CPU training of RoBERTa on 9k sentences is possible and unpleasant.

## 8. What “reproduced” means here

| Claim | How to verify |
| --- | --- |
| File inventory matches the docs | `examples/inspect_datasets.py` |
| Label maps are 0/1/2 with published counts | `examples/label_distribution.py` |
| Gaze columns are finite and named as documented | `examples/schema_validate.py` |
| ZuCo text ids match averaged gaze ids | `examples/sentence_gaze_join.py` |
| Fusion tensor is 768 + 16 → 3 | `examples/gaze_fusion_demo.py` |
| Training loss decreases | run `model_*.py` yourself; not claimed in this commit |

## 9. Seeds

Only `StratifiedKFold(..., random_state=42)` and the two `train_test_split(..., random_state=42)` calls are seeded. PyTorch, NumPy, and CUDA are not. Exact loss curves will not match across machines even with the same CSV.

## 10. Privacy

Subject CSVs are **already de-identified** (integer subject index 1–12, no names). Do not add raw EEG, video, or consent forms to this personal repo.
