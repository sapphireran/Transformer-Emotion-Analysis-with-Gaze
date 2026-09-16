# Reproduction

Two layers of reproduction:

1. **Examples** — deterministic, no downloads, minutes on CPU. This is what CI-less personal machines can always run.
2. **Training** — needs PyTorch, `transformers`, and a download of `bert-base-uncased` or `roberta-base`. GPU optional but recommended.

## 1. Example suite (always do this first)

From the repo root:

```bash
python3 -m pip install -r requirements-examples.txt
python3 examples/run_all.py
```

`run_all.py` executes each example in a child process and writes:

- `examples/output/run_all_summary.txt`
- plus whatever each script writes (`dataset_inventory.md`, `schema_report.md`, …)

A successful run means the checked-in CSVs still match the documented schemas and that the dummy baselines can be fitted.

Individual scripts (all accept no required flags):

```bash
python3 examples/inspect_datasets.py
python3 examples/schema_check.py
python3 examples/label_distribution.py
python3 examples/gaze_feature_profiles.py
python3 examples/aggregate_subjects.py
python3 examples/feature_correlations.py
python3 examples/split_sanity_check.py
python3 examples/word_level_preview.py
python3 examples/fusion_architecture_demo.py
python3 examples/dummy_baseline.py
```

## 2. ZuCo fusion training

```bash
python3 -m pip install -r requirements.txt
python3 model_ZuCo_SST.py
```

Requirements that must already be true:

- `ZuCo_SST_data/combined_sst_et_standard.csv` present (it is)
- network access the first time `RobertaModel.from_pretrained('roberta-base')` runs
- enough RAM to hold five sequential fine-tunes (the script does not keep old folds)

To compare text-only vs fusion, change `model_type` and re-run. Do not change `random_state=42` if you want the same folds.

## 3. Full SST training

```bash
mkdir -p models
python3 model_full_SST.py
```

Requirements:

- the three `SST_data/*_full_sst.csv` files
- disk for `models/best_{model_type}_model.pth`
- patience: 5 epochs × ~37 train batches of 256 × 128 tokens

If you need a smoke test, temporarily set `num_epochs = 1` and `batch_size = 8` at the top of the file. Revert before logging a real score.

After the test-loop bug is fixed (see [known-issues.md](known-issues.md)), re-score a saved checkpoint by loading it in a short eval-only snippet rather than trusting an old log line.

## 4. Regenerating ZuCo subject CSVs

Only if you have the MATLAB dump:

1. Place 12 task-1 files under `ZuCo_mat_data/task1/` (or change `get_matfiles` to a POSIX path).
2. Point `read_ZuCo_mat.py` at `ZuCo_et_csv_data/` instead of `et_csv_data/`.
3. Run `read_ZuCo_mat.py`, then `get_average_sentence_level.py` with the same folder.
4. Join `standard_scaled_average_data.csv` to `ssts_ZuCo.csv` on `id` / `sentence_id`.
5. Diff against the checked-in `combined_sst_et_standard.csv`.

Without the `.mat` files, treat the checked-in CSVs as the snapshot of record.

## 5. Regenerating SST splits

```bash
cd SST_data
python3 spilt.py
```

This overwrites `train_full_sst.csv`, `valid_full_sst.csv`, and `test_full_sst.csv` from `combined_full_sst_et.csv` with `random_state=42`. Row counts should stay 9482 / 1185 / 1186.

## Seeds that are actually pinned

| Location | Seed | Effect |
| --- | --- | --- |
| `model_ZuCo_SST.py` | `StratifiedKFold(..., random_state=42)` | Fold membership |
| `SST_data/spilt.py` | `train_test_split(..., random_state=42)` | SST holdout |
| `ZuCo_SST_data/spilt.py` | `train_test_split(..., random_state=42)` | 320/40/40 files |
| PyTorch / NumPy / CUDA | **not set** | Weight init and shuffle differ across runs |

If you need bit-identical training, add explicit seeds before `get_model()` and pass a `Generator` into each `DataLoader`. That change is not in the training scripts today.

## Environment I used while writing these docs

- Python 3.12
- pandas / numpy / scikit-learn from `requirements-examples.txt`
- No GPU in the documentation pass
- Examples executed via `python3 examples/run_all.py`

Record your own `pip freeze` next to a training log if you publish a number.
