# Reproducing locally (personal machine)

Three ladders. Each one assumes you are in the repo root.

## Ladder 0 — no extra packages (stdlib only)

A few examples import `examples.lib` and the Python standard library
only. After `pip` is available you will not need this path, but it is
the smoke test if a machine has nothing:

```bash
python3 -c "from pathlib import Path; print(Path('ZuCo_SST_data/combined_sst_et_standard.csv').stat().st_size)"
```

If that file is missing, you are not in this clone.

## Ladder 1 — docs/examples (recommended)

```bash
python3 -m pip install -r requirements-examples.txt
python3 examples/inspect_datasets.py
python3 examples/gaze_feature_stats.py --out examples/output
python3 examples/compare_scalings.py
python3 examples/split_balance.py
python3 examples/sentence_gaze_walkthrough.py --sentence-id 3
python3 examples/word_level_skip_analysis.py
python3 examples/subject_variability.py
python3 examples/toy_text_gaze_fusion.py --folds 5
python3 -m pytest tests/ -q
```

`examples/output/` is gitignored. A checked-in snapshot of one run lives
under `docs/sample-reports/` after the first successful execution.

Expected runtime on a laptop CPU: under a minute for the inspectors,
a bit longer for the 5-fold toy fusion.

## Ladder 2 — historical converters

Only if you have the missing raw inputs:

| Script | You also need |
| --- | --- |
| `read_ZuCo_mat.py` | `ZuCo_mat_data/task1/*.mat`, and a path fix |
| `convert_full_SST.py` | `ZuCo_SST_data/all/{NEGATIVE,NEUTRAL,POSITIVE}/*.txt` |
| `SST_data/convert_sst_to_et.py` | NLTK `punkt`; overwrites `sst_et_test.csv` |
| `get_average_sentence_level.py` | rename/copy into `et_csv_data` or edit the path |

You do **not** need this ladder to study the fusion math or the 400-row
tables.

## Ladder 3 — transformer trainers

```bash
python3 -m pip install -r requirements.txt
# GPU torch build recommended
python3 model_ZuCo_SST.py
python3 model_full_SST.py
```

Also:

1. Create `models/` before the full-SST script saves a checkpoint.
2. Patch the test loop (`extend` not `=`) before quoting test numbers.
3. Accept Hub downloads of `bert-base-uncased` / `roberta-base`.
4. Set seeds if you want more than “this run’s printout”.

---

## Import path for examples

Examples add the repo root to `sys.path` so `import examples.lib` works
when launched as `python3 examples/foo.py`. `pytest` collects from the
root and does the same via `tests/conftest.py`.

## Randomness

Inspectors are deterministic. `toy_text_gaze_fusion.py` uses
`random_state=42` for folds and logistic regression. Transformer
scripts do not seed `torch`.

## Disk

Do not commit `models/*.pth`, raw `.mat` files, or `examples/output/`.
`.gitignore` already lists them.
