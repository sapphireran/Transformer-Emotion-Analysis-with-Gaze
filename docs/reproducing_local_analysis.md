# Reproducing the local analysis (no GPU)

The `examples/` package is the part of this personal repo that should
run on a laptop with the CSVs already in git. It does not download
BERT, does not train, and does not write into `SST_data/` or `ZuCo_*`.

## Setup

```bash
python3 -m pip install -r requirements.txt
```

`requirements.txt` is pandas, numpy, and scikit-learn only.

## One-shot

From the repository root:

```bash
python3 examples/run_all.py --output-dir examples/output
```

This runs every numbered script in order, writes CSV/JSON summaries
under `examples/output/`, and prints a short report. `examples/output/`
is gitignored.

## One by one

```bash
python3 examples/01_inspect_datasets.py
python3 examples/02_label_and_length_profile.py
python3 examples/03_gaze_feature_stats.py
python3 examples/04_subject_variability.py
python3 examples/05_word_level_skip_rates.py
python3 examples/06_feature_fusion_walkthrough.py
python3 examples/07_split_leakage_check.py
python3 examples/08_gaze_prediction_compare.py
python3 examples/09_export_analysis_tables.py --output-dir examples/output
python3 examples/10_zuco_index_alignment.py --output-dir examples/output
```

Each script accepts `-h`. Most accept `--root` if you are not sitting
in the repo root.

## Tests

```bash
python3 -m unittest discover -s tests -v
```

The tests load the real CSVs (they are small enough) and check:

- expected row counts and required columns
- the integer label mapping `{0,1,2}`
- no `sentence_id` overlap across the full-SST splits
- fusion walkthrough output shapes
- skip rates in `[0, 1]`
- PROVO vs prediction schema differences

They do **not** download models and they do not call
`model_full_SST.py` / `model_ZuCo_SST.py`.

## What "success" looks like

- `01` prints a table of every documented CSV with row counts matching
  [datasets.md](datasets.md).
- `07` exits 0 and says the full-SST splits are disjoint.
- `10` shows subject 3 `SentLen` matching subject 1 only after remap.
- `06` prints logits of shape `(batch, 3)` and probabilities that sum
  to 1 per row.
- `run_all.py` writes `examples/output/summary.json` with
  `"ok": true`.

If a CSV is missing, the loader raises a `FileNotFoundError` that
names the path. That usually means you are not in the repo root.

## What this does *not* reproduce

- Transformer training or the numbers printed by the original scripts
- ZuCo `.mat` extraction
- The external gaze predictor that filled `prediction_test_v2.csv`
- The PNG pair-plots in `result/` (those are already rendered)

Training reproduction is [training_and_evaluation.md](training_and_evaluation.md).
