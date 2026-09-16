# Examples

Runnable personal notes. Every script reads the CSVs that are already in
git. None of them import `model_full_SST.py` or `model_ZuCo_SST.py`,
download a transformer, or need a GPU.

Install the light stack first:

```bash
python3 -m pip install -r requirements.txt
```

Run from the repository root so relative paths resolve. `--root` is
there if you cannot.

## Scripts

| Script | What it does |
| --- | --- |
| `01_inspect_datasets.py` | Row counts, columns, and notes for every documented table. |
| `02_label_and_length_profile.py` | Label balance and token-length stats for full SST and ZuCo 400. Also prints a majority-class baseline. |
| `03_gaze_feature_stats.py` | Per-class means, Pearson correlations, ANOVA F of gaze vs label. |
| `04_subject_variability.py` | How much the 12 ZuCo readers disagree on the same sentence. |
| `05_word_level_skip_rates.py` | Skip rate (`nFixations == 0`) by word length and by sentence polarity. |
| `06_feature_fusion_walkthrough.py` | Numpy clone of the 768 + 16 concat head. No PyTorch. |
| `07_split_leakage_check.py` | Asserts disjoint `sentence_id`s on the full-SST holdout. |
| `08_gaze_prediction_compare.py` | Schema and distribution notes: PROVO vs predicted SST tokens. |
| `09_export_analysis_tables.py` | Writes the tables above as CSV/JSON under `--output-dir`. |
| `10_zuco_index_alignment.py` | Proves subject 3 is reindexed; row-wise averages drift after id 149. |
| `run_all.py` | Runs 01–10 in order and writes `summary.json`. |

```bash
python3 examples/01_inspect_datasets.py
python3 examples/run_all.py --output-dir examples/output
```

`examples/output/` is gitignored. Regenerating it is the point of
`run_all.py`.

## Library

`examples/lib/` is imported by the scripts and by `tests/`. Public
entry points:

- `loading.documented_datasets` / `load_dataset`
- `gaze.per_class_means` / `feature_label_anova` / `inter_subject_cv`
- `fusion.FusionWalkthrough`
- `metrics.majority_baseline`

If you add a new checked-in CSV, register it in
`documented_datasets()` so `01` and the tests pick it up.

## What these examples are *not*

They are not a training tutorial and they do not claim that predicted
gaze is human eye tracking. See `docs/known_issues.md` before quoting
numbers from the original training scripts.
