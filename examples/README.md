# Examples

Runnable personal checks for the CSVs already in this repo. They do **not**:

- download BERT / RoBERTa
- start `model_ZuCo_SST.py` or `model_full_SST.py` (those files train on import)
- call social APIs
- need MATLAB or the original ZuCo `.mat` dump

Install the small stack from the repo root:

```bash
python3 -m pip install -r requirements-examples.txt
python3 examples/run_all.py
```

Reports land in `examples/output/` (gitignored except `.gitkeep`).

## Scripts

| Script | What it does | Writes |
| --- | --- | --- |
| `inspect_datasets.py` | Row counts, columns, gaze min/max for every inventoried table | `dataset_inventory.md` |
| `schema_check.py` | Frozen schemas, key NaNs, fusion-column completeness | `schema_report.md` (exit 1 on error) |
| `label_distribution.py` | 0/1/2 histograms and majority-class accuracy | `label_distribution.md`, `.csv` |
| `gaze_feature_profiles.py` | Class-conditional means and high-L2 ZuCo sentences | `gaze_feature_profiles.md`, `.csv` |
| `aggregate_subjects.py` | Re-average 12 subject files **by sentence id** | `aggregate_subjects.md`, coverage + mean CSVs |
| `feature_correlations.py` | Pearson r on ZuCo / SST / predicted word gaze | `feature_correlations.md`, three corr CSVs |
| `split_sanity_check.py` | Disjoint holdouts, no label drift vs combined | `split_sanity_check.md` |
| `word_level_preview.py` | One ZuCo sentence's tokens; placeholder vs predicted SST | `word_level_preview.md` |
| `fusion_architecture_demo.py` | NumPy late-fusion head with 768+16→3 shapes | `fusion_architecture_demo.md` |
| `dummy_baseline.py` | Majority + gaze-only logreg (ZuCo 5-fold, SST holdout) | `dummy_baseline.md` |
| `run_all.py` | Runs the list above | `run_all_summary.txt` |

`common.py` is the path and column-order module. Fusion order is `nFixations|nFix, FFD, GPT, TRT, GD`, matching the training loaders, not the SST on-disk column order.

## Typical personal workflow

1. After touching a CSV, run `schema_check.py` and `split_sanity_check.py`.
2. Before quoting a fusion score, run `dummy_baseline.py` and write those floors next to the transformer numbers.
3. If subject 3's missing sentences worry you, read `aggregate_subjects.md` instead of re-running the original positional average.

Each script is safe to run on its own:

```bash
python3 examples/dummy_baseline.py
python3 examples/word_level_preview.py --sentence-id 3
```
