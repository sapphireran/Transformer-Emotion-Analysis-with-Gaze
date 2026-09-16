# Examples

Personal inspection tooling for the CSVs in this checkout. Nothing here downloads BERT or RoBERTa. The package lives at `examples/gazekit/` so it cannot be confused with the original training scripts.

## Layout

```
examples/
  gazekit/                 # importable helpers
    schema.py              # label maps, canonical five channels
    paths.py               # repo-root path object
    io.py                  # CSV loaders + nFix → nFixations
    features.py            # balance, correlations, length bins
    splits.py              # leakage / class-drift reports
    baselines.py           # majority + logistic CV
    fusion.py              # NumPy late-fusion shapes
    metrics.py             # weighted + macro
    report.py              # stdout formatting
  scripts/
    01_explore_zuco_sst.py
    02_gaze_feature_report.py
    03_gaze_only_baseline.py
    04_split_sanity_check.py
    05_word_level_scanpath.py
    06_fusion_shape_check.py
```

## Run

From the repository root, after `pip install -r requirements.txt`:

```bash
python3 examples/scripts/01_explore_zuco_sst.py
python3 examples/scripts/02_gaze_feature_report.py
python3 examples/scripts/03_gaze_only_baseline.py
python3 examples/scripts/04_split_sanity_check.py
python3 examples/scripts/05_word_level_scanpath.py
python3 examples/scripts/06_fusion_shape_check.py
python3 -m pytest
```

Each script has a `build_report()` function that returns plain dicts / DataFrames so tests can import it without scraping stdout.

## What each script is for

| Script | Question it answers |
| --- | --- |
| `01_explore_zuco_sst.py` | Do the 400 combined rows match `ssts_ZuCo.csv`? What is the class prior? |
| `02_gaze_feature_report.py` | Are the five channels finite? How correlated is TRT with nFixations? Do class means move? |
| `03_gaze_only_baseline.py` | Is gaze above majority? Does length explain it? (scaler fit *inside* each fold) |
| `04_split_sanity_check.py` | Do the convenience CSVs leak ids? How far did class balance drift? |
| `05_word_level_scanpath.py` | Does nFixations rise with word length? What did readers linger on? |
| `06_fusion_shape_check.py` | Is the documented 5→16→784→3 math self-consistent? |

## Canonical gaze vector

Every loader that sees `nFix` renames it to `nFixations`. Downstream code should import `CANONICAL_GAZE` rather than hardcoding the five strings again.

```python
from gazekit import CANONICAL_GAZE
from gazekit.io import load_sentence_table
from gazekit.paths import default_paths

df = load_sentence_table(default_paths().zuco_combined_standard)
X = df[list(CANONICAL_GAZE)]
```

## Not in scope

- Fitting BERT / RoBERTa (`model_ZuCo_SST.py`, `model_full_SST.py`)
- Re-exporting `.mat` files
- Writing new combined CSVs back into `ZuCo_SST_data/`

Those steps are documented in `docs/reproduction.md`.

Measured numbers from a run against this checkout are recorded in [`docs/example-results.md`](../docs/example-results.md).
